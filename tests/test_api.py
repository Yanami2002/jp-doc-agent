"""実 DB と模擬モデルで HTTP の契約・引用・失敗・接続の終了を確認する。"""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError

from jp_doc_agent.api import app as application
from jp_doc_agent.api import dependencies, routes
from jp_doc_agent.api.app import create_app
from jp_doc_agent.config import OpenAISettings
from jp_doc_agent.models import ChunkEmbedding, ChunkSource, DocumentChunk, DocumentPage


@pytest.fixture
def api_client(engine, tmp_path):
    with TestClient(create_app(engine=engine, report_dir=tmp_path / "reports")) as client:
        yield client


@pytest.fixture
def model_api(monkeypatch, encoder_factory, api_response, answer_response):
    state = {"calls": [], "clients": [], "draft": None, "failure": None, "during_model": None}
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def handler(request):
        body = json.loads(request.content)
        state["calls"].append((request.url.path, body))
        if state["failure"] == "timeout":
            raise httpx.ReadTimeout("test-key secret-value", request=request)
        if state["failure"] == "api_error":
            return httpx.Response(500, json={"error": {"message": "test-key secret-value"}})
        if request.url.path.endswith("embeddings"):
            return httpx.Response(200, json=api_response(body["input"]))
        if state["during_model"]:
            state["during_model"]()
        payload = json.loads(body["input"][1]["content"])
        if body["text"]["format"]["name"] == "research_step":
            draft = {
                "action": "finish",
                "query": None,
                "document_id": None,
                "page_number": None,
                "reason": "質問の年月を裏付ける追加資料が必要です。",
            }
        else:
            hit = payload["evidence"][0]
            draft = state["draft"] or {
                "status": "answered",
                "statements": [
                    {
                        "text": "対象条件が資料に記載されています。",
                        "citations": [{"chunk_id": hit["chunk_id"], "quote": hit["text"]}],
                    }
                ],
                "missing_information": [],
            }
        return httpx.Response(200, json=answer_response(draft))

    def build(_):
        encoder = encoder_factory(handler)
        state["clients"].append(encoder.client)
        return encoder

    monkeypatch.setattr(dependencies.OpenAIEncoder, "from_settings", build)
    return state


def test_read_endpoints_work_without_model_settings(api_client, text_document, monkeypatch):
    identifier = text_document(["対象者は学生です。", ""])

    def forbidden():
        pytest.fail("参照 API はモデル設定を読み込んではいけません。")

    monkeypatch.setattr(dependencies, "OpenAISettings", forbidden)
    health = api_client.get("/health")
    assert health.status_code == 200 and health.json()["status"] == "ok"
    assert health.json()["vector_distance"] == 1.0
    documents = api_client.get("/documents")
    assert documents.status_code == 200 and documents.json()[0]["id"] == identifier
    page = api_client.get(f"/documents/{identifier}/pages/1")
    assert page.status_code == 200 and page.json()["text"] == "対象者は学生です。"
    blank = api_client.get(f"/documents/{identifier}/pages/2")
    assert blank.status_code == 200 and blank.json()["text"] == ""
    assert "file_path" not in documents.text and "POSTGRES_PASSWORD" not in health.text


def test_ask_returns_validated_cross_page_citations_and_saves_report(
    engine, api_client, ready_document, model_api, tmp_path
):
    identifier = ready_document(["対象者は学生です。", "期限は三月です。"])
    with engine.connect() as connection:
        before = {
            model: connection.scalar(select(func.count()).select_from(model))
            for model in (DocumentPage, DocumentChunk, ChunkSource, ChunkEmbedding)
        }
    response = api_client.post("/ask", json={"question": "申請条件は？", "document_id": identifier})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "answered" and "mode" not in result
    assert result["citations"][0]["page_numbers"] == [1, 2]
    assert result["citations"][0]["quote"] == "対象者は学生です。\n期限は三月です。"
    assert result["request_id"] == response.headers["X-Request-ID"]
    assert result["usage"]["answer_total_tokens"] == 70
    assert result["counts"]["searches"] == 1
    assert [event["name"] for event in result["trace"]] == ["search", "assess_and_answer"]
    report = tmp_path / "reports" / f"api-{result['request_id']}.json"
    assert json.loads(report.read_text()) == result
    assert all(client.is_closed() for client in model_api["clients"])
    with engine.connect() as connection:
        after = {
            model: connection.scalar(select(func.count()).select_from(model)) for model in before
        }
    assert before == after


def test_insufficient_evidence_is_http_success(api_client, ready_document, model_api):
    ready_document(["2024年度の資料です。"])
    model_api["draft"] = {
        "status": "insufficient_evidence",
        "statements": [],
        "missing_information": ["2028年度の本文。"],
    }
    response = api_client.post("/ask", json={"question": "2028年度の売上は？"})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "insufficient_evidence"
    assert result["statements"] == result["citations"] == []
    assert result["missing_information"] == ["2028年度の本文。"]
    assert result["stop_reason"] == "no_further_evidence"


@pytest.mark.parametrize(
    "payload",
    [
        {"question": " "},
        {"question": "🙂" * 8192},
        {"question": "条件は？", "top_k": 21},
        {"question": "条件は？", "top_k": "5"},
        {"question": "条件は？", "document_id": 0},
        {"question": "条件は？", "api_key": "test-key secret-value"},
        {"question": "条件は？", "mode": "baseline"},
        {"question": "条件は？", "mode": "agent"},
    ],
)
def test_invalid_json_input_never_creates_model_client_or_echoes_body(
    api_client, model_api, payload
):
    response = api_client.post("/ask", json=payload)
    assert response.status_code == 422 and response.json()["error"]["code"] == "invalid_request"
    assert response.json()["request_id"] == response.headers["X-Request-ID"]
    assert not model_api["calls"] and not model_api["clients"]
    assert "test-key" not in response.text and "secret-value" not in response.text


def test_missing_document_and_page_are_404_without_model_calls(api_client, model_api):
    answer = api_client.post("/ask", json={"question": "条件は？", "document_id": 999999})
    assert answer.status_code == 404 and answer.json()["error"]["code"] == "document_not_found"
    page = api_client.get("/documents/999999/pages/1")
    assert page.status_code == 404 and page.json()["error"]["code"] == "page_not_found"
    invalid = api_client.get("/documents/1/pages/0")
    assert invalid.status_code == 422
    assert not model_api["calls"] and not model_api["clients"]


def test_missing_key_is_503_but_database_routes_still_work(api_client, ready_document, monkeypatch):
    ready_document(["対象者は学生です。"])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(dependencies, "OpenAISettings", lambda: OpenAISettings(_env_file=None))
    response = api_client.post("/ask", json={"question": "条件は？"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_configuration_error"
    assert api_client.get("/health").status_code == 200
    assert api_client.get("/documents").status_code == 200


@pytest.mark.parametrize("failure", ["api_error", "timeout", "citation"])
def test_model_failures_are_safe_and_execution_is_saved(
    api_client, ready_document, model_api, tmp_path, failure
):
    ready_document(["対象者は学生です。"])
    model_api["failure"] = failure
    if failure == "citation":
        model_api["draft"] = {
            "status": "answered",
            "statements": [
                {"text": "未検証の結論。", "citations": [{"chunk_id": -1, "quote": "原文"}]}
            ],
            "missing_information": [],
        }
    response = api_client.post("/ask", json={"question": "条件は？"})
    assert response.status_code == 502
    result = response.json()
    assert result["error"]["code"] == "model_error"
    assert "未検証の結論" not in response.text and "test-key" not in response.text
    assert "secret-value" not in response.text
    report = json.loads((tmp_path / "reports" / f"api-{result['request_id']}.json").read_text())
    assert report["status"] == "error" and report["workflow"] == "agent"
    assert "mode" not in report
    assert "test-key" not in json.dumps(report)
    assert result["trace"][-1]["status"] == "error"
    assert all(client.is_closed() for client in model_api["clients"])


def test_database_error_does_not_expose_connection_details(api_client, monkeypatch):
    def fail(_):
        raise OperationalError("SELECT secret-value", {}, Exception("test-key password"))

    monkeypatch.setattr(routes, "list_documents", fail)
    response = api_client.get("/documents")
    assert (
        response.status_code == 503 and response.json()["error"]["code"] == "database_unavailable"
    )
    assert "secret-value" not in response.text and "test-key" not in response.text


def test_agent_database_error_keeps_trace_and_uses_503(
    api_client, model_api, monkeypatch, tmp_path
):
    def fail(*args, **kwargs):
        raise OperationalError("SELECT secret-value", {}, Exception("test-key password"))

    monkeypatch.setattr("jp_doc_agent.agent.tools.search", fail)
    response = api_client.post("/ask", json={"question": "条件は？"})
    assert response.status_code == 503
    result = response.json()
    assert result["error"]["code"] == "database_unavailable"
    assert result["trace"][-1]["status"] == "error"
    report = json.loads((tmp_path / "reports" / f"api-{result['request_id']}.json").read_text())
    assert report["error_type"] == "database" and report["counts"]["model_calls"] == 0
    assert "secret-value" not in response.text and "test-key" not in response.text
    assert not model_api["calls"] and all(client.is_closed() for client in model_api["clients"])


@pytest.mark.parametrize(
    "setting,value", [("OPENAI_ANSWER_MODEL", " "), ("OPENAI_ANSWER_MAX_OUTPUT_TOKENS", "0")]
)
def test_bad_model_configuration_is_503_without_api_call(
    api_client, model_api, monkeypatch, setting, value
):
    monkeypatch.setenv(setting, value)
    response = api_client.post("/ask", json={"question": "条件は？"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_configuration_error"
    assert not model_api["calls"] and all(client.is_closed() for client in model_api["clients"])


def test_report_failure_is_explicit_after_model_call(
    api_client, ready_document, model_api, monkeypatch
):
    ready_document(["対象者は学生です。"])

    def fail(*args):
        raise PermissionError("/private/secret-value")

    monkeypatch.setattr(routes, "write_report", fail)
    response = api_client.post("/ask", json={"question": "条件は？"})
    assert response.status_code == 503 and response.json()["error"]["code"] == "report_unavailable"
    assert "secret-value" not in response.text and "answer" not in response.json()
    assert model_api["calls"] and all(client.is_closed() for client in model_api["clients"])


def test_health_responds_while_question_waits_for_model(api_client, ready_document, model_api):
    ready_document(["対象者は学生です。"])
    entered, release = Event(), Event()

    def wait_for_release():
        entered.set()
        assert release.wait(timeout=5)

    model_api["during_model"] = wait_for_release
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(api_client.post, "/ask", json={"question": "条件は？"})
        try:
            assert entered.wait(timeout=5)
            assert api_client.get("/health").status_code == 200
            assert not pending.done()
        finally:
            release.set()
        assert pending.result(timeout=5).status_code == 200


def test_application_owns_only_engine_created_at_startup(engine, monkeypatch, tmp_path):
    disposed = []
    monkeypatch.setattr(application, "create_database_engine", lambda _: engine)
    original = engine.dispose
    monkeypatch.setattr(engine, "dispose", lambda: disposed.append(True))
    try:
        with TestClient(create_app(report_dir=tmp_path)) as client:
            assert client.get("/health").status_code == 200
        assert disposed == [True]
        with TestClient(create_app(engine=engine, report_dir=tmp_path)) as client:
            assert client.get("/health").status_code == 200
        assert disposed == [True]
    finally:
        monkeypatch.setattr(engine, "dispose", original)


def test_openapi_describes_typed_success_and_error_responses(api_client):
    schema = api_client.get("/openapi.json").json()
    assert set(schema["paths"]) == {
        "/health",
        "/documents",
        "/documents/{document_id}/pages/{page_number}",
        "/ask",
    }
    request = schema["components"]["schemas"]["AskRequest"]
    assert set(request["properties"]) == {"question", "document_id", "top_k"}
    assert "mode" not in schema["components"]["schemas"]["AskResponse"]["properties"]
    assert request["additionalProperties"] is False
    error = schema["paths"]["/ask"]["post"]["responses"]["422"]
    assert error["content"]["application/json"]["schema"]["$ref"].endswith("ErrorResponse")
    assert api_client.get("/docs").status_code == 200


def test_unknown_routes_methods_and_internal_errors_share_error_contract(api_client, monkeypatch):
    unknown = api_client.get("/missing")
    assert unknown.status_code == 404 and unknown.json()["error"]["code"] == "not_found"
    method = api_client.get("/ask")
    assert method.status_code == 405 and method.headers["Allow"] == "POST"

    def fail(_):
        raise RuntimeError("test-key secret-value")

    monkeypatch.setattr(routes, "list_documents", fail)
    # Starlette は 500 応答後に例外を再送出するため、HTTP の本文を確認する。
    with TestClient(api_client.app, raise_server_exceptions=False) as client:
        response = client.get("/documents")
    assert response.status_code == 500 and response.json()["error"]["code"] == "internal_error"
    assert response.json()["request_id"] == response.headers["X-Request-ID"]
    assert "test-key" not in response.text and "secret-value" not in response.text
