"""実ネットワークを使わず OpenAI SDK の応答検証とリトライを検証する。"""

import json

import httpx
import pytest

from jp_doc_agent.config import EMBEDDING_MODEL
from jp_doc_agent.embedding.encoder import MAX_QUERY_TOKENS, EmbeddingError


def test_api_keeps_raw_text_and_reorders_vectors_by_input_index(api_response, encoder_factory):
    texts = ["学生の申請条件です。", "期限は来週です。"]

    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/embeddings"
        assert body["input"] == texts
        assert body["model"] == EMBEDDING_MODEL
        assert body["dimensions"] == 1536
        assert body["encoding_format"] == "float"
        response = api_response(texts)
        response["data"].reverse()
        return httpx.Response(200, json=response)

    result = encoder_factory(handler).encode(texts)
    assert result.vectors[0][0] == result.vectors[1][1] == 1.0
    assert result.vectors[0][1] == result.vectors[1][0] == 0.0
    assert result.total_tokens == 2


@pytest.mark.parametrize("problem", ["model", "count", "index", "dimension", "zero", "missing"])
def test_invalid_api_response_is_rejected(api_response, encoder_factory, problem):
    def handler(request):
        response = api_response(["本文。"])
        if problem == "model":
            response["model"] = "another-model"
        elif problem == "count":
            response["data"] = []
        elif problem == "index":
            response["data"][0]["index"] = 1
        elif problem == "dimension":
            response["data"][0]["embedding"] = [1.0]
        elif problem == "zero":
            response["data"][0]["embedding"] = [0.0] * 1536
        else:
            del response["usage"]
        return httpx.Response(200, json=response)

    with pytest.raises(EmbeddingError, match="不正"):
        encoder_factory(handler).encode(["本文。"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_vectors_are_rejected(api_response, encoder_factory, value):
    def handler(request):
        response = api_response(["本文。"])
        response["data"][0]["embedding"][0] = value
        # HTTP JSON ライブラリの制約を避け、不正な外部 JSON の読み込みを再現する。
        return httpx.Response(200, content=json.dumps(response))

    with pytest.raises(EmbeddingError, match="不正"):
        encoder_factory(handler).encode(["本文。"])


@pytest.mark.parametrize("status", [429, 500])
def test_sdk_retries_transient_errors(api_response, encoder_factory, status):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(
                status, json={"error": {"message": "temporary"}}, headers={"retry-after-ms": "1"}
            )
        return httpx.Response(200, json=api_response(["本文。"]))

    assert encoder_factory(handler, retries=2).encode(["本文。"]).vectors
    assert len(calls) == 3


def test_auth_error_does_not_retry_or_expose_response_body(encoder_factory):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401, json={"error": {"message": "secret-value test-key"}})

    with pytest.raises(EmbeddingError, match="HTTP 401") as error:
        encoder_factory(handler, retries=2).encode(["本文。"])
    assert "secret-value" not in str(error.value)
    assert "test-key" not in str(error.value)
    assert len(calls) == 1


@pytest.mark.parametrize("texts", [[], [" "], ["あ" * 301], ["本文"] * 33])
def test_invalid_input_never_calls_api(encoder_factory, texts):
    def handler(request):
        pytest.fail("不正な入力で API を呼び出してはいけません。")

    with pytest.raises(ValueError):
        encoder_factory(handler).encode(texts)


def test_query_can_exceed_chunk_limit_and_keeps_original_text(api_response, encoder_factory):
    query = "あ" * 301

    def handler(request):
        assert json.loads(request.content)["input"] == [query]
        return httpx.Response(200, json=api_response([query]))

    assert encoder_factory(handler).encode_query(query).vectors[0][0] == 1.0


@pytest.mark.parametrize("query", ["", " \n", "あ" * (MAX_QUERY_TOKENS + 1)])
def test_invalid_query_never_calls_api(encoder_factory, query):
    def handler(request):
        pytest.fail("不正な質問で API を呼び出してはいけません。")

    with pytest.raises(ValueError, match="質問"):
        encoder_factory(handler).encode_query(query)
