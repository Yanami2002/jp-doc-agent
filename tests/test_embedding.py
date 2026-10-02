"""Embedding API の応答・保存・再開・競合を確認する。"""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import httpx
import pytest
from sqlalchemy import delete, event, func, select, update

from jp_doc_agent import cli
from jp_doc_agent.chunking.service import chunk_document
from jp_doc_agent.chunking.splitter import ChunkingConfig
from jp_doc_agent.config import EMBEDDING_MODEL, OpenAISettings
from jp_doc_agent.embedding.encoder import MAX_QUERY_TOKENS, EmbeddingError, input_hash, profile_id
from jp_doc_agent.embedding.service import embed_document, embed_documents, embedding_status
from jp_doc_agent.models import ChunkEmbedding, Document, DocumentChunk, EmbeddingProfile


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


@pytest.fixture
def mock_encoder(encoder_factory, api_response):
    state = {"calls": [], "handler": None}

    def handler(request):
        body = json.loads(request.content)
        state["calls"].append(body["input"])
        if state["handler"]:
            return state["handler"](request)
        return httpx.Response(200, json=api_response(body["input"]))

    return encoder_factory(handler), state


def test_vectors_round_trip_batches_and_repeat_skips_api(engine, text_document, mock_encoder):
    identifier = text_document(["対象者は学生です。期限を確認してください。" * 30])
    chunk_document(engine, identifier, ChunkingConfig(40, 8))
    encoder, state = mock_encoder
    before = embedding_status(engine, document_id=identifier)
    assert before["embedded"] == 0
    assert before["pending"] > 2
    result = embed_document(engine, encoder, identifier, batch_size=2)
    assert result["status"] == "embedded"
    assert result["embedded"] == before["chunks"]
    assert all(1 <= len(batch) <= 2 for batch in state["calls"])
    with engine.connect() as connection:
        rows = (
            connection.execute(select(ChunkEmbedding, DocumentChunk.text).join(DocumentChunk))
            .mappings()
            .all()
        )
        assert connection.scalar(select(func.count()).select_from(EmbeddingProfile)) == 1
        query = [1.0] + [0.0] * 1535
        assert connection.scalar(
            select(func.min(ChunkEmbedding.embedding.cosine_distance(query)))
        ) == pytest.approx(0.0)
    for row in rows:
        assert len(row["embedding"]) == 1536
        assert row["input_hash"] == input_hash(row["text"])
        assert row["profile_id"] == profile_id()
    timestamps = {row["chunk_id"]: row["created_at"] for row in rows}
    call_count = len(state["calls"])
    repeat = embed_document(engine, encoder, identifier)
    assert repeat["status"] == "duplicate"
    assert repeat["skipped"] == before["chunks"]
    assert repeat["api_requests"] == 0
    assert len(state["calls"]) == call_count
    after = embedding_status(engine, document_id=identifier)
    assert after["pending"] == 0
    assert after["embedded"] == before["chunks"]
    with engine.connect() as connection:
        assert (
            dict(
                connection.execute(select(ChunkEmbedding.chunk_id, ChunkEmbedding.created_at)).all()
            )
            == timestamps
        )


def test_failed_batch_keeps_completed_batches_and_resumes(
    engine, text_document, mock_encoder, api_response
):
    identifier = text_document(["申請条件を確認してください。" * 40])
    chunk_document(engine, identifier, ChunkingConfig(40, 8))
    encoder, state = mock_encoder

    def fail_second(request):
        if len(state["calls"]) == 2:
            return httpx.Response(429, json={"error": {"message": "limited"}})
        return httpx.Response(200, json=api_response(json.loads(request.content)["input"]))

    state["handler"] = fail_second
    first = embed_document(engine, encoder, identifier, batch_size=2)
    assert first["status"] == "failed"
    assert first["embedded"] == 2
    assert embedding_status(engine, document_id=identifier)["embedded"] == 2
    state["handler"] = None
    second = embed_document(engine, encoder, identifier, batch_size=2)
    assert second["status"] == "embedded"
    assert second["skipped"] == 2
    assert second["embedded"] == second["chunks"] - 2
    assert embedding_status(engine, document_id=identifier)["pending"] == 0


def test_chunk_rebuild_and_document_delete_cascade_vectors(engine, text_document, mock_encoder):
    identifier = text_document(["資料の本文です。" * 30])
    chunk_document(engine, identifier, ChunkingConfig(80, 10))
    encoder, _ = mock_encoder
    embed_document(engine, encoder, identifier)
    assert embedding_status(engine)["embedded"] > 0
    chunk_document(engine, identifier, ChunkingConfig(40, 8))
    assert embedding_status(engine)["embedded"] == 0
    assert embedding_status(engine)["pending"] > 0
    embed_document(engine, encoder, identifier)
    with engine.begin() as connection:
        connection.execute(delete(Document).where(Document.id == identifier))
        assert connection.scalar(select(func.count()).select_from(ChunkEmbedding)) == 0


def test_changed_chunk_text_is_detected_by_hash(engine, text_document, mock_encoder):
    identifier = text_document(["申請条件です。"])
    chunk_document(engine, identifier, ChunkingConfig())
    encoder, state = mock_encoder
    embed_document(engine, encoder, identifier)
    with engine.begin() as connection:
        connection.execute(
            update(DocumentChunk)
            .where(DocumentChunk.document_id == identifier)
            .values(text="申請期限です。")
        )
    assert embedding_status(engine)["pending"] == 1
    assert embed_document(engine, encoder, identifier)["embedded"] == 1
    assert len(state["calls"]) == 2
    assert embedding_status(engine)["pending"] == 0


def test_source_rebuild_during_api_discards_old_vectors(
    engine, text_document, mock_encoder, api_response
):
    identifier = text_document(["対象者は学生です。" * 30])
    chunk_document(engine, identifier, ChunkingConfig())
    encoder, state = mock_encoder

    def rebuild(request):
        chunk_document(engine, identifier, ChunkingConfig(40, 8))
        return httpx.Response(200, json=api_response(json.loads(request.content)["input"]))

    state["handler"] = rebuild
    result = embed_document(engine, encoder, identifier)
    assert result["status"] == "failed"
    assert "変更" in result["reason"]
    assert result["embedded"] == 0
    assert embedding_status(engine)["embedded"] == 0


def test_database_failure_rolls_back_only_current_batch(engine, text_document, mock_encoder):
    identifier = text_document(["本文です。" * 80])
    chunk_document(engine, identifier, ChunkingConfig(40, 8))
    encoder, _ = mock_encoder
    writes = []

    def fail_second(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO chunk_embeddings"):
            writes.append(statement)
            if len(writes) == 2:
                raise RuntimeError("simulated persistence failure")

    event.listen(engine, "before_cursor_execute", fail_second)
    try:
        with pytest.raises(RuntimeError, match="simulated"):
            embed_document(engine, encoder, identifier, batch_size=2)
    finally:
        event.remove(engine, "before_cursor_execute", fail_second)
    assert embedding_status(engine)["embedded"] == 2
    assert embed_document(engine, encoder, identifier, batch_size=2)["skipped"] == 2
    assert embedding_status(engine)["pending"] == 0


def test_concurrent_runs_do_not_overwrite_or_duplicate_vectors(
    engine, text_document, encoder_factory, api_response
):
    identifier = text_document(["日本語本文です。"])
    chunk_document(engine, identifier, ChunkingConfig())
    barrier = Barrier(2)

    def handler(request):
        barrier.wait(timeout=5)
        return httpx.Response(200, json=api_response(json.loads(request.content)["input"]))

    encoders = [encoder_factory(handler), encoder_factory(handler)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda encoder: embed_document(engine, encoder, identifier), encoders)
        )
    assert sorted(result["status"] for result in results) == ["duplicate", "embedded"]
    assert embedding_status(engine)["embedded"] == 1


def test_unprocessed_empty_and_missing_documents(engine, text_document, mock_encoder):
    identifier = text_document([""])
    encoder, state = mock_encoder
    assert embed_document(engine, encoder, identifier)["status"] == "failed"
    chunk_document(engine, identifier, ChunkingConfig())
    assert embed_document(engine, encoder, identifier)["status"] == "empty"
    assert embed_documents(engine, encoder, document_id=identifier + 1)[0]["status"] == "failed"
    with pytest.raises(ValueError, match="文書"):
        embedding_status(engine, document_id=identifier + 1)
    with pytest.raises(ValueError, match="batch-size"):
        embed_documents(engine, encoder, batch_size=0)
    assert not state["calls"]


def test_cli_embedding_and_status(engine, text_document, mock_encoder, monkeypatch, capsys):
    identifier = text_document(["日本語の本文です。"])
    chunk_document(engine, identifier, ChunkingConfig())
    encoder, _ = mock_encoder
    monkeypatch.setattr(cli, "create_database_engine", lambda _: engine)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(cli.OpenAIEncoder, "from_settings", lambda _: encoder)
    monkeypatch.setattr(
        "sys.argv", ["jp-doc-agent", "embed-chunks", "--document-id", str(identifier)]
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["embedded"] == 1
    monkeypatch.setattr("sys.argv", ["jp-doc-agent", "embedding-status"])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["pending"] == 0


def test_cli_missing_key_is_safe_and_existing_commands_do_not_need_key(engine, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(cli, "OpenAISettings", lambda: OpenAISettings(_env_file=None))
    monkeypatch.setattr(cli, "create_database_engine", lambda _: engine)
    monkeypatch.setattr("sys.argv", ["jp-doc-agent", "embed-chunks"])
    assert cli.main() == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err
    monkeypatch.setattr("sys.argv", ["jp-doc-agent", "embedding-status"])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["chunks"] == 0
