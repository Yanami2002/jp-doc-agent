"""模擬 API と実際の PostgreSQL で順位・検索範囲・出典・並行更新を検証する。"""

import json

import httpx
import pytest
from sqlalchemy import delete, event, insert, select, update

from jp_doc_agent import cli
from jp_doc_agent.chunking.service import chunk_document
from jp_doc_agent.chunking.splitter import ChunkingConfig
from jp_doc_agent.config import OpenAISettings
from jp_doc_agent.embedding.encoder import EmbeddingError, profile_id
from jp_doc_agent.embedding.service import embed_document
from jp_doc_agent.models import (
    ChunkEmbedding,
    ChunkSource,
    DocumentChunk,
    DocumentPage,
    EmbeddingProfile,
)
from jp_doc_agent.retrieval.service import search


@pytest.fixture
def ready_document(engine, text_document, encoder_factory, api_response):
    def build(pages, *, vector=(1.0, 0.0)):
        identifier = text_document(pages)
        chunk_document(engine, identifier, ChunkingConfig())

        def handler(request):
            response = api_response(json.loads(request.content)["input"])
            for item in response["data"]:
                item["embedding"] = list(vector) + [0.0] * (1536 - len(vector))
            return httpx.Response(200, json=response)

        result = embed_document(engine, encoder_factory(handler), identifier)
        assert result["status"] == "embedded"
        return identifier

    return build


@pytest.fixture
def query_encoder(encoder_factory, api_response):
    state = {"calls": [], "during_api": None}

    def handler(request):
        body = json.loads(request.content)
        state["calls"].append(body)
        if state["during_api"]:
            state["during_api"]()
        return httpx.Response(200, json=api_response(body["input"]))

    return encoder_factory(handler), state


def test_cosine_ranking_top_k_filter_and_no_query_persistence(
    engine, ready_document, query_encoder
):
    lower = ready_document(["資料 A。"], vector=(0.6, 0.8))
    best = ready_document(["資料 B。"], vector=(2.0, 0.0))
    last = ready_document(["資料 C。"], vector=(-1.0, 0.0))
    encoder, state = query_encoder
    with engine.connect() as connection:
        before = connection.execute(
            select(ChunkEmbedding.chunk_id, ChunkEmbedding.profile_id, ChunkEmbedding.created_at)
        ).all()
    result = search(engine, encoder, "対象者を教えてください。", top_k=2)
    assert [hit["document_id"] for hit in result["results"]] == [best, lower]
    assert [hit["rank"] for hit in result["results"]] == [1, 2]
    assert [hit["cosine_distance"] for hit in result["results"]] == pytest.approx([0.0, 0.4])
    assert [hit["cosine_similarity"] for hit in result["results"]] == pytest.approx([1.0, 0.6])
    assert result["coverage"] == {
        "chunks": 3,
        "searchable": 3,
        "pending": 0,
        "unchunked_documents": 0,
    }
    assert result["api_tokens"] == 1
    assert state["calls"][0]["input"] == [result["query"]]
    scoped = search(engine, encoder, "条件は？", document_id=last)
    assert [hit["document_id"] for hit in scoped["results"]] == [last]
    assert scoped["results"][0]["cosine_distance"] == pytest.approx(2.0)
    assert scoped["coverage"]["chunks"] == 1
    with engine.connect() as connection:
        after = connection.execute(
            select(ChunkEmbedding.chunk_id, ChunkEmbedding.profile_id, ChunkEmbedding.created_at)
        ).all()
    assert after == before


def test_cross_page_sources_are_complete_and_match_original(engine, ready_document, query_encoder):
    identifier = ready_document(["対象者は学生です。", "期限は三月です。"])
    encoder, _ = query_encoder
    hit = search(engine, encoder, "申請条件は？", document_id=identifier)["results"][0]
    assert hit["text"] == "対象者は学生です。\n期限は三月です。"
    assert hit["page_numbers"] == [1, 2]
    assert hit["title"] == "日本語資料"
    assert hit["source_url"] == hit["resolved_url"] == "https://example.com/document.pdf"
    with engine.connect() as connection:
        pages = dict(
            connection.execute(
                select(DocumentPage.page_number, DocumentPage.text).where(
                    DocumentPage.document_id == identifier
                )
            ).all()
        )
    for source in hit["sources"]:
        assert (
            pages[source["page_number"]][source["start_char"] : source["end_char"]]
            == (hit["text"][source["chunk_start_char"] : source["chunk_end_char"]])
        )


def test_stale_and_other_profile_vectors_are_excluded_before_limit(
    engine, ready_document, query_encoder
):
    stale = ready_document(["条件は学生です。"])
    valid = ready_document(["期限は明日です。"], vector=(0.0, 1.0))
    with engine.begin() as connection:
        connection.execute(
            update(DocumentChunk)
            .where(DocumentChunk.document_id == stale)
            .values(text="条件は教員です。")
        )
        profile = dict(connection.execute(select(EmbeddingProfile)).mappings().one())
        profile["id"] = "other-profile"
        profile["model"] = "another-model"
        connection.execute(insert(EmbeddingProfile), profile)
        embedding = dict(
            connection.execute(
                select(ChunkEmbedding).join(DocumentChunk).where(DocumentChunk.document_id == valid)
            )
            .mappings()
            .one()
        )
        embedding.update(profile_id="other-profile", embedding=[1.0] + [0.0] * 1535)
        connection.execute(insert(ChunkEmbedding), embedding)
    encoder, _ = query_encoder
    result = search(engine, encoder, "条件は？", top_k=1)
    assert [hit["document_id"] for hit in result["results"]] == [valid]
    assert result["results"][0]["cosine_distance"] == pytest.approx(1.0)
    assert result["coverage"]["searchable"] == result["coverage"]["pending"] == 1
    assert result["profile_id"] == profile_id()


@pytest.mark.parametrize("options", [{"query": " "}, {"top_k": 0}, {"top_k": 101}])
def test_invalid_options_never_call_api(engine, query_encoder, options):
    encoder, state = query_encoder
    with pytest.raises(ValueError):
        search(engine, encoder, **({"query": "条件は？"} | options))
    assert not state["calls"]


def test_missing_unprocessed_and_stale_only_documents_never_call_api(
    engine, text_document, ready_document, query_encoder
):
    encoder, state = query_encoder
    with pytest.raises(ValueError, match="ベクトル"):
        search(engine, encoder, "条件は？")
    identifier = text_document(["未処理の本文。"])
    with pytest.raises(ValueError, match="文書"):
        search(engine, encoder, "条件は？", document_id=identifier + 1)
    with pytest.raises(ValueError, match="ベクトル"):
        search(engine, encoder, "条件は？", document_id=identifier)
    ready = ready_document(["分割済みの本文。"])
    chunk_document(engine, identifier, ChunkingConfig())
    result = search(engine, encoder, "条件は？")
    assert result["coverage"]["pending"] == 1
    assert result["coverage"]["unchunked_documents"] == 0
    with engine.begin() as connection:
        connection.execute(update(ChunkEmbedding).values(input_hash="0" * 64))
    with pytest.raises(ValueError, match="ベクトル"):
        search(engine, encoder, "条件は？", document_id=ready)
    assert len(state["calls"]) == 1


def test_partial_corpus_reports_unchunked_documents(
    engine, text_document, ready_document, query_encoder
):
    ready_document(["分割済み資料。"])
    text_document(["未分割資料。"])
    encoder, _ = query_encoder
    assert search(engine, encoder, "対象は？")["coverage"]["unchunked_documents"] == 1


def test_equal_distances_have_stable_document_and_chunk_order(
    engine, ready_document, query_encoder
):
    identifiers = [ready_document([text]) for text in ("資料一。", "資料二。")]
    encoder, _ = query_encoder
    for _ in range(2):
        result = search(engine, encoder, "対象は？")
        assert [hit["document_id"] for hit in result["results"]] == identifiers


def test_rebuild_during_api_does_not_return_old_chunks(engine, ready_document, query_encoder):
    identifier = ready_document(["対象者を確認してください。" * 30])
    encoder, state = query_encoder
    state["during_api"] = lambda: chunk_document(engine, identifier, ChunkingConfig(40, 8))
    with pytest.raises(ValueError, match="ベクトル"):
        search(engine, encoder, "対象は？", document_id=identifier)


def test_sources_use_same_snapshot_when_chunks_are_rebuilt_after_ranking(
    engine, ready_document, query_encoder
):
    identifier = ready_document(["条件を確認してください。", "期限を守ってください。"])
    encoder, _ = query_encoder
    rebuilt = []

    def rebuild(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("SELECT document_chunks.id AS chunk_id") and not rebuilt:
            rebuilt.append(True)
            chunk_document(engine, identifier, ChunkingConfig(15, 3))

    event.listen(engine, "after_cursor_execute", rebuild)
    try:
        result = search(engine, encoder, "条件は？", document_id=identifier)
    finally:
        event.remove(engine, "after_cursor_execute", rebuild)
    assert rebuilt
    assert result["results"][0]["page_numbers"] == [1, 2]
    assert len(result["results"][0]["sources"]) == 2


def test_corrupt_page_source_is_not_returned(engine, ready_document, query_encoder):
    identifier = ready_document(["対象者は学生です。"])
    with engine.begin() as connection:
        connection.execute(
            update(DocumentPage)
            .where(DocumentPage.document_id == identifier)
            .values(text="対象者は教員です。")
        )
    encoder, _ = query_encoder
    with pytest.raises(ValueError, match="出典が本文と一致"):
        search(engine, encoder, "対象は？")


@pytest.mark.parametrize("page_number", [1, 2])
def test_missing_cross_page_source_is_not_returned(
    engine, ready_document, query_encoder, page_number
):
    identifier = ready_document(["対象者は学生です。", "期限は明日です。"])
    with engine.begin() as connection:
        connection.execute(
            delete(ChunkSource).where(
                ChunkSource.document_id == identifier, ChunkSource.page_number == page_number
            )
        )
    encoder, _ = query_encoder
    with pytest.raises(ValueError, match="出典範囲"):
        search(engine, encoder, "対象は？")


def test_cli_search_defaults_and_closes_client(
    engine, ready_document, query_encoder, monkeypatch, capsys
):
    identifier = ready_document(["条件は学生です。"])
    encoder, _ = query_encoder
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(cli, "create_database_engine", lambda _: engine)
    monkeypatch.setattr(cli.OpenAIEncoder, "from_settings", lambda _: encoder)
    monkeypatch.setattr(
        "sys.argv", ["jp-doc-agent", "search", "対象は？", "--document-id", str(identifier)]
    )
    assert cli.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["top_k"] == 5
    assert result["results"][0]["page_numbers"] == [1]
    assert encoder.client.is_closed()


def test_cli_missing_api_key_is_safe(engine, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(cli, "OpenAISettings", lambda: OpenAISettings(_env_file=None))
    monkeypatch.setattr(cli, "create_database_engine", lambda _: engine)
    monkeypatch.setattr("sys.argv", ["jp-doc-agent", "search", "対象は？"])
    assert cli.main() == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_query_api_failure_does_not_expose_secret(engine, ready_document, encoder_factory):
    ready_document(["対象者は学生です。"])

    def handler(request):
        return httpx.Response(401, json={"error": {"message": "secret-value test-key"}})

    with pytest.raises(EmbeddingError, match="HTTP 401") as error:
        search(engine, encoder_factory(handler), "対象は？")
    assert "secret-value" not in str(error.value)
    assert "test-key" not in str(error.value)
