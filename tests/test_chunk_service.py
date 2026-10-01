"""チャンクの保存・参照・再生成と CLI を実際の PostgreSQL で検証する。"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete, event, func, insert, select, update

from jp_doc_agent import cli
from jp_doc_agent.chunking.service import chunk_document, chunk_documents, list_chunks
from jp_doc_agent.chunking.splitter import ChunkingConfig, count_tokens
from jp_doc_agent.ingestion.download import save_pdf
from jp_doc_agent.ingestion.service import import_pdf
from jp_doc_agent.models import ChunkSource, Document, DocumentChunk, DocumentPage


@pytest.fixture
def text_document(engine):
    def build(pages):
        with engine.begin() as connection:
            identifier = connection.execute(
                insert(Document)
                .values(
                    sha256=hashlib.sha256(json.dumps(pages).encode()).hexdigest(),
                    title="日本語資料",
                    source_url="https://example.com/document.pdf",
                    resolved_url="https://example.com/document.pdf",
                    dataset="test",
                    file_path="/test/document.pdf",
                    page_count=len(pages),
                    acquired_at=datetime.now(UTC),
                    parser_version="test-parser",
                )
                .returning(Document.id)
            ).scalar_one()
            connection.execute(
                insert(DocumentPage),
                [
                    {"document_id": identifier, "page_number": index, "text": text}
                    for index, text in enumerate(pages, start=1)
                ],
            )
        return identifier

    return build


def test_cross_page_chunk_is_findable_by_either_source_page(engine, text_document):
    identifier = text_document(["対象者は、申請時点で", "日本国内に居住する学生です。"])
    chunk_document(engine, identifier, ChunkingConfig())
    first = list_chunks(engine, identifier, page_number=1)
    second = list_chunks(engine, identifier, page_number=2)
    assert first["chunks"] == second["chunks"]
    assert first["total"] == second["total"] == 1
    assert first["chunks"][0]["page_numbers"] == [1, 2]
    assert first["chunks"][0]["chunk_index"] == 0


def test_chunking_retains_page_numbers_and_deduplicates(engine, text_document):
    pages = ["対象者は学生です。" * 30, "", "資料の続き。(cid:123)" * 20]
    identifier = text_document(pages)
    config = ChunkingConfig(80, 10)
    first = chunk_document(engine, identifier, config)
    preview = list_chunks(engine, identifier, limit=100)
    assert first["status"] == "chunked"
    assert preview["total"] == first["chunks"] == len(preview["chunks"])
    assert {number for row in preview["chunks"] for number in row["page_numbers"]} == {1, 3}
    assert preview["source_url"] == "https://example.com/document.pdf"
    assert preview["chunking_config"] == config.metadata()
    for row in preview["chunks"]:
        assert row["token_count"] == count_tokens(row["text"])
        assert row["token_count"] <= config.chunk_size
        assert row["text"] == "\n".join(pages)[row["start_char"] : row["end_char"]]
        for source in row["sources"]:
            assert (
                row["text"][source["chunk_start_char"] : source["chunk_end_char"]]
                == pages[source["page_number"] - 1][source["start_char"] : source["end_char"]]
            )
    assert any(row["has_unmapped_characters"] for row in preview["chunks"])
    assert list_chunks(engine, identifier, page_number=2)["chunks"] == []
    assert chunk_document(engine, identifier, config) == {
        "status": "duplicate",
        "document_id": identifier,
        "chunks": first["chunks"],
    }
    assert list_chunks(engine, identifier, limit=100) == preview
    assert list_chunks(engine, identifier, limit=2, offset=1)["chunks"] == preview["chunks"][1:3]


def test_changed_config_and_source_rebuild_only_target_document(engine, text_document):
    first_id = text_document(["同じ本文です。" * 100])
    second_id = text_document(["別の資料です。"])
    chunk_documents(engine, ChunkingConfig(100, 10))
    untouched = list_chunks(engine, second_id)
    old_ids = {row["id"] for row in list_chunks(engine, first_id, limit=100)["chunks"]}
    result = chunk_document(engine, first_id, ChunkingConfig(50, 5))
    assert result["status"] == "chunked"
    preview = list_chunks(engine, first_id, limit=100)
    assert old_ids.isdisjoint({row["id"] for row in preview["chunks"]})
    assert all(row["token_count"] <= 50 for row in preview["chunks"])
    with engine.begin() as connection:
        connection.execute(
            update(DocumentPage).where(DocumentPage.document_id == first_id).values(text="変更後。")
        )
    assert chunk_document(engine, first_id, ChunkingConfig(50, 5))["status"] == "chunked"
    assert [row["text"] for row in list_chunks(engine, first_id)["chunks"]] == ["変更後。"]
    assert list_chunks(engine, second_id) == untouched


@pytest.mark.parametrize("failure_table", ["document_chunks", "chunk_sources"])
def test_rebuild_failure_rolls_back_chunks_and_config(engine, text_document, failure_table):
    identifier = text_document(["申請条件です。" * 100])
    chunk_document(engine, identifier, ChunkingConfig(100, 10))
    before = list_chunks(engine, identifier, limit=100)

    def fail_insert(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith(f"INSERT INTO {failure_table}"):
            raise RuntimeError("simulated chunk insert failure")

    event.listen(engine, "before_cursor_execute", fail_insert)
    try:
        with pytest.raises(RuntimeError, match="simulated"):
            chunk_document(engine, identifier, ChunkingConfig(50, 5))
    finally:
        event.remove(engine, "before_cursor_execute", fail_insert)
    assert list_chunks(engine, identifier, limit=100) == before
    assert chunk_document(engine, identifier, ChunkingConfig(100, 10))["status"] == "duplicate"


def test_concurrent_chunking_does_not_duplicate_rows(engine, text_document):
    identifier = text_document(["本文です。" * 100])
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda _: chunk_document(engine, identifier, ChunkingConfig()), range(2))
        )
    assert sorted(result["status"] for result in results) == ["chunked", "duplicate"]
    assert list_chunks(engine, identifier)["total"] == results[0]["chunks"]


def test_reparse_removes_old_chunks_and_resets_config(engine, tmp_path, pdf_bytes):
    pdf = save_pdf(pdf_bytes(["Original text"]), tmp_path, "https://example.com/doc.pdf")

    def load():
        return import_pdf(
            engine, pdf, title="Document", source_url=pdf.resolved_url, dataset="test"
        )

    identifier = load()["document_id"]
    chunk_document(engine, identifier, ChunkingConfig())
    with engine.begin() as connection:
        connection.execute(
            update(Document).where(Document.id == identifier).values(parser_version="old")
        )
    assert load()["status"] == "updated"
    preview = list_chunks(engine, identifier)
    assert preview["total"] == 0
    assert preview["chunking_config"] is None
    with engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(ChunkSource)) == 0
    assert chunk_document(engine, identifier, ChunkingConfig())["status"] == "chunked"
    with engine.begin() as connection:
        connection.execute(delete(Document).where(Document.id == identifier))
        assert connection.scalar(select(func.count()).select_from(DocumentChunk)) == 0
        assert connection.scalar(select(func.count()).select_from(ChunkSource)) == 0


def test_empty_pages_can_be_marked_processed(engine, text_document):
    identifier = text_document(["", " \n"])
    assert chunk_document(engine, identifier, ChunkingConfig())["chunks"] == 0
    assert chunk_document(engine, identifier, ChunkingConfig())["status"] == "duplicate"


def test_missing_documents_pages_and_invalid_pagination(engine, text_document):
    identifier = text_document(["本文です。"])
    with pytest.raises(ValueError, match="文書"):
        chunk_documents(engine, ChunkingConfig(), document_id=identifier + 1)
    with pytest.raises(ValueError, match="文書"):
        list_chunks(engine, identifier + 1)
    with pytest.raises(ValueError, match="ページ"):
        list_chunks(engine, identifier, page_number=2)
    for limit, offset in [(0, 0), (101, 0), (20, -1)]:
        with pytest.raises(ValueError, match="limit"):
            list_chunks(engine, identifier, limit=limit, offset=offset)


def test_cli_chunking_and_preview(engine, text_document, monkeypatch, capsys):
    identifier = text_document(["日本語の本文です。" * 10])
    monkeypatch.setattr(cli, "create_database_engine", lambda _: engine)
    monkeypatch.setattr(
        "sys.argv", ["jp-doc-agent", "chunk-documents", "--document-id", str(identifier)]
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["chunked"] == 1
    monkeypatch.setattr("sys.argv", ["jp-doc-agent", "chunks", str(identifier), "--page", "1"])
    assert cli.main() == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["chunks"][0]["page_numbers"] == [1]
    assert preview["chunks"][0]["token_count"] <= 300
    assert preview["chunking_config"]["chunk_size"] == 300
    assert preview["chunking_config"]["chunk_overlap"] == 30
    assert preview["chunking_config"]["length_unit"] == "tokens"
    assert preview["chunking_config"]["encoding"] == "cl100k_base"
    monkeypatch.setattr("sys.argv", ["jp-doc-agent", "chunk-documents", "--chunk-size", "0"])
    assert cli.main() == 1
    assert "チャンクサイズ" in capsys.readouterr().err
