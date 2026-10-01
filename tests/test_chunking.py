"""Japanese text fidelity, source provenance, and atomic database chunk rebuilds."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete, event, func, insert, select, update

from jp_doc_agent import cli
from jp_doc_agent.chunking import (
    ChunkingConfig,
    chunk_document,
    chunk_documents,
    list_chunks,
    split_page,
)
from jp_doc_agent.ingestion.download import save_pdf
from jp_doc_agent.ingestion.service import import_pdf
from jp_doc_agent.models import Document, DocumentChunk, DocumentPage


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


@pytest.mark.parametrize(
    "text",
    [
        "",
        " \n\t\u3000 ",
        "対象者は学生です。申請条件は日本在住です。必要書類を確認してください！",
        "\n\n概要\n本文は複数行です。\n\n詳細\n数字 1,234 と単位を保持します。\n",
        "あ" * 213,
        "同じ文章です。" * 50,
        "ABC🙂e\u0301日本語\t" * 30,
        "本文。" + "\n" * 90 + "続き。",
        "表\n項目\t数値\n売上\t1,234\n利益\t56\n" * 20,
    ],
)
def test_split_preserves_source_and_covers_all_nonblank_text(text):
    config = ChunkingConfig(40, 8)
    chunks = split_page(text, config)
    covered = set()
    for chunk in chunks:
        assert chunk.text == text[chunk.start_char : chunk.end_char]
        assert 0 < len(chunk.text) <= config.chunk_size
        assert chunk.text.strip()
        covered.update(range(chunk.start_char, chunk.end_char))
    assert all(index in covered for index, char in enumerate(text) if not char.isspace())
    assert chunks == split_page(text, config)
    assert [chunk.start_char for chunk in chunks] == sorted({chunk.start_char for chunk in chunks})


def test_split_prefers_sentence_end_and_keeps_punctuation():
    text = "対象者は学生です。申請条件を確認します。必要書類を提出します。"
    chunks = split_page(text, ChunkingConfig(20, 0))
    assert len(chunks) > 1
    assert all(chunk.text.endswith("。") for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == text


def test_hard_split_has_configured_overlap():
    text = "あ" * 100
    chunks = split_page(text, ChunkingConfig(40, 8))
    assert len(chunks) == 3
    for left, right in zip(chunks, chunks[1:], strict=False):
        assert left.end_char - right.start_char == 8


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (20, -1), (20, 20), (20, 21)])
def test_invalid_config_is_rejected(size, overlap):
    with pytest.raises(ValueError):
        ChunkingConfig(size, overlap)


def test_chunking_retains_page_numbers_and_deduplicates(engine, text_document):
    pages = ["対象者は学生です。" * 30, "", "資料の続き。(cid:123)" * 20]
    identifier = text_document(pages)
    config = ChunkingConfig(80, 10)
    first = chunk_document(engine, identifier, config)
    preview = list_chunks(engine, identifier, limit=100)
    assert first["status"] == "chunked"
    assert preview["total"] == first["chunks"] == len(preview["chunks"])
    assert {row["page_number"] for row in preview["chunks"]} == {1, 3}
    assert preview["source_url"] == "https://example.com/document.pdf"
    assert preview["chunking_config"] == config.metadata()
    for row in preview["chunks"]:
        assert row["text"] == pages[row["page_number"] - 1][row["start_char"] : row["end_char"]]
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
    assert all(len(row["text"]) <= 50 for row in preview["chunks"])
    with engine.begin() as connection:
        connection.execute(
            update(DocumentPage).where(DocumentPage.document_id == first_id).values(text="変更後。")
        )
    assert chunk_document(engine, first_id, ChunkingConfig(50, 5))["status"] == "chunked"
    assert [row["text"] for row in list_chunks(engine, first_id)["chunks"]] == ["変更後。"]
    assert list_chunks(engine, second_id) == untouched


def test_rebuild_failure_rolls_back_chunks_and_config(engine, text_document):
    identifier = text_document(["申請条件です。" * 100])
    chunk_document(engine, identifier, ChunkingConfig(100, 10))
    before = list_chunks(engine, identifier, limit=100)

    def fail_insert(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO document_chunks"):
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
    assert chunk_document(engine, identifier, ChunkingConfig())["status"] == "chunked"
    with engine.begin() as connection:
        connection.execute(delete(Document).where(Document.id == identifier))
        assert connection.scalar(select(func.count()).select_from(DocumentChunk)) == 0


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
    assert json.loads(capsys.readouterr().out)["chunks"][0]["page_number"] == 1
    monkeypatch.setattr("sys.argv", ["jp-doc-agent", "chunk-documents", "--chunk-size", "0"])
    assert cli.main() == 1
    assert "チャンクサイズ" in capsys.readouterr().err
