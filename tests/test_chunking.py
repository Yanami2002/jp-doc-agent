"""分割・原文位置・保存・再生成を確認する。"""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import tiktoken
from sqlalchemy import delete, event, func, select, update

from jp_doc_agent import cli
from jp_doc_agent.chunking.service import chunk_document, chunk_documents, list_chunks
from jp_doc_agent.chunking.splitter import ChunkingConfig, count_tokens, split_document, split_text
from jp_doc_agent.ingestion.download import save_pdf
from jp_doc_agent.ingestion.service import import_pdf
from jp_doc_agent.models import ChunkSource, Document, DocumentChunk, DocumentPage


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
        "hello world " * 200,
        "🙂𠮷👩‍💻" * 100,
        "本文に <|endoftext|> と <|fim_prefix|> を含みます。" * 20,
    ],
)
def test_split_preserves_source_and_covers_all_nonblank_text(text):
    config = ChunkingConfig(40, 8)
    chunks = split_text(text, config)
    covered = set()
    for chunk in chunks:
        assert chunk.text == text[chunk.start_char : chunk.end_char]
        assert 0 < count_tokens(chunk.text) <= config.chunk_size
        assert chunk.text.strip()
        covered.update(range(chunk.start_char, chunk.end_char))
    assert all(index in covered for index, char in enumerate(text) if not char.isspace())
    assert chunks == split_text(text, config)
    assert [chunk.start_char for chunk in chunks] == sorted({chunk.start_char for chunk in chunks})


def test_split_prefers_sentence_end_and_keeps_punctuation():
    text = "対象者は学生です。申請条件を確認します。必要書類を提出します。"
    chunks = split_text(text, ChunkingConfig(20, 0))
    assert len(chunks) > 1
    assert all(chunk.text.endswith("。") for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == text


def test_hard_split_has_configured_overlap():
    text = "あ" * 100
    chunks = split_text(text, ChunkingConfig(40, 8))
    assert len(chunks) > 1
    for left, right in zip(chunks, chunks[1:], strict=False):
        assert count_tokens(text[right.start_char : left.end_char]) == 8


@pytest.mark.parametrize("blank_page", [False, True])
def test_cross_page_sentence_keeps_all_page_sources(blank_page):
    pages = ["本制度の対象者は、申請時点で", "日本国内に居住する学生です。"]
    if blank_page:
        pages.insert(1, "")
    chunks = split_document(pages, ChunkingConfig())
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == "\n".join(pages)
    assert [source.page_number for source in chunk.sources] == [1, len(pages)]
    for source in chunk.sources:
        assert (
            pages[source.page_number - 1][source.start_char : source.end_char]
            == (chunk.text[source.chunk_start_char : source.chunk_end_char])
        )


def test_overlap_and_ranges_can_span_pages():
    pages = ["あ" * 15 + "。" + "い" * 15, "う" * 5 + "。" + "え" * 15 + "。"]
    chunks = split_document(pages, ChunkingConfig(40, 25))
    assert any(len(chunk.sources) == 2 for chunk in chunks)
    assert any(
        left.end_char > right.start_char for left, right in zip(chunks, chunks[1:], strict=False)
    )
    for chunk in chunks:
        assert count_tokens(chunk.text) <= 40
        for source in chunk.sources:
            assert (
                pages[source.page_number - 1][source.start_char : source.end_char]
                == (chunk.text[source.chunk_start_char : source.chunk_end_char])
            )


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (20, -1), (20, 20), (20, 21)])
def test_invalid_config_is_rejected(size, overlap):
    with pytest.raises(ValueError):
        ChunkingConfig(size, overlap)


@pytest.mark.parametrize(
    "text",
    [
        "The application deadline is next Friday. " * 100,
        "学生は申請条件を確認してください。" * 100,
        "🙂𠮷👩‍💻" * 100,
    ],
)
def test_default_budget_counts_model_tokens_and_preserves_unicode(text):
    config = ChunkingConfig()
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    chunks = split_text(text, config)
    assert len(chunks) > 1
    assert config.chunk_size == 300
    assert config.chunk_overlap == 30
    for chunk in chunks:
        assert len(encoding.encode_ordinary(chunk.text)) <= 300
        assert chunk.text == text[chunk.start_char : chunk.end_char]
        assert "\ufffd" not in chunk.text
    for left, right in zip(chunks, chunks[1:], strict=False):
        if right.start_char < left.end_char:
            assert len(encoding.encode_ordinary(text[right.start_char : left.end_char])) <= 30
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(text)
    if text.isascii():
        assert any(len(chunk.text) > 300 for chunk in chunks)


def test_zero_overlap_and_repeated_english_text_have_exact_character_positions():
    text = "retrieval augmented generation " * 100
    chunks = split_text(text, ChunkingConfig(40, 0))
    assert "".join(chunk.text for chunk in chunks) == text
    assert chunks[0].start_char == 0
    for left, right in zip(chunks, chunks[1:], strict=False):
        assert right.start_char == left.end_char


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
