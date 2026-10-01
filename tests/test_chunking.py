"""日本語本文の保持、ページをまたぐ分割、出典範囲を DB なしで検証する。"""

import pytest
import tiktoken

from jp_doc_agent.chunking.splitter import ChunkingConfig, count_tokens, split_document, split_text


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
