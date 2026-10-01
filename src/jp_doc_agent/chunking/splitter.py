"""文書全体のチャンク分割とページ原文への対応付け。"""

from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

SEPARATORS = ("\n\n", "。", "！", "？", "\n", "、", " ", "")
PAGE_DELIMITER = "\n"
EMBEDDING_MODEL = "text-embedding-3-small"
TOKEN_ENCODING = "cl100k_base"


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(TOKEN_ENCODING)


def count_tokens(text: str) -> int:
    """特殊トークン風の文字列も、PDF 本文の通常の文字列として数える。"""
    return len(_encoding().encode_ordinary(text))


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 300
    chunk_overlap: int = 30

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError("チャンクサイズは 1 Token 以上にしてください。")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("重複 Token 数は 0 以上、チャンクサイズ未満にしてください。")

    def metadata(self) -> dict:
        return {
            "algorithm": "RecursiveCharacterTextSplitter",
            "policy_version": "document-token-v3",
            "page_delimiter": PAGE_DELIMITER,
            "library_version": version("langchain-text-splitters"),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "length_unit": "tokens",
            "embedding_model": EMBEDDING_MODEL,
            "tokenizer": "tiktoken",
            "tokenizer_version": version("tiktoken"),
            "encoding": TOKEN_ENCODING,
            "content_token_budget": self.chunk_size - self.chunk_overlap,
            "overlap_policy": "unicode-safe-prefix-context",
            "separators": list(SEPARATORS),
            "keep_separator": "end",
            "strip_whitespace": False,
        }


@dataclass(frozen=True)
class TextChunk:
    text: str
    start_char: int
    end_char: int


def _context_start(text: str, start: int, end: int, minimum: int, config: ChunkingConfig) -> int:
    """Token 上限内の前文を補い、UTF-8 の途中を引用開始位置にしない。"""
    if not config.chunk_overlap or minimum >= start:
        return start
    encoding = _encoding()
    previous_tokens = encoding.encode_ordinary(text[minimum:start])
    # 先頭に不完全な UTF-8 が含まれる場合だけ除外し、原文の文字境界に戻す。
    suffix = encoding.decode_bytes(previous_tokens[-config.chunk_overlap :]).decode(
        "utf-8", errors="ignore"
    )
    context_start = start - len(suffix)
    # 部分文字列の再エンコードで Token 数が変わる場合にも両方の上限を守る。
    while context_start < start and (
        count_tokens(text[context_start:start]) > config.chunk_overlap
        or count_tokens(text[context_start:end]) > config.chunk_size
    ):
        context_start += 1
    return context_start


def split_text(text: str, config: ChunkingConfig) -> list[TextChunk]:
    """原文を保持し、Python の Unicode 文字インデックスで位置を返す。"""
    if not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size - config.chunk_overlap,
        chunk_overlap=0,
        length_function=count_tokens,
        separators=list(SEPARATORS),
        keep_separator="end",
        strip_whitespace=False,
    )
    chunks = []
    covered_until = 0
    cursor = 0
    for content in splitter.split_text(text):
        # 非重複部分だけを順に対応付ける。Token 数を文字位置の計算には使わない。
        start = text.find(content, cursor)
        end = start + len(content)
        if start < cursor or text[cursor:start].strip():
            raise ValueError("チャンクの原文位置が不正です。保存を中止しました。")
        cursor = end
        if not content.strip():
            continue
        minimum = chunks[-1].start_char + 1 if chunks else start
        start = _context_start(text, start, end, minimum, config)
        content = text[start:end]
        if (
            start < 0
            or (chunks and start <= chunks[-1].start_char)
            or end <= covered_until
            or count_tokens(content) > config.chunk_size
            or text[covered_until:start].strip()
        ):
            raise ValueError("チャンクの原文位置または Token 数が不正です。保存を中止しました。")
        chunks.append(TextChunk(content, start, end))
        covered_until = end
    if text[covered_until:].strip():
        raise ValueError("分割後に未収録の本文が残っています。保存を中止しました。")
    return chunks


@dataclass(frozen=True)
class SourceRange:
    page_number: int
    start_char: int
    end_char: int
    chunk_start_char: int
    chunk_end_char: int


@dataclass(frozen=True)
class DocumentTextChunk:
    text: str
    start_char: int
    end_char: int
    sources: tuple[SourceRange, ...]


def split_document(pages: list[str], config: ChunkingConfig) -> list[DocumentTextChunk]:
    """ページ境界を越えて分割し、変更していない各ページの原文位置に対応付ける。"""
    text = PAGE_DELIMITER.join(pages)
    page_ranges = []
    offset = 0
    for number, page in enumerate(pages, start=1):
        page_ranges.append((number, offset, offset + len(page)))
        offset += len(page) + len(PAGE_DELIMITER)

    chunks = []
    for chunk in split_text(text, config):
        sources = []
        for number, page_start, page_end in page_ranges:
            start = max(chunk.start_char, page_start)
            end = min(chunk.end_char, page_end)
            if start >= end or not text[start:end].strip():
                continue
            source = SourceRange(
                page_number=number,
                start_char=start - page_start,
                end_char=end - page_start,
                chunk_start_char=start - chunk.start_char,
                chunk_end_char=end - chunk.start_char,
            )
            if (
                pages[number - 1][source.start_char : source.end_char]
                != chunk.text[source.chunk_start_char : source.chunk_end_char]
            ):
                raise ValueError("チャンクとページ原文が一致しません。保存を中止しました。")
            sources.append(source)
        if not sources:
            raise ValueError("チャンクの出典ページを特定できません。保存を中止しました。")
        chunks.append(
            DocumentTextChunk(chunk.text, chunk.start_char, chunk.end_char, tuple(sources))
        )
    return chunks
