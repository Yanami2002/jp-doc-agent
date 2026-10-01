"""Page-local Japanese chunks with verified offsets and transactional rebuilds."""

import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import version

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import Engine, delete, func, insert, select, update

from jp_doc_agent.ingestion.pdf import has_unmapped_characters
from jp_doc_agent.models import Document, DocumentChunk, DocumentPage

SEPARATORS = ("\n\n", "。", "！", "？", "\n", "、", " ", "")


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 800
    chunk_overlap: int = 100

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError("チャンクサイズは 1 以上にしてください。")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("重複文字数は 0 以上、チャンクサイズ未満にしてください。")

    def metadata(self) -> dict:
        return {
            "algorithm": "RecursiveCharacterTextSplitter",
            "policy_version": "page-v1",
            "library_version": version("langchain-text-splitters"),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "length_unit": "python_characters",
            "separators": list(SEPARATORS),
            "keep_separator": "end",
            "strip_whitespace": False,
        }


@dataclass(frozen=True)
class TextChunk:
    text: str
    start_char: int
    end_char: int


def split_page(text: str, config: ChunkingConfig) -> list[TextChunk]:
    """Preserve the stored page verbatim; offsets use Python's Unicode character indices."""
    if not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len,
        separators=list(SEPARATORS),
        keep_separator="end",
        add_start_index=True,
        strip_whitespace=False,
    )
    chunks = []
    covered_until = 0
    previous_start = -1
    for document in splitter.create_documents([text]):
        content = document.page_content
        if not content.strip():
            continue
        start = document.metadata["start_index"]
        end = start + len(content)
        if (
            start < 0
            or start <= previous_start
            or end <= covered_until
            or len(content) > config.chunk_size
            or text[start:end] != content
            or text[covered_until:start].strip()
        ):
            raise ValueError("チャンクの原文位置または文字数が不正です。保存を中止しました。")
        chunks.append(TextChunk(content, start, end))
        covered_until = end
        previous_start = start
    if text[covered_until:].strip():
        raise ValueError("分割後に未収録の本文が残っています。保存を中止しました。")
    return chunks


def chunk_document(engine: Engine, document_id: int, config: ChunkingConfig) -> dict:
    """Serialize against import/reparse and commit all chunks of a document together."""
    with engine.begin() as connection:
        document = (
            connection.execute(select(Document).where(Document.id == document_id).with_for_update())
            .mappings()
            .one_or_none()
        )
        if document is None:
            raise ValueError("文書が見つかりません。documents コマンドで ID を確認してください。")
        pages = connection.execute(
            select(DocumentPage.page_number, DocumentPage.text)
            .where(DocumentPage.document_id == document_id)
            .order_by(DocumentPage.page_number)
        ).all()
        if [page.page_number for page in pages] != list(range(1, document["page_count"] + 1)):
            raise ValueError("ページ構成が文書情報と一致しません。保存を中止しました。")
        metadata = config.metadata()
        signature = hashlib.sha256(
            json.dumps(
                [metadata, document["parser_version"], [list(page) for page in pages]],
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if document["chunking_signature"] == signature:
            count = connection.scalar(
                select(func.count())
                .select_from(DocumentChunk)
                .where(DocumentChunk.document_id == document_id)
            )
            return {"status": "duplicate", "document_id": document_id, "chunks": count}

        rows = []
        for page in pages:
            for index, chunk in enumerate(split_page(page.text, config)):
                rows.append(
                    {
                        "document_id": document_id,
                        "page_number": page.page_number,
                        "chunk_index": index,
                        "text": chunk.text,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                    }
                )
        connection.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
        if rows:
            connection.execute(insert(DocumentChunk), rows)
        connection.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(chunking_signature=signature, chunking_config=metadata)
        )
    return {"status": "chunked", "document_id": document_id, "chunks": len(rows)}


def chunk_documents(
    engine: Engine, config: ChunkingConfig, *, document_id: int | None = None
) -> list[dict]:
    if document_id is not None:
        return [chunk_document(engine, document_id, config)]
    with engine.connect() as connection:
        document_ids = connection.scalars(select(Document.id).order_by(Document.id)).all()
    return [chunk_document(engine, identifier, config) for identifier in document_ids]


def list_chunks(
    engine: Engine,
    document_id: int,
    *,
    page_number: int | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict:
    if not 1 <= limit <= 100 or offset < 0:
        raise ValueError("limit は 1〜100、offset は 0 以上にしてください。")
    with engine.connect() as connection:
        # Keep metadata, counts, and chunk rows consistent with concurrent rebuilds.
        document = (
            connection.execute(
                select(Document.title, Document.source_url, Document.chunking_config)
                .where(Document.id == document_id)
                .with_for_update(read=True)
            )
            .mappings()
            .one_or_none()
        )
        if document is None:
            raise ValueError("文書が見つかりません。documents コマンドで ID を確認してください。")
        if (
            page_number is not None
            and connection.scalar(
                select(DocumentPage.page_number).where(
                    DocumentPage.document_id == document_id,
                    DocumentPage.page_number == page_number,
                )
            )
            is None
        ):
            raise ValueError("指定されたページが見つかりません。")
        conditions = [DocumentChunk.document_id == document_id]
        if page_number is not None:
            conditions.append(DocumentChunk.page_number == page_number)
        count = connection.scalar(
            select(func.count()).select_from(DocumentChunk).where(*conditions)
        )
        rows = connection.execute(
            select(DocumentChunk)
            .where(*conditions)
            .order_by(DocumentChunk.page_number, DocumentChunk.chunk_index)
            .limit(limit)
            .offset(offset)
        ).mappings()
        chunks = [
            {**row, "has_unmapped_characters": has_unmapped_characters(row["text"])} for row in rows
        ]
    return {
        "document_id": document_id,
        **document,
        "total": count,
        "limit": limit,
        "offset": offset,
        "chunks": chunks,
    }
