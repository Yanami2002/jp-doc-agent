"""文書単位でのチャンク再生成と出典付きの参照。"""

import hashlib
import json
from dataclasses import asdict

from sqlalchemy import Engine, delete, func, insert, select, update

from jp_doc_agent.chunking.splitter import ChunkingConfig, count_tokens, split_document
from jp_doc_agent.ingestion.pdf import has_unmapped_characters
from jp_doc_agent.models import ChunkSource, Document, DocumentChunk, DocumentPage


def chunk_document(engine: Engine, document_id: int, config: ChunkingConfig) -> dict:
    """取り込み・再解析と更新を直列化し、文書の全チャンクを一括保存する。"""
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

        chunks = split_document([page.text for page in pages], config)
        rows = [
            {
                "document_id": document_id,
                "chunk_index": index,
                "text": chunk.text,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            }
            for index, chunk in enumerate(chunks)
        ]
        connection.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
        if rows:
            chunk_ids = connection.scalars(
                insert(DocumentChunk).returning(DocumentChunk.id, sort_by_parameter_order=True),
                rows,
            ).all()
            source_rows = [
                {"chunk_id": chunk_id, "document_id": document_id, **asdict(source)}
                for chunk_id, chunk in zip(chunk_ids, chunks, strict=True)
                for source in chunk.sources
            ]
            connection.execute(insert(ChunkSource), source_rows)
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
        # 並行再生成中もメタデータ・件数・本文を同じ状態で参照する。
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
            conditions.append(
                DocumentChunk.id.in_(
                    select(ChunkSource.chunk_id).where(
                        ChunkSource.document_id == document_id,
                        ChunkSource.page_number == page_number,
                    )
                )
            )
        count = connection.scalar(
            select(func.count()).select_from(DocumentChunk).where(*conditions)
        )
        rows = (
            connection.execute(
                select(DocumentChunk)
                .where(*conditions)
                .order_by(DocumentChunk.chunk_index)
                .limit(limit)
                .offset(offset)
            )
            .mappings()
            .all()
        )
        source_groups = {row["id"]: [] for row in rows}
        if source_groups:
            sources = connection.execute(
                select(
                    ChunkSource.chunk_id,
                    ChunkSource.page_number,
                    ChunkSource.start_char,
                    ChunkSource.end_char,
                    ChunkSource.chunk_start_char,
                    ChunkSource.chunk_end_char,
                )
                .where(ChunkSource.chunk_id.in_(source_groups))
                .order_by(ChunkSource.chunk_id, ChunkSource.page_number)
            ).mappings()
            for source in sources:
                item = dict(source)
                source_groups[item.pop("chunk_id")].append(item)
        chunks = [
            {
                **row,
                "token_count": count_tokens(row["text"]),
                "page_numbers": [source["page_number"] for source in source_groups[row["id"]]],
                "sources": source_groups[row["id"]],
                "has_unmapped_characters": has_unmapped_characters(row["text"]),
            }
            for row in rows
        ]
    return {
        "document_id": document_id,
        **document,
        "total": count,
        "limit": limit,
        "offset": offset,
        "chunks": chunks,
    }
