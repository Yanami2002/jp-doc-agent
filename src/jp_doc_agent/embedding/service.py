"""チャンクのベクトル化をバッチ単位で保存し、未完了分から再開する。"""

from sqlalchemy import Engine, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from jp_doc_agent.chunking.splitter import count_tokens
from jp_doc_agent.config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL
from jp_doc_agent.embedding.encoder import (
    EmbeddingBatch,
    EmbeddingError,
    OpenAIEncoder,
    input_hash,
    profile_config,
    profile_id,
)
from jp_doc_agent.models import ChunkEmbedding, Document, DocumentChunk, EmbeddingProfile


def _snapshot(engine: Engine, document_id: int) -> tuple[dict, list[dict]]:
    with engine.connect() as connection:
        document = (
            connection.execute(
                select(Document.id, Document.chunking_signature)
                .where(Document.id == document_id)
                .with_for_update(read=True)
            )
            .mappings()
            .one_or_none()
        )
        if document is None:
            raise ValueError("文書が見つかりません。documents コマンドで ID を確認してください。")
        if document["chunking_signature"] is None:
            raise ValueError("未分割の文書です。先に chunk-documents を実行してください。")
        rows = (
            connection.execute(
                select(DocumentChunk.id, DocumentChunk.text, ChunkEmbedding.input_hash)
                .outerjoin(
                    ChunkEmbedding,
                    (ChunkEmbedding.chunk_id == DocumentChunk.id)
                    & (ChunkEmbedding.profile_id == profile_id()),
                )
                .where(DocumentChunk.document_id == document_id)
                .order_by(DocumentChunk.chunk_index)
            )
            .mappings()
            .all()
        )
        return dict(document), [dict(row) for row in rows]


def _save_batch(engine: Engine, document: dict, chunks: list[dict], result: EmbeddingBatch) -> int:
    with engine.begin() as connection:
        current = connection.scalar(
            select(Document.chunking_signature)
            .where(Document.id == document["id"])
            .with_for_update()
        )
        current_chunks = connection.execute(
            select(DocumentChunk.id, DocumentChunk.text)
            .where(DocumentChunk.document_id == document["id"])
            .where(DocumentChunk.id.in_([chunk["id"] for chunk in chunks]))
        ).all()
        expected = {chunk["id"]: chunk["text"] for chunk in chunks}
        if current != document["chunking_signature"] or dict(current_chunks) != expected:
            raise ValueError(
                "API 実行中に本文が変更されました。embed-chunks を再実行してください。"
            )

        connection.execute(
            pg_insert(EmbeddingProfile)
            .values(
                id=profile_id(),
                provider="openai",
                model=EMBEDDING_MODEL,
                dimensions=EMBEDDING_DIMENSIONS,
                config=profile_config(),
            )
            .on_conflict_do_nothing(index_elements=[EmbeddingProfile.id])
        )
        rows = [
            {
                "chunk_id": chunk["id"],
                "profile_id": profile_id(),
                "input_hash": input_hash(chunk["text"]),
                "token_count": count_tokens(chunk["text"]),
                "embedding": vector,
            }
            for chunk, vector in zip(chunks, result.vectors, strict=True)
        ]
        statement = pg_insert(ChunkEmbedding).values(rows)
        statement = statement.on_conflict_do_update(
            index_elements=[ChunkEmbedding.chunk_id, ChunkEmbedding.profile_id],
            set_={
                "input_hash": statement.excluded.input_hash,
                "token_count": statement.excluded.token_count,
                "embedding": statement.excluded.embedding,
                "created_at": func.now(),
            },
            where=ChunkEmbedding.input_hash != statement.excluded.input_hash,
        ).returning(ChunkEmbedding.chunk_id)
        return len(connection.scalars(statement).all())


def embed_document(
    engine: Engine, encoder: OpenAIEncoder, document_id: int, *, batch_size: int = 32
) -> dict:
    if not 1 <= batch_size <= 32:
        raise ValueError("batch-size は 1〜32 にしてください。")
    stats = {
        "document_id": document_id,
        "status": "duplicate",
        "chunks": 0,
        "embedded": 0,
        "skipped": 0,
        "api_requests": 0,
        "api_tokens": 0,
    }
    try:
        document, chunks = _snapshot(engine, document_id)
        stats["chunks"] = len(chunks)
        pending = [chunk for chunk in chunks if chunk["input_hash"] != input_hash(chunk["text"])]
        stats["skipped"] = len(chunks) - len(pending)
        if any(not chunk["text"].strip() or count_tokens(chunk["text"]) > 300 for chunk in pending):
            raise ValueError("300 Token を超えるか空白だけの本文があります。再分割してください。")
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset : offset + batch_size]
            # API 待機中は DB 接続・文書ロックを保持しない。
            result = encoder.encode([chunk["text"] for chunk in batch])
            stats["api_requests"] += 1
            stats["api_tokens"] += result.total_tokens
            saved = _save_batch(engine, document, batch, result)
            stats["embedded"] += saved
            stats["skipped"] += len(batch) - saved
        if stats["embedded"]:
            stats["status"] = "embedded"
        elif not chunks:
            stats["status"] = "empty"
    except (EmbeddingError, ValueError) as error:
        stats.update(status="failed", reason=str(error))
    return stats


def embed_documents(
    engine: Engine,
    encoder: OpenAIEncoder,
    *,
    document_id: int | None = None,
    batch_size: int = 32,
) -> list[dict]:
    if not 1 <= batch_size <= 32:
        raise ValueError("batch-size は 1〜32 にしてください。")
    if document_id is not None:
        identifiers = [document_id]
    else:
        with engine.connect() as connection:
            identifiers = connection.scalars(select(Document.id).order_by(Document.id)).all()
    return [
        embed_document(engine, encoder, identifier, batch_size=batch_size)
        for identifier in identifiers
    ]


def embedding_status(engine: Engine, *, document_id: int | None = None) -> dict:
    """現在のモデル設定・本文ハッシュに一致する保存件数を API なしで確認する。"""
    with engine.connect() as connection:
        query = select(Document.id, Document.title, Document.chunking_signature).order_by(
            Document.id
        )
        if document_id is not None:
            query = query.where(Document.id == document_id)
        documents = connection.execute(query.with_for_update(read=True)).mappings().all()
        if document_id is not None and not documents:
            raise ValueError("文書が見つかりません。documents コマンドで ID を確認してください。")
        counts = {
            doc["id"]: {
                "document_id": doc["id"],
                "title": doc["title"],
                "chunked": doc["chunking_signature"] is not None,
                "chunks": 0,
                "embedded": 0,
                "pending": 0,
            }
            for doc in documents
        }
        rows = connection.execute(
            select(DocumentChunk.document_id, DocumentChunk.text, ChunkEmbedding.input_hash)
            .outerjoin(
                ChunkEmbedding,
                (ChunkEmbedding.chunk_id == DocumentChunk.id)
                & (ChunkEmbedding.profile_id == profile_id()),
            )
            .where(DocumentChunk.document_id.in_(counts))
        )
        for row in rows:
            counts[row.document_id]["chunks"] += 1
            if row.input_hash == input_hash(row.text):
                counts[row.document_id]["embedded"] += 1
            else:
                counts[row.document_id]["pending"] += 1
    return {
        "profile_id": profile_id(),
        "model": EMBEDDING_MODEL,
        "dimensions": EMBEDDING_DIMENSIONS,
        "chunks": sum(item["chunks"] for item in counts.values()),
        "embedded": sum(item["embedded"] for item in counts.values()),
        "pending": sum(item["pending"] for item in counts.values()),
        "documents": list(counts.values()),
    }
