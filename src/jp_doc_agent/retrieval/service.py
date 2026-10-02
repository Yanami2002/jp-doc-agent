"""現在の本文に対応するベクトルを検索し、同じ DB スナップショットから出典を返す。"""

from sqlalchemy import Select, func, select
from sqlalchemy.engine import Connection, Engine

from jp_doc_agent.config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL
from jp_doc_agent.embedding.encoder import OpenAIEncoder, profile_id, validate_query
from jp_doc_agent.ingestion.pdf import has_unmapped_characters
from jp_doc_agent.models import ChunkEmbedding, ChunkSource, Document, DocumentChunk, DocumentPage


def _eligible_embeddings(document_id: int | None) -> Select:
    # PostgreSQL 標準の SHA-256 を使用する。pgcrypto の追加は不要。
    current_hash = func.encode(func.sha256(func.convert_to(DocumentChunk.text, "UTF8")), "hex")
    statement = (
        select(ChunkEmbedding.chunk_id, ChunkEmbedding.embedding, ChunkEmbedding.token_count)
        .join(DocumentChunk, DocumentChunk.id == ChunkEmbedding.chunk_id)
        .where(
            ChunkEmbedding.profile_id == profile_id(),
            ChunkEmbedding.input_hash == current_hash,
        )
    )
    if document_id is not None:
        statement = statement.where(DocumentChunk.document_id == document_id)
    return statement


def _coverage(connection: Connection, document_id: int | None) -> dict:
    if (
        document_id is not None
        and connection.scalar(select(Document.id).where(Document.id == document_id)) is None
    ):
        raise ValueError("文書が見つかりません。documents コマンドで ID を確認してください。")
    chunks = select(func.count()).select_from(DocumentChunk)
    unchunked = (
        select(func.count()).select_from(Document).where(Document.chunking_signature.is_(None))
    )
    if document_id is not None:
        chunks = chunks.where(DocumentChunk.document_id == document_id)
        unchunked = unchunked.where(Document.id == document_id)
    total = connection.scalar(chunks)
    searchable = connection.scalar(
        select(func.count()).select_from(_eligible_embeddings(document_id).subquery())
    )
    return {
        "chunks": total,
        "searchable": searchable,
        "pending": total - searchable,
        "unchunked_documents": connection.scalar(unchunked),
    }


def _require_embeddings(coverage: dict) -> None:
    if not coverage["searchable"]:
        raise ValueError(
            "検索できるベクトルがありません。対象文書の chunk-documents と "
            "embed-chunks を実行してください。"
        )


def _search_results(
    connection: Connection, vector: list[float], top_k: int, document_id: int | None
) -> list[dict]:
    candidates = _eligible_embeddings(document_id).subquery()
    distance = candidates.c.embedding.cosine_distance(vector).label("cosine_distance")
    rows = (
        connection.execute(
            select(
                DocumentChunk.id.label("chunk_id"),
                DocumentChunk.document_id,
                DocumentChunk.chunk_index,
                DocumentChunk.text,
                DocumentChunk.start_char,
                DocumentChunk.end_char,
                Document.title,
                Document.source_url,
                Document.resolved_url,
                candidates.c.token_count,
                distance,
            )
            .join(candidates, candidates.c.chunk_id == DocumentChunk.id)
            .join(Document, Document.id == DocumentChunk.document_id)
            .order_by(
                distance, DocumentChunk.document_id, DocumentChunk.chunk_index, DocumentChunk.id
            )
            .limit(top_k)
        )
        .mappings()
        .all()
    )
    hits = {row["chunk_id"]: dict(row) for row in rows}
    sources = {chunk_id: [] for chunk_id in hits}
    if hits:
        source_rows = connection.execute(
            select(ChunkSource, DocumentPage.text.label("page_text"))
            .join(
                DocumentPage,
                (DocumentPage.document_id == ChunkSource.document_id)
                & (DocumentPage.page_number == ChunkSource.page_number),
            )
            .where(ChunkSource.chunk_id.in_(hits))
            .order_by(ChunkSource.chunk_id, ChunkSource.page_number)
        ).mappings()
        for row in source_rows:
            item = dict(row)
            page_text = item.pop("page_text")
            chunk_id = item.pop("chunk_id")
            item.pop("document_id")
            chunk_text = hits[chunk_id]["text"]
            if (
                item["end_char"] > len(page_text)
                or item["chunk_end_char"] > len(chunk_text)
                or page_text[item["start_char"] : item["end_char"]]
                != chunk_text[item["chunk_start_char"] : item["chunk_end_char"]]
            ):
                raise ValueError("検索結果の出典が本文と一致しません。文書を再解析してください。")
            sources[chunk_id].append(item)
    results = []
    for rank, (chunk_id, hit) in enumerate(hits.items(), start=1):
        if not sources[chunk_id]:
            raise ValueError("検索結果の出典がありません。文書を再解析してください。")
        cursor = 0
        for source in sources[chunk_id]:
            # ページ結合時の改行以外に、出典のない本文や重複範囲を許可しない。
            if source["chunk_start_char"] < cursor or hit["text"][
                cursor : source["chunk_start_char"]
            ].strip("\n"):
                raise ValueError("検索結果の出典範囲が不正です。文書を再解析してください。")
            cursor = source["chunk_end_char"]
        if hit["text"][cursor:].strip("\n"):
            raise ValueError("検索結果の出典範囲が不足しています。文書を再解析してください。")
        results.append(
            {
                "rank": rank,
                **hit,
                "cosine_similarity": 1.0 - hit["cosine_distance"],
                "page_numbers": [source["page_number"] for source in sources[chunk_id]],
                "sources": sources[chunk_id],
                "has_unmapped_characters": has_unmapped_characters(hit["text"]),
            }
        )
    return results


def search(
    engine: Engine,
    encoder: OpenAIEncoder,
    query: str,
    *,
    top_k: int = 5,
    document_id: int | None = None,
) -> dict:
    """余弦距離による厳密検索。質問ベクトルは永続化しない。"""
    query_tokens = validate_query(query)
    if not 1 <= top_k <= 100:
        raise ValueError("top-k は 1〜100 にしてください。")
    # 対象がない場合は課金 API を呼ばない。API 待機中は DB 接続を保持しない。
    with engine.connect() as connection:
        _require_embeddings(_coverage(connection, document_id))
    batch = encoder.encode_query(query)
    with engine.connect().execution_options(isolation_level="REPEATABLE READ") as connection:
        with connection.begin():
            coverage = _coverage(connection, document_id)
            _require_embeddings(coverage)
            results = _search_results(connection, batch.vectors[0], top_k, document_id)
    return {
        "query": query,
        "query_tokens": query_tokens,
        "model": EMBEDDING_MODEL,
        "dimensions": EMBEDDING_DIMENSIONS,
        "profile_id": profile_id(),
        "top_k": top_k,
        "document_id": document_id,
        "coverage": coverage,
        "api_tokens": batch.total_tokens,
        "results": results,
    }
