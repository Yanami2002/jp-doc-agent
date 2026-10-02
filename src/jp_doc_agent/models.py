"""Document metadata and physical PDF pages; evaluation answers stay separate."""

from datetime import datetime

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from jp_doc_agent.config import EMBEDDING_DIMENSIONS


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (CheckConstraint("page_count > 0", name="positive_page_count"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    title: Mapped[str] = mapped_column(Text)
    source_url: Mapped[str] = mapped_column(Text)
    resolved_url: Mapped[str] = mapped_column(Text)
    dataset: Mapped[str] = mapped_column(String(100))
    file_path: Mapped[str] = mapped_column(Text)
    page_count: Mapped[int]
    acquired_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    imported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    parser_version: Mapped[str] = mapped_column(String(100))
    chunking_signature: Mapped[str | None] = mapped_column(String(64))
    chunking_config: Mapped[dict | None] = mapped_column(JSON)


class DocumentPage(Base):
    __tablename__ = "document_pages"
    __table_args__ = (CheckConstraint("page_number > 0", name="positive_page_number"),)

    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    page_number: Mapped[int] = mapped_column(primary_key=True)
    text: Mapped[str] = mapped_column(Text)


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="unique_document_chunk"),
        UniqueConstraint("id", "document_id", name="unique_chunk_document"),
        CheckConstraint("chunk_index >= 0", name="nonnegative_chunk_index"),
        CheckConstraint("start_char >= 0 AND end_char > start_char", name="valid_chunk_range"),
        CheckConstraint("char_length(text) = end_char - start_char", name="chunk_text_length"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    chunk_index: Mapped[int]
    text: Mapped[str] = mapped_column(Text)
    start_char: Mapped[int]
    end_char: Mapped[int]


class ChunkSource(Base):
    __tablename__ = "chunk_sources"
    __table_args__ = (
        ForeignKeyConstraint(
            ["chunk_id", "document_id"],
            ["document_chunks.id", "document_chunks.document_id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["document_id", "page_number"],
            ["document_pages.document_id", "document_pages.page_number"],
            ondelete="CASCADE",
        ),
        CheckConstraint("start_char >= 0 AND end_char > start_char", name="valid_source_range"),
        CheckConstraint(
            "chunk_start_char >= 0 AND chunk_end_char > chunk_start_char",
            name="valid_source_chunk_range",
        ),
        CheckConstraint(
            "end_char - start_char = chunk_end_char - chunk_start_char",
            name="source_range_length",
        ),
    )

    chunk_id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int]
    page_number: Mapped[int] = mapped_column(primary_key=True)
    start_char: Mapped[int]
    end_char: Mapped[int]
    chunk_start_char: Mapped[int]
    chunk_end_char: Mapped[int]


class EmbeddingProfile(Base):
    __tablename__ = "embedding_profiles"
    __table_args__ = (CheckConstraint("dimensions = 1536", name="embedding_profile_dimensions"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    provider: Mapped[str] = mapped_column(String(50))
    model: Mapped[str] = mapped_column(String(100))
    dimensions: Mapped[int]
    config: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"
    __table_args__ = (
        CheckConstraint("token_count > 0 AND token_count <= 300", name="embedding_token_count"),
        CheckConstraint("vector_norm(embedding) > 0", name="nonzero_embedding"),
    )

    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("document_chunks.id", ondelete="CASCADE"), primary_key=True
    )
    profile_id: Mapped[str] = mapped_column(ForeignKey("embedding_profiles.id"), primary_key=True)
    input_hash: Mapped[str] = mapped_column(String(64))
    token_count: Mapped[int]
    embedding: Mapped[list[float]] = mapped_column(VECTOR(EMBEDDING_DIMENSIONS))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
