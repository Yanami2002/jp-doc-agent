"""Document metadata and physical PDF pages; evaluation answers stay separate."""

from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


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


class DocumentPage(Base):
    __tablename__ = "document_pages"
    __table_args__ = (CheckConstraint("page_number > 0", name="positive_page_number"),)

    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    page_number: Mapped[int] = mapped_column(primary_key=True)
    text: Mapped[str] = mapped_column(Text)
