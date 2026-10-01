"""Preserve page-local data on upgrade; invalidate only derived data on downgrade."""

from datetime import UTC, datetime

from alembic import command
from alembic.config import Config
from sqlalchemy import func, insert, select, text

from jp_doc_agent.models import ChunkSource, Document, DocumentChunk, DocumentPage


def test_cross_page_migration_preserves_old_chunks_and_pages(engine):
    with engine.begin() as connection:
        config = Config("alembic.ini")
        config.attributes.update(
            connection=connection,
            version_table_schema=connection.scalar(text("SELECT current_schema()")),
        )
        command.downgrade(config, "0002_document_chunks")
        identifier = connection.execute(
            insert(Document)
            .values(
                sha256="a" * 64,
                title="Migration fixture",
                source_url="https://example.com/test.pdf",
                resolved_url="https://example.com/test.pdf",
                dataset="test",
                file_path="/test.pdf",
                page_count=3,
                acquired_at=datetime.now(UTC),
                parser_version="test",
                chunking_signature="b" * 64,
                chunking_config={"policy_version": "page-v1"},
            )
            .returning(Document.id)
        ).scalar_one()
        connection.execute(
            insert(DocumentPage),
            [
                {"document_id": identifier, "page_number": number, "text": value}
                for number, value in enumerate(["原文", "", "続き。"], start=1)
            ],
        )
        connection.execute(
            text("""
                INSERT INTO document_chunks
                    (document_id, page_number, chunk_index, text, start_char, end_char)
                VALUES (:document_id, 1, 0, '原文', 0, 2), (:document_id, 3, 0, '続き。', 0, 3)
            """),
            {"document_id": identifier},
        )
        old_ids = connection.scalars(select(DocumentChunk.id).order_by(DocumentChunk.id)).all()
        command.upgrade(config, "head")
        chunks = (
            connection.execute(select(DocumentChunk).order_by(DocumentChunk.chunk_index))
            .mappings()
            .all()
        )
        assert [row["id"] for row in chunks] == old_ids
        assert [row["chunk_index"] for row in chunks] == [0, 1]
        assert [(row["start_char"], row["end_char"]) for row in chunks] == [(0, 2), (4, 7)]
        sources = (
            connection.execute(select(ChunkSource).order_by(ChunkSource.page_number))
            .mappings()
            .all()
        )
        assert [row["page_number"] for row in sources] == [1, 3]
        assert [(row["start_char"], row["end_char"]) for row in sources] == [(0, 2), (0, 3)]
        assert connection.scalar(select(Document.chunking_signature)) is None
        assert connection.scalar(select(Document.chunking_config)) is None

        command.downgrade(config, "0002_document_chunks")
        assert connection.scalar(select(func.count()).select_from(DocumentPage)) == 3
        assert connection.scalar(select(func.count()).select_from(DocumentChunk)) == 0
        command.upgrade(config, "head")
        assert connection.scalar(select(func.count()).select_from(ChunkSource)) == 0
