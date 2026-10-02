"""Embedding のマイグレーションが文書・本文・出典を保持することを確認する。"""

import json

import httpx
from alembic import command
from alembic.config import Config
from sqlalchemy import func, select, text

from jp_doc_agent.chunking.service import chunk_document
from jp_doc_agent.chunking.splitter import ChunkingConfig
from jp_doc_agent.embedding.service import embed_document
from jp_doc_agent.models import ChunkEmbedding, ChunkSource, DocumentChunk, DocumentPage


def test_embedding_upgrade_and_downgrade_preserve_source_data(
    engine, text_document, encoder_factory, api_response
):
    identifier = text_document(["対象者は学生です。", "期限を確認してください。"])
    chunk_document(engine, identifier, ChunkingConfig())
    encoder = encoder_factory(
        lambda request: httpx.Response(200, json=api_response(json.loads(request.content)["input"]))
    )
    assert embed_document(engine, encoder, identifier)["embedded"] == 1
    with engine.begin() as connection:

        def source_snapshot():
            return {
                model.__tablename__: connection.execute(select(model)).mappings().all()
                for model in (DocumentPage, DocumentChunk, ChunkSource)
            }

        before = source_snapshot()
        config = Config("alembic.ini")
        config.attributes.update(
            connection=connection,
            version_table_schema=connection.scalar(text("SELECT current_schema()")),
        )
        command.downgrade(config, "0003_cross_page_chunks")
        assert source_snapshot() == before
        command.upgrade(config, "head")
        assert source_snapshot() == before
        assert connection.scalar(select(func.count()).select_from(ChunkEmbedding)) == 0
