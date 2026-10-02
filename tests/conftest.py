"""Run integration tests in temporary PostgreSQL schemas, never in application tables."""

import hashlib
import json
from datetime import UTC, datetime
from io import BytesIO
from uuid import uuid4

import httpx
import pytest
from alembic import command
from alembic.config import Config
from openai import OpenAI
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from sqlalchemy import create_engine, insert, text

from jp_doc_agent.config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, Settings
from jp_doc_agent.database import create_database_engine
from jp_doc_agent.embedding.encoder import OpenAIEncoder
from jp_doc_agent.models import Document, DocumentPage


@pytest.fixture
def engine():
    settings = Settings()
    admin = create_database_engine(settings)
    schema = f"test_import_{uuid4().hex}"
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(
        settings.database_url,
        connect_args={"options": f"-c search_path={schema},public -c statement_timeout=5000"},
    )
    try:
        with engine.begin() as connection:
            config = Config("alembic.ini")
            config.attributes.update(connection=connection, version_table_schema=schema)
            command.upgrade(config, "head")
        yield engine
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


@pytest.fixture
def pdf_bytes():
    """Small synthetic fixtures test edge cases; real PDFs are validated separately."""

    def build(pages):
        writer = PdfWriter()
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        for content in pages:
            page = writer.add_blank_page(width=300, height=300)
            page[NameObject("/Resources")] = DictionaryObject(
                {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
            )
            if content:
                stream = DecodedStreamObject()
                stream.set_data(f"BT /F1 12 Tf 20 250 Td ({content}) Tj ET".encode("ascii"))
                page[NameObject("/Contents")] = writer._add_object(stream)
        buffer = BytesIO()
        writer.write(buffer)
        return buffer.getvalue()

    return build


@pytest.fixture
def text_document(engine):
    def build(pages):
        with engine.begin() as connection:
            identifier = connection.execute(
                insert(Document)
                .values(
                    sha256=hashlib.sha256(json.dumps(pages).encode()).hexdigest(),
                    title="日本語資料",
                    source_url="https://example.com/document.pdf",
                    resolved_url="https://example.com/document.pdf",
                    dataset="test",
                    file_path="/test/document.pdf",
                    page_count=len(pages),
                    acquired_at=datetime.now(UTC),
                    parser_version="test-parser",
                )
                .returning(Document.id)
            ).scalar_one()
            connection.execute(
                insert(DocumentPage),
                [
                    {"document_id": identifier, "page_number": index, "text": text}
                    for index, text in enumerate(pages, start=1)
                ],
            )
        return identifier

    return build


@pytest.fixture
def api_response():
    def embedding_response(texts: list[str]) -> dict:
        data = []
        for index, _ in enumerate(texts):
            vector = [0.0] * EMBEDDING_DIMENSIONS
            vector[index] = 1.0
            data.append({"object": "embedding", "index": index, "embedding": vector})
        return {
            "object": "list",
            "model": EMBEDDING_MODEL,
            "data": data,
            "usage": {"prompt_tokens": len(texts), "total_tokens": len(texts)},
        }

    return embedding_response


@pytest.fixture
def encoder_factory():
    clients = []

    def build(handler, *, retries=0):
        client = OpenAI(
            api_key="test-key",
            max_retries=retries,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        clients.append(client)
        return OpenAIEncoder(client)

    yield build
    for client in clients:
        client.close()
