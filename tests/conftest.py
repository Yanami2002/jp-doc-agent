"""Run integration tests in temporary PostgreSQL schemas, never in application tables."""

from io import BytesIO
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from sqlalchemy import create_engine, text

from jp_doc_agent.config import Settings
from jp_doc_agent.database import create_database_engine


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
