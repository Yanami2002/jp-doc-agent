"""Import correctness, duplicate handling, rollback, and failed downloads."""

import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import httpx
import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.errors import PyPdfError
from sqlalchemy import event, func, select, update

from jp_doc_agent.ingestion import download, service
from jp_doc_agent.ingestion.download import download_pdf, save_pdf
from jp_doc_agent.ingestion.pdf import extract_pages
from jp_doc_agent.ingestion.service import import_pdf, list_documents, read_page
from jp_doc_agent.models import Document, DocumentPage


def load(engine, pdf):
    return import_pdf(
        engine, pdf, title="Test document", source_url="https://example.com/doc.pdf", dataset="test"
    )


def test_import_keeps_physical_pages_and_deduplicates(engine, tmp_path, pdf_bytes):
    pdf = save_pdf(pdf_bytes(["First", "", "Third"]), tmp_path, "https://example.com/doc.pdf")
    first = load(engine, pdf)
    second = load(engine, pdf)
    assert first["empty_pages"] == [2]
    assert second == {"status": "duplicate", "document_id": first["document_id"], "pages": 3}
    assert read_page(engine, first["document_id"], 2)["text"] == ""
    assert read_page(engine, first["document_id"], 3)["text"] == "Third"
    assert len(list_documents(engine)) == 1
    with engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(DocumentPage)) == 3


def test_changed_content_is_a_new_version(engine, tmp_path, pdf_bytes):
    first = load(engine, save_pdf(pdf_bytes(["Old"]), tmp_path, "https://example.com/doc.pdf"))
    second = load(engine, save_pdf(pdf_bytes(["New"]), tmp_path, "https://example.com/doc.pdf"))
    assert first["document_id"] != second["document_id"]
    assert len(list_documents(engine)) == 2


def test_parser_upgrade_replaces_pages_without_duplicating_document(engine, tmp_path, pdf_bytes):
    pdf = save_pdf(pdf_bytes(["Correct"]), tmp_path, "https://example.com/doc.pdf")
    first = load(engine, pdf)
    with engine.begin() as connection:
        connection.execute(update(Document).values(parser_version="old-parser"))
        connection.execute(update(DocumentPage).values(text="old extraction"))
    second = load(engine, pdf)
    assert second["status"] == "updated"
    assert first["document_id"] == second["document_id"]
    assert read_page(engine, first["document_id"], 1)["text"] == "Correct"
    assert len(list_documents(engine)) == 1


def test_page_write_failure_rolls_back_document(engine, tmp_path, pdf_bytes):
    pdf = save_pdf(pdf_bytes(["First"]), tmp_path, "https://example.com/doc.pdf")

    def fail_pages(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO document_pages"):
            raise RuntimeError("simulated page write failure")

    event.listen(engine, "before_cursor_execute", fail_pages)
    try:
        with pytest.raises(RuntimeError, match="simulated"):
            load(engine, pdf)
    finally:
        event.remove(engine, "before_cursor_execute", fail_pages)
    with engine.connect() as connection:
        assert connection.scalar(select(func.count()).select_from(Document)) == 0
        assert connection.scalar(select(func.count()).select_from(DocumentPage)) == 0


def test_concurrent_import_is_idempotent(engine, tmp_path, pdf_bytes):
    pdf = save_pdf(pdf_bytes(["Concurrent"]), tmp_path, "https://example.com/doc.pdf")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: load(engine, pdf), range(2)))
    assert sorted(row["status"] for row in results) == ["duplicate", "imported"]
    assert results[0]["document_id"] == results[1]["document_id"]
    assert len(list_documents(engine)) == 1


def test_all_empty_pdf_does_not_create_document(engine, tmp_path, pdf_bytes):
    pdf = save_pdf(pdf_bytes(["", ""]), tmp_path, "https://example.com/doc.pdf")
    with pytest.raises(ValueError, match="OCR"):
        load(engine, pdf)
    assert list_documents(engine) == []


def test_corrupt_pdf_fails_parsing(tmp_path):
    pdf = save_pdf(b"%PDF-1.7\nbroken", tmp_path, "https://example.com/doc.pdf")
    with pytest.raises(PyPdfError):
        extract_pages(pdf.path)


@pytest.mark.parametrize("password", ["", "required-password"])
def test_encrypted_pdf_open_password(tmp_path, pdf_bytes, password):
    writer = PdfWriter(clone_from=PdfReader(BytesIO(pdf_bytes(["Readable"]))))
    writer.encrypt(user_password=password, owner_password="owner", algorithm="AES-256")
    buffer = BytesIO()
    writer.write(buffer)
    pdf = save_pdf(buffer.getvalue(), tmp_path, "https://example.com/encrypted.pdf")
    if password:
        with pytest.raises(ValueError, match="打开密码"):
            extract_pages(pdf.path)
    else:
        assert extract_pages(pdf.path) == ["Readable"]


@pytest.mark.parametrize("status,body", [(404, b"not found"), (200, b"<html>error</html>")])
def test_http_failures_do_not_leave_pdf_files(tmp_path, status, body):
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(status, content=body))
    ) as client:
        with pytest.raises((httpx.HTTPStatusError, ValueError)):
            download_pdf(client, "https://example.com/doc.pdf", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_download_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "MAX_BYTES", 16)
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"%PDF-" + b"x" * 20))
    ) as client:
        with pytest.raises(ValueError, match="限制"):
            download_pdf(client, "https://example.com/doc.pdf", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_manifest_records_failure_and_continues(engine, tmp_path, pdf_bytes, monkeypatch):
    content = pdf_bytes(["Valid"])

    def respond(request):
        return (
            httpx.Response(404)
            if request.url.path == "/bad.pdf"
            else httpx.Response(200, content=content)
        )

    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(service.httpx, "Client", lambda **kwargs: client)
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps(
            [
                {"title": name, "url": f"https://example.com/{name}.pdf", "dataset": "test"}
                for name in ("bad", "good")
            ]
        )
    )
    results = service.import_manifest(engine, manifest, tmp_path)
    assert [row["status"] for row in results] == ["failed", "imported"]
    assert "404" in results[0]["reason"]
    assert len(list_documents(engine)) == 1


def test_pinned_hash_mismatch_is_not_imported(engine, tmp_path, pdf_bytes, monkeypatch):
    content = pdf_bytes(["Changed"])
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=content))
    )
    monkeypatch.setattr(service.httpx, "Client", lambda **kwargs: client)
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "title": "Changed",
                    "url": "https://example.com/doc.pdf",
                    "dataset": "test",
                    "sha256": "0" * 64,
                }
            ]
        )
    )
    results = service.import_manifest(engine, manifest, tmp_path)
    assert results[0]["status"] == "failed"
    assert "SHA-256" in results[0]["reason"]
    assert list_documents(engine) == []
