"""Bounded downloads and content-addressed local PDF storage."""

import hashlib
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

MAX_BYTES = 50 * 1024 * 1024


@dataclass(frozen=True)
class PdfFile:
    path: Path
    sha256: str
    resolved_url: str
    acquired_at: datetime


def save_pdf(content: bytes, directory: Path, resolved_url: str) -> PdfFile:
    """Only complete PDF bytes are published under their SHA-256 filename."""
    if not content.startswith(b"%PDF-"):
        raise ValueError("文件不是 PDF，可能下载到了 HTML 错误页面。")
    if len(content) > MAX_BYTES:
        raise ValueError("PDF 超过 50 MiB 限制。")
    digest = hashlib.sha256(content).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{digest}.pdf"
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(content)
                stream.close()
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    return PdfFile(target, digest, resolved_url, datetime.now(UTC))


def download_pdf(client: httpx.Client, url: str, directory: Path) -> PdfFile:
    if httpx.URL(url).scheme not in {"http", "https"}:
        raise ValueError("下载地址必须使用 http 或 https。")
    content = bytearray()
    with client.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes(chunk_size=65536):
            content.extend(chunk)
            if len(content) > MAX_BYTES:
                raise ValueError("PDF 超过 50 MiB 限制。")
        return save_pdf(bytes(content), directory, str(response.url))


def copy_local_pdf(path: Path, directory: Path) -> PdfFile:
    with path.open("rb") as stream:
        content = stream.read(MAX_BYTES + 1)
    return save_pdf(content, directory, path.resolve().as_uri())
