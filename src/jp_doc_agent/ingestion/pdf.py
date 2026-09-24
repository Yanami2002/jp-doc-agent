"""Extract page text while preserving one-based physical PDF page numbers."""

import re
from pathlib import Path

import pdfminer
import pypdf
from pdfminer.high_level import extract_text
from pdfminer.pdfexceptions import PDFException
from pdfminer.psparser import PSException
from pypdf import PdfReader

PARSER_VERSION = f"pdfminer/{pdfminer.__version__};pypdf/{pypdf.__version__};text-v1"


def has_unmapped_characters(text: str) -> bool:
    return bool(re.search(r"\(cid:\d+\)", text))


def extract_pages(path: Path) -> list[str]:
    with path.open("rb") as stream:
        reader = PdfReader(stream)
        # Public PDFs can have encryption metadata but no document-open password.
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF 需要打开密码，当前不支持导入。")
        if not 1 <= len(reader.pages) <= 500:
            raise ValueError("仅支持 1～500 页的 PDF。")
        expected_pages = len(reader.pages)

    # pdfminer resolves the Japanese CMaps that pypdf misreads in this corpus.
    try:
        # TextConverter emits a form feed after every physical page, including blank pages.
        parts = extract_text(path).split("\f")
        if parts[-1] == "":
            parts.pop()
        pages = [page.replace("\x00", "").strip() for page in parts]
    except (PDFException, PSException) as error:
        raise ValueError(f"PDF 正文解析失败：{error}") from error
    if len(pages) != expected_pages:
        raise ValueError("提取页数与原始 PDF 不一致，未写入数据库。")

    if not any(pages):
        raise ValueError("所有页面均未提取到正文，可能需要 OCR；未写入数据库。")
    # Empty pages are retained so subsequent citations never shift by a page.
    return pages
