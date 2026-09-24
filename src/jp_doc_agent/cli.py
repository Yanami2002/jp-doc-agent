"""Command-line entry point for local development."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import ValidationError
from pypdf.errors import PyPdfError
from sqlalchemy.exc import SQLAlchemyError

from jp_doc_agent.benchmark import fetch_benchmark
from jp_doc_agent.config import Settings
from jp_doc_agent.database import check_database, create_database_engine
from jp_doc_agent.ingestion.download import copy_local_pdf
from jp_doc_agent.ingestion.service import import_manifest, import_pdf, list_documents, read_page


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JP Doc Agent 开发工具")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check-db", help="检查 PostgreSQL 和 pgvector")
    commands.add_parser("documents", help="列出已导入文档")

    batch = commands.add_parser("import-documents", help="按来源清单下载并导入 PDF")
    batch.add_argument("--manifest", type=Path, default=Path("sources/fujitsu.json"))
    batch.add_argument("--data-dir", type=Path, default=Path("data"))

    benchmark = commands.add_parser("fetch-benchmark", help="下载原始评测标注及使用条款")
    benchmark.add_argument("--data-dir", type=Path, default=Path("data"))

    local = commands.add_parser("import-pdf", help="导入本地 PDF")
    local.add_argument("path", type=Path)
    local.add_argument("--title")
    local.add_argument("--source-url", help="原始来源链接；省略时记录本地文件地址")
    local.add_argument("--dataset", default="local")
    local.add_argument("--data-dir", type=Path, default=Path("data"))

    page = commands.add_parser("page", help="读取某份文档的指定页面")
    page.add_argument("document_id", type=int)
    page.add_argument("page_number", type=int, help="从 1 开始的 PDF 物理页码")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "fetch-benchmark":
        try:
            result = fetch_benchmark(args.data_dir)
        except (httpx.HTTPError, ValueError, OSError) as error:
            print(str(error), file=sys.stderr)
            return 1
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    try:
        settings = Settings()
    except ValidationError:
        print("配置无效：请在项目目录检查 .env 中的数据库参数。", file=sys.stderr)
        return 1

    engine = create_database_engine(settings)
    try:
        if args.command == "check-db":
            result = {"status": "ok", **check_database(engine)}
        elif args.command == "import-documents":
            results = import_manifest(engine, args.manifest, args.data_dir)
            report_dir = args.data_dir / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            report = report_dir / f"import-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.json"
            report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            counts = {
                status: sum(item["status"] == status for item in results)
                for status in ("imported", "updated", "duplicate", "failed")
            }
            print(json.dumps({**counts, "report": str(report)}, ensure_ascii=False, indent=2))
            return 1 if counts["failed"] else 0
        elif args.command == "import-pdf":
            if not 1 <= len(args.dataset) <= 100:
                raise ValueError("dataset 长度必须为 1～100。")
            pdf = copy_local_pdf(args.path, args.data_dir / "pdfs")
            result = import_pdf(
                engine,
                pdf,
                title=args.title or args.path.name,
                source_url=args.source_url or pdf.resolved_url,
                dataset=args.dataset,
            )
        elif args.command == "documents":
            result = list_documents(engine)
        else:
            result = read_page(engine, args.document_id, args.page_number)
    except SQLAlchemyError:
        print(
            "数据库操作失败：请检查连接配置，并确认已执行 uv run alembic upgrade head。",
            file=sys.stderr,
        )
        return 1
    except (RuntimeError, ValueError, OSError, PyPdfError) as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        engine.dispose()

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
