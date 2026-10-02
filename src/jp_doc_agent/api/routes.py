"""HTTP の引数を既存サービスに渡し、問答と実行記録を返す。"""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Request
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from jp_doc_agent.agent.service import agent_ask
from jp_doc_agent.api.dependencies import get_engine, model_resources
from jp_doc_agent.api.errors import APIError
from jp_doc_agent.api.schema import (
    AskRequest,
    AskResponse,
    DocumentResponse,
    ErrorResponse,
    HealthResponse,
    PageResponse,
)
from jp_doc_agent.database import check_database
from jp_doc_agent.embedding.encoder import EmbeddingError
from jp_doc_agent.ingestion.service import list_documents, read_page
from jp_doc_agent.llm import ModelError
from jp_doc_agent.models import Document
from jp_doc_agent.reports import write_report

router = APIRouter(responses={code: {"model": ErrorResponse} for code in (500, 503)})
Database = Annotated[Engine, Depends(get_engine)]
PositiveId = Annotated[int, Path(ge=1)]


@router.get("/health", response_model=HealthResponse, summary="DB と pgvector の接続を確認")
def health(engine: Database) -> dict:
    try:
        return {"status": "ok", **check_database(engine)}
    except RuntimeError:
        raise APIError(
            503, "database_unavailable", "データベースの vector 拡張を確認してください。"
        ) from None


@router.get("/documents", response_model=list[DocumentResponse], summary="登録済み文書の一覧")
def documents(engine: Database) -> list[dict]:
    return list_documents(engine)


@router.get(
    "/documents/{document_id}/pages/{page_number}",
    response_model=PageResponse,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    summary="PDF の指定物理ページを参照",
)
def document_page(document_id: PositiveId, page_number: PositiveId, engine: Database) -> dict:
    try:
        return read_page(engine, document_id, page_number)
    except ValueError:
        raise APIError(
            404, "page_not_found", "指定された文書または物理ページが見つかりません。"
        ) from None


def _save_run(request: Request, report: dict) -> None:
    try:
        write_report(report, request.app.state.report_dir / f"api-{request.state.request_id}.json")
    except OSError:
        raise APIError(
            503, "report_unavailable", "問答の実行記録を保存できません。保存先を確認してください。"
        ) from None


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={code: {"model": ErrorResponse} for code in (404, 409, 422, 502)},
    summary="日本語で質問し、回答・引用・調査過程を取得",
    description="Agent は必要に応じて追加調査します。根拠不足も正常な結果として返します。",
)
def answer(payload: AskRequest, request: Request, engine: Database) -> AskResponse:
    if payload.document_id is not None:
        with engine.connect() as connection:
            if (
                connection.scalar(select(Document.id).where(Document.id == payload.document_id))
                is None
            ):
                raise APIError(404, "document_not_found", "指定された文書が見つかりません。")
    metadata = {"request_id": request.state.request_id, **payload.model_dump()}
    try:
        with model_resources() as (encoder, generator):
            result = agent_ask(
                engine,
                encoder,
                generator,
                payload.question,
                top_k=payload.top_k,
                document_id=payload.document_id,
            )
    except (ModelError, EmbeddingError) as error:
        _save_run(request, {**metadata, "status": "error", "error": str(error)})
        raise APIError(502, "model_error", str(error)) from None
    except SQLAlchemyError:
        message = "データベースに接続・アクセスできません。"
        _save_run(request, {**metadata, "status": "error", "error": message})
        raise APIError(503, "database_unavailable", message) from None
    except ValueError as error:
        _save_run(request, {**metadata, "status": "error", "error": str(error)})
        raise APIError(409, "document_state_error", str(error)) from None
    if result["status"] == "error":
        _save_run(request, {**metadata, **result})
        status_code, code = {
            "model": (502, "model_error"),
            "database": (503, "database_unavailable"),
            "execution": (500, "agent_execution_error"),
        }[result["error_type"]]
        raise APIError(status_code, code, result["error"], trace=result["trace"])
    response = AskResponse.model_validate(
        {
            **result,
            "request_id": request.state.request_id,
        }
    )
    _save_run(request, response.model_dump())
    return response
