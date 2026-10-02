"""HTTP ステータスと秘密情報を含まないエラー本文を対応付ける。"""

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException

from jp_doc_agent.api.schema import ErrorDetail, ErrorResponse


class APIError(Exception):
    def __init__(self, status_code: int, code: str, message: str, *, trace: list | None = None):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.trace = trace or []


def error_response(request: Request, error: APIError) -> JSONResponse:
    body = ErrorResponse(
        request_id=request.state.request_id,
        error=ErrorDetail(code=error.code, message=error.message),
        trace=error.trace,
    )
    return JSONResponse(
        status_code=error.status_code,
        content=body.model_dump(),
        headers={"X-Request-ID": request.state.request_id},
    )


async def handle_api_error(request: Request, error: APIError) -> JSONResponse:
    return error_response(request, error)


async def handle_validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
    return error_response(
        request,
        APIError(
            422,
            "invalid_request",
            "入力が不正です。質問は空白以外の文字を含む 1〜8191 Token・32768 文字以内、"
            "top_k は 1〜20、文書 ID・ページ番号は正の整数です。"
            "定義にない項目は指定できません。",
        ),
    )


async def handle_database_error(request: Request, error: SQLAlchemyError) -> JSONResponse:
    return error_response(
        request, APIError(503, "database_unavailable", "データベースに接続・アクセスできません。")
    )


async def handle_http_error(request: Request, error: HTTPException) -> JSONResponse:
    messages = {
        404: ("not_found", "指定された API が見つかりません。"),
        405: ("method_not_allowed", "この API では指定された HTTP メソッドを利用できません。"),
    }
    code, message = messages.get(
        error.status_code, ("http_error", "HTTP リクエストに失敗しました。")
    )
    response = error_response(request, APIError(error.status_code, code, message))
    if error.headers:
        response.headers.update(error.headers)
    return response


async def handle_unexpected_error(request: Request, error: Exception) -> JSONResponse:
    return error_response(
        request, APIError(500, "internal_error", "サーバー内部でエラーが発生しました。")
    )
