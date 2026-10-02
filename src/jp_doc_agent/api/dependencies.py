"""DB はアプリケーション単位、モデル接続は問答リクエスト単位で管理する。"""

from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import Request
from pydantic import ValidationError
from sqlalchemy.engine import Engine

from jp_doc_agent.answering.generator import OpenAIAnswerGenerator
from jp_doc_agent.api.errors import APIError
from jp_doc_agent.config import AnsweringSettings, OpenAISettings
from jp_doc_agent.embedding.encoder import OpenAIEncoder


def get_engine(request: Request) -> Engine:
    return request.app.state.engine


@contextmanager
def model_resources() -> Iterator[tuple[OpenAIEncoder, OpenAIAnswerGenerator]]:
    try:
        openai_settings = OpenAISettings()
        answer_settings = AnsweringSettings()
    except ValidationError:
        raise APIError(
            503,
            "model_configuration_error",
            "サーバーの OPENAI_API_KEY・回答モデル・出力 Token 上限を確認してください。",
        ) from None
    encoder = OpenAIEncoder.from_settings(openai_settings)
    try:
        try:
            generator = OpenAIAnswerGenerator(
                encoder.client,
                model=answer_settings.answer_model,
                max_output_tokens=answer_settings.answer_max_output_tokens,
            )
        except ValueError:
            raise APIError(
                503,
                "model_configuration_error",
                "サーバーの回答モデルと出力 Token 上限を確認してください。",
            ) from None
        yield encoder, generator
    finally:
        encoder.client.close()
