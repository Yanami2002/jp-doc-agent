"""OpenAI Embedding API の入力・応答を検証する。"""

import hashlib
import json
import math
from dataclasses import dataclass

from openai import APIConnectionError, APIError, APIStatusError, OpenAI
from openai.types import CreateEmbeddingResponse
from pydantic import ValidationError

from jp_doc_agent.chunking.splitter import count_tokens
from jp_doc_agent.config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    TOKEN_ENCODING,
    OpenAISettings,
)

MAX_QUERY_TOKENS = 8191


def validate_query(query: str) -> int:
    token_count = count_tokens(query)
    if not query.strip() or not 1 <= token_count <= MAX_QUERY_TOKENS:
        raise ValueError(f"質問は空白以外の文字を含む 1〜{MAX_QUERY_TOKENS} Token が必要です。")
    return token_count


class EmbeddingError(RuntimeError):
    """API の本文や秘密情報を含めない、利用者向けのエラー。"""


def profile_config() -> dict:
    return {
        "provider": "openai",
        "endpoint": "https://api.openai.com/v1/embeddings",
        "model": EMBEDDING_MODEL,
        "dimensions": EMBEDDING_DIMENSIONS,
        "encoding": TOKEN_ENCODING,
        "input_policy": "raw-chunk-text-v1",
    }


def profile_id() -> str:
    return hashlib.sha256(json.dumps(profile_config(), sort_keys=True).encode()).hexdigest()


def input_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: list[list[float]]
    total_tokens: int


class OpenAIEncoder:
    def __init__(self, client: OpenAI):
        self.client = client

    @classmethod
    def from_settings(cls, settings: OpenAISettings) -> "OpenAIEncoder":
        return cls(
            OpenAI(
                api_key=settings.api_key.get_secret_value(),
                base_url="https://api.openai.com/v1",
                timeout=30.0,
                max_retries=2,
            )
        )

    def encode(self, texts: list[str]) -> EmbeddingBatch:
        if not 1 <= len(texts) <= 32:
            raise ValueError("Embedding の 1 回の入力は 1〜32 件にしてください。")
        if any(not text.strip() or not 1 <= count_tokens(text) <= 300 for text in texts):
            raise ValueError("Embedding の本文は空白以外の文字を含む 1〜300 Token が必要です。")
        return self._request(texts)

    def encode_query(self, query: str) -> EmbeddingBatch:
        """質問の長さはチャンクの 300 Token 上限とは別に検証する。"""
        validate_query(query)
        return self._request([query])

    def _request(self, texts: list[str]) -> EmbeddingBatch:
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                dimensions=EMBEDDING_DIMENSIONS,
                encoding_format="float",
                input=texts,
            )
        except APIStatusError as error:
            raise EmbeddingError(
                f"Embedding API が HTTP {error.status_code} を返しました。"
                "API キー・利用枠・接続状況を確認し、再実行してください。"
            ) from None
        except APIConnectionError:
            raise EmbeddingError(
                "Embedding API に接続できません。接続状況を確認してください。"
            ) from None
        except APIError:
            raise EmbeddingError(
                "Embedding API の応答を読み取れません。再実行してください。"
            ) from None

        try:
            response = CreateEmbeddingResponse.model_validate(response.model_dump(), strict=True)
        except ValidationError:
            raise EmbeddingError(
                "Embedding API の応答形式が不正です。保存を中止しました。"
            ) from None
        if response.model != EMBEDDING_MODEL or len(response.data) != len(texts):
            raise EmbeddingError(
                "Embedding のモデル名または応答件数が不正です。保存を中止しました。"
            )
        ordered = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in ordered] != list(range(len(texts))):
            raise EmbeddingError("Embedding の入力番号が不正です。保存を中止しました。")
        vectors = [item.embedding for item in ordered]
        for vector in vectors:
            if (
                len(vector) != EMBEDDING_DIMENSIONS
                or any(not math.isfinite(value) for value in vector)
                or not any(value != 0 for value in vector)
            ):
                raise EmbeddingError("Embedding の次元数または値が不正です。保存を中止しました。")
        if response.usage.total_tokens < 0:
            raise EmbeddingError("Embedding の Token 使用量が不正です。保存を中止しました。")
        return EmbeddingBatch(vectors, response.usage.total_tokens)
