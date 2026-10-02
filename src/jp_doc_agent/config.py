"""Read database configuration from environment variables or a local .env file."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
TOKEN_ENCODING = "cl100k_base"
ANSWER_MODEL = "gpt-4.1-mini-2025-04-14"


class OpenAISettings(BaseSettings):
    """モデル API の実行時だけ API キーを読み込む。"""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="OPENAI_", extra="ignore"
    )

    api_key: SecretStr = Field(min_length=1)


class AnsweringSettings(BaseSettings):
    """回答生成の設定は DB 操作・ベクトル化から分離する。"""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="OPENAI_", extra="ignore"
    )

    answer_model: str = Field(default=ANSWER_MODEL, min_length=1, max_length=100)
    answer_max_output_tokens: int = Field(default=2048, ge=256, le=8192)


class Settings(BaseSettings):
    """Environment variables override values in the current directory's .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="POSTGRES_",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = Field(default=5433, ge=1, le=65535)
    db: str = Field(default="jp_doc_agent", min_length=1)
    user: str = Field(default="jp_doc_agent", min_length=1)
    password: SecretStr = Field(min_length=1)

    @property
    def database_url(self) -> URL:
        """Build the connection URL without manually escaping the password."""
        return URL.create(
            drivername="postgresql+psycopg",
            username=self.user,
            password=self.password.get_secret_value(),
            host=self.host,
            port=self.port,
            database=self.db,
        )
