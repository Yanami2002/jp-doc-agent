"""HTTP の入出力を定義し、質問をモデル API の実行前に検証する。"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jp_doc_agent.embedding.encoder import validate_query


class AskRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra={
            "examples": [
                {
                    "question": "2024年度第2四半期の売上収益はいくらですか？",
                    "top_k": 5,
                },
                {
                    "question": "2021年1月1日時点の富士通の組織構成はどうなっていますか？",
                    "top_k": 5,
                },
            ]
        },
    )

    question: str = Field(min_length=1, max_length=32768, description="日本語の質問")
    document_id: int | None = Field(default=None, ge=1, description="対象文書 ID。省略時は全件")
    top_k: int = Field(default=5, ge=1, le=20, description="1 回の検索で取得する件数")

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        validate_query(value)
        return value


class HealthResponse(BaseModel):
    status: Literal["ok"]
    postgresql: str
    pgvector: str
    vector_distance: float


class DocumentResponse(BaseModel):
    id: int
    title: str
    dataset: str
    page_count: int
    sha256: str


class PageResponse(BaseModel):
    document_id: int
    title: str
    source_url: str
    page_number: int
    text: str
    has_unmapped_characters: bool


class SourceRange(BaseModel):
    page_number: int
    start_char: int
    end_char: int
    chunk_start_char: int
    chunk_end_char: int


class CitationResponse(BaseModel):
    id: int
    chunk_id: int
    document_id: int
    title: str
    source_url: str
    resolved_url: str
    quote: str
    quote_start_char: int
    quote_end_char: int
    page_numbers: list[int]
    sources: list[SourceRange]
    has_unmapped_characters: bool


class StatementResponse(BaseModel):
    text: str
    citation_ids: list[int]


class UsageResponse(BaseModel):
    embedding_tokens: int = Field(ge=0)
    answer_input_tokens: int = Field(ge=0)
    answer_output_tokens: int = Field(ge=0)
    answer_total_tokens: int = Field(ge=0)


class CountsResponse(BaseModel):
    searches: int
    page_reads: int
    tool_calls: int
    model_calls: int


class RetrievedEvidence(BaseModel):
    chunk_id: int
    document_id: int
    page_numbers: list[int]


class RetrievalResponse(BaseModel):
    profile_id: str
    top_k: int
    document_id: int | None
    results: list[RetrievedEvidence]


class AskResponse(BaseModel):
    request_id: str
    question: str
    status: Literal["answered", "insufficient_evidence"]
    answer: str
    statements: list[StatementResponse]
    citations: list[CitationResponse]
    missing_information: list[str]
    model: str
    requested_model: str
    prompt_version: str
    max_output_tokens: int
    elapsed_seconds: float
    usage: UsageResponse
    retrieval: RetrievalResponse
    counts: CountsResponse
    stop_reason: str
    limits: dict[str, int]
    trace: list[dict[str, Any]]


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    request_id: str
    error: ErrorDetail
    trace: list[dict[str, Any]] = Field(default_factory=list)
