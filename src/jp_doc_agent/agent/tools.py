"""検索・文書一覧・原文参照の許可された道具。モデルは SQL を生成しない。"""

from sqlalchemy.engine import Engine

from jp_doc_agent.agent.schema import ToolDecision
from jp_doc_agent.embedding.encoder import OpenAIEncoder
from jp_doc_agent.ingestion.service import list_documents
from jp_doc_agent.retrieval.service import read_page_evidence, search


class ResearchTools:
    def __init__(
        self, engine: Engine, encoder: OpenAIEncoder, *, document_id: int | None, top_k: int
    ):
        self.engine = engine
        self.encoder = encoder
        self.document_id = document_id
        self.top_k = top_k

    def normalize(self, decision: ToolDecision) -> ToolDecision:
        if decision.action in ("search", "page") and self.document_id is not None:
            if decision.document_id not in (None, self.document_id):
                raise ValueError("指定された文書の範囲外にはアクセスできません。")
            return decision.model_copy(update={"document_id": self.document_id})
        return decision

    def execute(self, decision: ToolDecision) -> dict:
        if decision.action == "search":
            return search(
                self.engine,
                self.encoder,
                decision.query,
                document_id=decision.document_id,
                top_k=self.top_k,
            )
        if decision.action == "documents":
            documents = list_documents(self.engine)
            if self.document_id is not None:
                documents = [item for item in documents if item["id"] == self.document_id]
            return {
                "documents": [
                    {key: item[key] for key in ("id", "title", "page_count")}
                    for item in documents[:100]
                ],
                "truncated": len(documents) > 100,
            }
        if decision.action == "page":
            return read_page_evidence(self.engine, decision.document_id, decision.page_number)
        raise ValueError("この処理は道具の呼び出しではありません。")
