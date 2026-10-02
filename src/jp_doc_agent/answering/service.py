"""1 回の検索と回答生成を連携し、引用を原文と照合する。"""

import re
from time import perf_counter

from sqlalchemy.engine import Engine

from jp_doc_agent.answering.generator import PROMPT_VERSION, OpenAIAnswerGenerator
from jp_doc_agent.answering.schema import AnswerDraft
from jp_doc_agent.embedding.encoder import OpenAIEncoder
from jp_doc_agent.llm import ModelError
from jp_doc_agent.retrieval.service import search


def _resolve_quote(hit: dict, quote: str) -> dict:
    start = hit["text"].find(quote)
    if not quote.strip() or start < 0:
        raise ModelError("引用原文が検索した本文に存在しません。回答を中止しました。")
    end = start + len(quote)
    sources = []
    for source in hit["sources"]:
        left = max(start, source["chunk_start_char"])
        right = min(end, source["chunk_end_char"])
        if left < right:
            sources.append(
                {
                    "page_number": source["page_number"],
                    "start_char": source["start_char"] + left - source["chunk_start_char"],
                    "end_char": source["start_char"] + right - source["chunk_start_char"],
                    "chunk_start_char": left,
                    "chunk_end_char": right,
                }
            )
    if not sources:
        raise ModelError("引用原文の出典を特定できません。回答を中止しました。")
    return {
        "chunk_id": hit["chunk_id"],
        "document_id": hit["document_id"],
        "title": hit["title"],
        "source_url": hit["source_url"],
        "resolved_url": hit["resolved_url"],
        "quote": quote,
        "quote_start_char": start,
        "quote_end_char": end,
        "page_numbers": [source["page_number"] for source in sources],
        "sources": sources,
        "has_unmapped_characters": hit["has_unmapped_characters"],
    }


def resolve_answer(draft: AnswerDraft, hits: list[dict]) -> dict:
    """引用番号と出典は検証済みの検索スナップショットから生成する。"""
    hit_by_id = {hit["chunk_id"]: hit for hit in hits}
    citations = []
    citation_ids = {}
    statements = []
    for statement in draft.statements:
        if re.search(r"\[\d+\]|https?://", statement.text):
            raise ModelError("結論に生成済みの引用番号または URL があります。回答を中止しました。")
        identifiers = []
        for citation in statement.citations:
            if citation.chunk_id not in hit_by_id:
                raise ModelError("引用先が今回の検索結果に存在しません。回答を中止しました。")
            key = (citation.chunk_id, citation.quote)
            if key not in citation_ids:
                identifier = len(citations) + 1
                citations.append(
                    {
                        "id": identifier,
                        **_resolve_quote(hit_by_id[citation.chunk_id], citation.quote),
                    }
                )
                citation_ids[key] = identifier
            if citation_ids[key] not in identifiers:
                identifiers.append(citation_ids[key])
        statements.append({"text": statement.text.strip(), "citation_ids": identifiers})
    if draft.status == "answered":
        answer = "\n".join(
            item["text"] + " " + "".join(f"[{identifier}]" for identifier in item["citation_ids"])
            for item in statements
        )
    else:
        answer = "取得した資料だけでは回答できません。\n不足情報: " + "／".join(
            draft.missing_information
        )
    return {
        "status": draft.status,
        "answer": answer,
        "statements": statements,
        "citations": citations,
        "missing_information": draft.missing_information,
    }


def ask(
    engine: Engine,
    encoder: OpenAIEncoder,
    generator: OpenAIAnswerGenerator,
    query: str,
    *,
    top_k: int = 5,
    document_id: int | None = None,
) -> dict:
    started = perf_counter()
    retrieval = search(engine, encoder, query, top_k=top_k, document_id=document_id)
    generated = generator.generate(query, retrieval["results"])
    resolved = resolve_answer(generated.draft, retrieval["results"])
    return {
        "question": query,
        **resolved,
        "model": generated.model,
        "requested_model": generator.model,
        "prompt_version": PROMPT_VERSION,
        "max_output_tokens": generator.max_output_tokens,
        "elapsed_seconds": round(perf_counter() - started, 3),
        "usage": {
            "embedding_tokens": retrieval["api_tokens"],
            "answer_input_tokens": generated.input_tokens,
            "answer_output_tokens": generated.output_tokens,
            "answer_total_tokens": generated.input_tokens + generated.output_tokens,
        },
        "retrieval": {
            **{
                key: retrieval[key]
                for key in (
                    "model",
                    "profile_id",
                    "top_k",
                    "document_id",
                    "coverage",
                )
            },
            "results": [
                {
                    key: hit[key]
                    for key in (
                        "rank",
                        "chunk_id",
                        "document_id",
                        "page_numbers",
                        "cosine_similarity",
                    )
                }
                for hit in retrieval["results"]
            ],
        },
    }
