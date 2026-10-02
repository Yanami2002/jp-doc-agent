"""構造化応答の検証・引用照合・根拠不足・CLI を実 API に接続せず検証する。"""

import json

import httpx
import pytest

from jp_doc_agent.answering.citations import resolve_answer
from jp_doc_agent.answering.generator import OpenAIAnswerGenerator
from jp_doc_agent.answering.schema import AnswerDraft
from jp_doc_agent.config import ANSWER_MODEL
from jp_doc_agent.llm import ModelError


@pytest.fixture
def hit():
    return {
        "chunk_id": 11,
        "document_id": 2,
        "title": "申請資料",
        "source_url": "https://example.com/original.pdf",
        "resolved_url": "https://example.com/cache.pdf",
        "text": "学生が対象です。\n期限は三月です。",
        "page_numbers": [2, 3],
        "has_unmapped_characters": False,
        "sources": [
            {
                "page_number": 2,
                "start_char": 20,
                "end_char": 28,
                "chunk_start_char": 0,
                "chunk_end_char": 8,
            },
            {
                "page_number": 3,
                "start_char": 0,
                "end_char": 8,
                "chunk_start_char": 9,
                "chunk_end_char": 17,
            },
        ],
    }


def draft_for(hit, *, quote=None, text="対象者は学生です。"):
    return {
        "status": "answered",
        "statements": [
            {
                "text": text,
                "citations": [
                    {
                        "chunk_id": hit["chunk_id"],
                        "quote": quote or hit["text"],
                    }
                ],
            }
        ],
        "missing_information": [],
    }


def test_quotes_preserve_cross_page_ranges_and_deduplicate(hit):
    draft = draft_for(hit, quote="対象です。\n期限は")
    draft["statements"] *= 2
    draft["statements"][0]["citations"] *= 2
    result = resolve_answer(AnswerDraft.model_validate(draft), [hit])
    assert len(result["citations"]) == 1
    assert result["answer"].count("[1]") == 2
    citation = result["citations"][0]
    assert citation["page_numbers"] == [2, 3]
    assert citation["source_url"] == hit["source_url"]
    assert citation["sources"] == [
        {
            "page_number": 2,
            "start_char": 23,
            "end_char": 28,
            "chunk_start_char": 3,
            "chunk_end_char": 8,
        },
        {
            "page_number": 3,
            "start_char": 0,
            "end_char": 3,
            "chunk_start_char": 9,
            "chunk_end_char": 12,
        },
    ]


def test_quote_only_cites_pages_it_intersects(hit):
    result = resolve_answer(
        AnswerDraft.model_validate(draft_for(hit, quote="期限は三月です。")), [hit]
    )
    assert result["citations"][0]["page_numbers"] == [3]


@pytest.mark.parametrize("problem", ["unknown_chunk", "modified_quote", "marker", "url"])
def test_invalid_citations_are_rejected(hit, problem):
    draft = draft_for(hit)
    if problem == "unknown_chunk":
        draft["statements"][0]["citations"][0]["chunk_id"] = 12
    elif problem == "modified_quote":
        draft["statements"][0]["citations"][0]["quote"] = hit["text"].replace("\n", " ")
    elif problem == "marker":
        draft["statements"][0]["text"] += " [99]"
    else:
        draft["statements"][0]["text"] += " https://fake.example.com"
    with pytest.raises(ModelError):
        resolve_answer(AnswerDraft.model_validate(draft), [hit])


def test_generator_sends_only_evidence_and_uses_strict_schema(
    hit, encoder_factory, answer_response
):
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/responses"
        assert body["model"] == ANSWER_MODEL
        assert body["store"] is False
        assert body["max_output_tokens"] == 2048
        assert body["text"]["format"]["type"] == "json_schema"
        assert body["text"]["format"]["strict"] is True
        payload = json.loads(body["input"][1]["content"])
        assert payload == {
            "question": "対象者は？",
            "evidence": [
                {
                    "chunk_id": 11,
                    "title": hit["title"],
                    "text": hit["text"],
                }
            ],
        }
        return httpx.Response(200, json=answer_response(draft_for(hit)))

    generator = OpenAIAnswerGenerator(encoder_factory(handler).client)
    result = generator.generate("対象者は？", [hit])
    assert result.input_tokens == 50
    assert result.output_tokens == 20


@pytest.mark.parametrize(
    "problem",
    [
        "incomplete",
        "refusal",
        "empty",
        "malformed",
        "missing_usage",
        "negative_usage",
        "string_usage",
        "invalid_status",
        "no_citations",
        "empty_quote",
        "extra_url",
        "insufficient_claim",
        "missing_information",
        "blank_statement",
    ],
)
def test_bad_api_output_is_rejected_without_exposing_body(
    hit, encoder_factory, answer_response, problem, capsys
):
    def handler(request):
        draft = draft_for(hit)
        response = answer_response(draft)
        if problem == "incomplete":
            response["status"] = "incomplete"
        elif problem == "refusal":
            response["output"][0]["content"] = [{"type": "refusal", "refusal": "test-key"}]
        elif problem == "empty":
            response["output"] = []
        elif problem == "malformed":
            response["output"][0]["content"][0]["text"] = "test-key invalid JSON"
        elif problem == "missing_usage":
            del response["usage"]
        elif problem == "negative_usage":
            response["usage"]["input_tokens"] = -1
        elif problem == "string_usage":
            response["usage"]["input_tokens"] = "test-key"
        else:
            if problem == "invalid_status":
                draft["status"] = "invented"
            elif problem == "no_citations":
                draft["statements"][0]["citations"] = []
            elif problem == "empty_quote":
                draft["statements"][0]["citations"][0]["quote"] = " "
            elif problem == "extra_url":
                draft["statements"][0]["citations"][0]["url"] = "https://fake.example.com"
            elif problem == "insufficient_claim":
                draft["status"] = "insufficient_evidence"
                draft["missing_information"] = ["対象年度の情報。"]
            elif problem == "missing_information":
                draft.update(status="insufficient_evidence", statements=[])
            else:
                draft["statements"][0]["text"] = " "
            response = answer_response(draft)
        return httpx.Response(200, json=response)

    with pytest.raises(ModelError) as error:
        OpenAIAnswerGenerator(encoder_factory(handler).client).generate("対象者は？", [hit])
    assert "test-key" not in str(error.value)
    assert "test-key" not in capsys.readouterr().err


@pytest.mark.parametrize("status", [401, 429, 500])
def test_api_errors_are_safe(hit, encoder_factory, status):
    def handler(request):
        return httpx.Response(status, json={"error": {"message": "test-key secret-value"}})

    with pytest.raises(match=f"HTTP {status}") as error:
        OpenAIAnswerGenerator(encoder_factory(handler).client).generate("対象者は？", [hit])
    assert "secret-value" not in str(error.value)
    assert "test-key" not in str(error.value)
