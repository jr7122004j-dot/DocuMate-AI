"""Unit tests for the response processor and schemas modules.

Tests answer cleaning, structured citation building, markdown rendering,
and DTO serialization.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from documate_ai.config import NO_INFO_RESPONSE, SOURCE_CONTENT_PREVIEW_LENGTH
from documate_ai.response_processor import (
    build_citations,
    clean_answer,
    extract_source_metadata,
    format_citations,
    format_response_with_citations,
    render_citations_markdown,
)
from documate_ai.schemas import ChatResponse, Citation


class TestSchemas:
    """Tests for Citation and ChatResponse data schemas."""

    def test_citation_to_dict(self):
        citation = Citation(source="annual_report.pdf", page=12, snippet="Revenue rose 10%.")
        d = citation.to_dict()
        assert d == {
            "source": "annual_report.pdf",
            "page": 12,
            "snippet": "Revenue rose 10%.",
        }

    def test_chat_response_defaults_and_to_dict(self):
        resp = ChatResponse(answer="Here is the answer.")
        assert resp.answer == "Here is the answer."
        assert resp.citations == []
        assert resp.from_cache is False
        assert resp.similarity is None
        assert resp.matched_question is None
        assert resp.standalone_question is None
        assert resp.sub_questions == []
        assert resp.timings == {}

        resp_with_citations = ChatResponse(
            answer="Summary.",
            citations=[Citation(source="doc.pdf", page=1, snippet="Text")],
            from_cache=True,
            similarity=0.96,
        )
        d = resp_with_citations.to_dict()
        assert d["from_cache"] is True
        assert d["similarity"] == 0.96
        assert len(d["citations"]) == 1
        assert d["citations"][0]["source"] == "doc.pdf"


class TestCleanAnswer:
    """Tests for clean_answer."""

    def test_exact_no_info_response(self):
        assert clean_answer(NO_INFO_RESPONSE) == NO_INFO_RESPONSE

    def test_no_info_case_and_punctuation_normalisation(self):
        variations = [
            "i don't have enough information to answer this question",
            "I don't have enough information to answer this question.",
            "  I don't have enough information to answer this question!  ",
            "I DO NOT HAVE ENOUGH INFORMATION TO ANSWER THIS QUESTION.",
            "I do not have enough information to answer this question",
        ]
        for var in variations:
            assert clean_answer(var) == NO_INFO_RESPONSE

    def test_mixed_substantive_and_no_info_disclaimer(self):
        answer = (
            "The net profit was 1.2 million. "
            "I don't have enough information to answer the second part."
        )
        cleaned = clean_answer(answer, question="What was profit and revenue?")
        assert "net profit was 1.2 million" in cleaned
        assert "I don't have enough information to answer the second part." in cleaned

    def test_removes_question_echo_from_answer(self):
        question = "What was the total revenue in 2023?"
        answer = (
            "What was the total revenue in 2023? "
            "The total revenue in 2023 reached 45 million USD."
        )
        cleaned = clean_answer(answer, question=question)
        assert cleaned == "The total revenue in 2023 reached 45 million USD."

    def test_empty_or_whitespace_answer(self):
        assert clean_answer("") == ""
        assert clean_answer("   ") == "   "


class TestBuildCitations:
    """Tests for build_citations."""

    def test_empty_or_none_docs(self):
        assert build_citations(None) == []
        assert build_citations([]) == []

    def test_deduplicates_by_source_and_page(self):
        docs = [
            Document(page_content="Snippet A1", metadata={"source": "report.pdf", "page": 1}),
            Document(page_content="Snippet A2", metadata={"source": "report.pdf", "page": 1}),
            Document(page_content="Snippet B1", metadata={"source": "report.pdf", "page": 2}),
        ]
        citations = build_citations(docs)
        assert len(citations) == 2
        assert citations[0].source == "report.pdf"
        assert citations[0].page == 1
        assert citations[0].snippet == "Snippet A1"
        assert citations[1].page == 2

    def test_tuple_dedup_avoids_string_collision(self):
        """Distinct (source, page) combinations that would collide under string concat."""
        # 'doc-1' page 2 vs 'doc' page '1-2' (or similar collision hazard)
        docs = [
            Document(page_content="Content 1", metadata={"source": "file-1.pdf", "page": 2}),
            Document(page_content="Content 2", metadata={"source": "file", "page": 1}),
        ]
        citations = build_citations(docs)
        assert len(citations) == 2

    def test_preserves_retrieval_rank_order(self):
        docs = [
            Document(page_content="From Page 5", metadata={"source": "a.pdf", "page": 5}),
            Document(page_content="From Page 2", metadata={"source": "b.pdf", "page": 2}),
            Document(page_content="From Page 1", metadata={"source": "a.pdf", "page": 1}),
        ]
        citations = build_citations(docs)
        assert [(c.source, c.page) for c in citations] == [
            ("a.pdf", 5),
            ("b.pdf", 2),
            ("a.pdf", 1),
        ]

    def test_snippet_capped_at_source_preview_length(self):
        long_content = "X" * (SOURCE_CONTENT_PREVIEW_LENGTH + 200)
        docs = [Document(page_content=long_content, metadata={"source": "a.pdf", "page": 1})]
        citations = build_citations(docs)
        assert len(citations[0].snippet) == SOURCE_CONTENT_PREVIEW_LENGTH

    def test_handles_string_or_missing_page(self):
        docs = [
            Document(page_content="Text 1", metadata={"source": "a.pdf", "page": "4"}),
            Document(page_content="Text 2", metadata={"source": "b.pdf"}),
        ]
        citations = build_citations(docs)
        assert citations[0].page == 4
        assert citations[1].page == 1


class TestRenderCitationsMarkdown:
    """Tests for render_citations_markdown."""

    def test_empty_citations(self):
        assert render_citations_markdown([]) == ""

    def test_renders_numbered_list(self):
        citations = [
            Citation(source="report.pdf", page=3, snippet="abc"),
            Citation(source="notes.pdf", page=7, snippet="xyz"),
        ]
        md = render_citations_markdown(citations)
        expected = "\n\n\nSources:\n1. report.pdf, page 3\n2. notes.pdf, page 7\n"
        assert md == expected


class TestFormatResponseWithCitations:
    """Tests for format_response_with_citations."""

    def test_keyword_only_parameters_enforced(self):
        from typing import Any

        docs = [Document(page_content="Text", metadata={"source": "a.pdf", "page": 1})]
        fn: Any = format_response_with_citations
        with pytest.raises(TypeError):
            # Positional arguments must raise TypeError
            fn("Answer", docs, "Question")

    def test_no_info_response_omits_citations(self):
        docs = [Document(page_content="Irrelevant", metadata={"source": "a.pdf", "page": 1})]
        res = format_response_with_citations(
            answer=NO_INFO_RESPONSE,
            source_documents=docs,
            question="What is X?",
        )
        assert res == NO_INFO_RESPONSE
        assert "Sources:" not in res

    def test_substantive_response_includes_citations(self):
        docs = [
            Document(page_content="Relevant info", metadata={"source": "facts.pdf", "page": 4})
        ]
        res = format_response_with_citations(
            answer="The facts show XYZ.",
            source_documents=docs,
            question="What do facts show?",
        )
        assert "The facts show XYZ." in res
        assert "Sources:\n1. facts.pdf, page 4" in res


class TestExtractSourceMetadataAndFormatCitations:
    """Tests for extract_source_metadata and format_citations helpers."""

    def test_extract_source_metadata_delegates_to_build_citations(self):
        docs = [
            Document(page_content="Snippet content", metadata={"source": "doc.pdf", "page": 2}),
            Document(page_content="Duplicate", metadata={"source": "doc.pdf", "page": 2}),
        ]
        meta = extract_source_metadata(docs)
        assert len(meta) == 1
        assert meta[0] == {
            "source": "doc.pdf",
            "page": 2,
            "content": "Snippet content",
        }

    def test_format_citations_backwards_compatible(self):
        docs = [
            Document(page_content="Test", metadata={"source": "data.pdf", "page": 10}),
        ]
        result = format_citations(docs)
        assert "Sources:\n1. data.pdf, page 10" in result
