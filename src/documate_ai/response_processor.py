"""Response post-processing and structured citation formatting.

Cleans up raw LLM output (question echoing, partial "no information"
disclaimers) and constructs structured source citations derived from retrieved
document chunks' metadata (source filename + page number).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from documate_ai.config import NO_INFO_RESPONSE, SOURCE_CONTENT_PREVIEW_LENGTH
from documate_ai.question_parser import remove_question_repetition
from documate_ai.schemas import Citation

if TYPE_CHECKING:
    from langchain_core.documents import Document

# Sentinel patterns for "I don't have enough information" responses
_NO_INFO_SPLIT_PATTERN = re.compile(
    r"(I (?:don't|do not) have enough information to answer [^.]*\.?)",
    re.IGNORECASE,
)
_NO_INFO_MARKERS = ("don't have enough information", "do not have enough information")


def _is_no_info_response(text: str) -> bool:
    """Check if the text represents a 'no information' response ignoring case and punctuation."""
    norm_text = re.sub(r"[^\w\s]", "", text).lower().strip()
    norm_text = re.sub(r"\bdo not\b", "dont", norm_text)
    norm_sentinel = re.sub(r"[^\w\s]", "", NO_INFO_RESPONSE).lower().strip()
    norm_sentinel = re.sub(r"\bdo not\b", "dont", norm_sentinel)
    return norm_text == norm_sentinel


def clean_answer(answer: str, *, question: str = "") -> str:
    """Clean raw LLM answer by removing question repetition and standardising disclaimers.

    Processing pipeline:
        1. If the response matches the "no information" boilerplate (ignoring case
           and punctuation), return the canonical ``NO_INFO_RESPONSE``.
        2. If the response contains a "no information" segment mixed with substantive
           content, split and clean each part.
        3. Remove any leading question echo from the response.

    Args:
        answer: Raw LLM-generated answer text.
        question: The user's question (used for echo detection).

    Returns:
        The cleaned answer string.
    """
    if not answer or not answer.strip():
        return answer

    # Step 1 — check if whole answer is the canonical "no info" response
    if _is_no_info_response(answer):
        return NO_INFO_RESPONSE

    # Step 2 — partial "no info" mixed with substantive content
    answer_lower = answer.lower()
    if any(marker in answer_lower for marker in _NO_INFO_MARKERS):
        parts = _NO_INFO_SPLIT_PATTERN.split(answer)
        if len(parts) > 1:
            cleaned_parts: list[str] = []
            for part in parts:
                stripped = part.strip()
                if not stripped:
                    continue
                if any(marker in stripped.lower() for marker in _NO_INFO_MARKERS):
                    cleaned_parts.append(stripped)
                else:
                    cleaned = remove_question_repetition(stripped, question)
                    if cleaned.strip():
                        cleaned_parts.append(cleaned.strip())
            answer = " ".join(cleaned_parts)

    # Step 3 — drop leading question echo from overall response
    answer = remove_question_repetition(answer, question)
    cleaned_answer = answer.strip()

    if _is_no_info_response(cleaned_answer):
        return NO_INFO_RESPONSE

    return cleaned_answer


def build_citations(docs: list[Document] | None) -> list[Citation]:
    """Extract deduplicated citations from retrieved documents in retrieval rank order.

    Deduplicates by ``(source, page)`` tuple so the same page from the same
    source is only cited once, preserving the rank order of first occurrence.
    Snippets are capped at ``SOURCE_CONTENT_PREVIEW_LENGTH`` characters.

    Args:
        docs: List of retrieved LangChain ``Document`` objects, or None.

    Returns:
        List of structured ``Citation`` objects.
    """
    if not docs:
        return []

    citations: list[Citation] = []
    seen: set[tuple[str, int]] = set()

    for doc in docs:
        source = str(doc.metadata.get("source") or "Unknown Document")
        raw_page = doc.metadata.get("page", 1)
        try:
            page = int(raw_page)
        except (ValueError, TypeError):
            page = 1

        dedup_key = (source, page)
        if dedup_key not in seen:
            seen.add(dedup_key)
            content = doc.page_content or ""
            snippet = content[:SOURCE_CONTENT_PREVIEW_LENGTH]
            citations.append(
                Citation(
                    source=source,
                    page=page,
                    snippet=snippet,
                )
            )

    return citations


def render_citations_markdown(citations: list[Citation]) -> str:
    """Render a list of citations into markdown format.

    Args:
        citations: List of structured ``Citation`` objects.

    Returns:
        Formatted markdown string with numbered citations, or empty string
        if *citations* is empty.
    """
    if not citations:
        return ""

    citation_text = "\n\n\nSources:\n"
    for index, citation in enumerate(citations, start=1):
        citation_text += f"{index}. {citation.source}, page {citation.page}\n"
    return citation_text


def format_response_with_citations(
    *,
    answer: str,
    source_documents: list[Document] | None = None,
    question: str = "",
) -> str:
    """Clean up an LLM answer and append formatted source citations.

    Thin wrapper around ``clean_answer``, ``build_citations``, and
    ``render_citations_markdown``. Keyword-only parameters prevent accidental
    argument transposition between old and new signatures.

    Args:
        answer: Raw LLM-generated answer text.
        source_documents: Retrieved chunks with source metadata (or None).
        question: The user's question (for echo detection and disclaimer cleaning).

    Returns:
        A cleaned answer string, optionally followed by formatted source citations.
    """
    cleaned = clean_answer(answer, question=question)
    if _is_no_info_response(cleaned):
        return NO_INFO_RESPONSE

    citations = build_citations(source_documents)
    citations_md = render_citations_markdown(citations)
    return f"{cleaned}{citations_md}"


def format_citations(source_documents: list[Document] | None) -> str:
    """Format source documents into numbered citation lines (backwards-compatible wrapper)."""
    citations = build_citations(source_documents)
    return render_citations_markdown(citations)


def extract_source_metadata(source_documents: list[Document] | None) -> list[dict[str, Any]]:
    """Extract unique source metadata dicts from retrieved documents.

    Reuses ``build_citations`` to deduplicate and structure sources.

    Args:
        source_documents: Retrieved ``Document`` objects, or None.

    Returns:
        A deduplicated list of dicts with keys ``"source"`` (filename),
        ``"page"`` (int), and ``"content"`` (preview snippet).
    """
    citations = build_citations(source_documents)
    return [
        {
            "source": c.source,
            "page": c.page,
            "content": c.snippet,
        }
        for c in citations
    ]
