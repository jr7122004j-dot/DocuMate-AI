"""Response post-processing and citation formatting.

Cleans up raw LLM output (question echoing, partial "no information"
disclaimers) and appends source citations derived from the retrieved
document chunks' metadata (source filename + page number).
"""

from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

from documate_ai.config import SOURCE_CONTENT_PREVIEW_LENGTH
from documate_ai.question_parser import remove_question_repetition

if TYPE_CHECKING:
    from langchain_core.documents import Document

# Sentinel patterns for "I don't know" responses
_NO_INFO_EXACT = "I don't have enough information to answer this question."
_NO_INFO_MARKER = "don't have enough information"
_NO_INFO_SPLIT_PATTERN = re.compile(
    r"(I don't have enough information to answer [^.]*\.)",
    re.IGNORECASE,
)


def format_citations(source_documents: list[Document]) -> str:
    """Format source documents into numbered citation lines.

    Deduplicates sources by ``(filename, page)`` so the same page
    is not cited twice.

    Args:
        source_documents: List of retrieved ``Document`` objects, each
            with ``metadata["source"]`` and ``metadata["page"]``.

    Returns:
        A formatted string like::

            Sources:
            1. report.pdf, page 12
            2. report.pdf, page 15

        Returns an empty string if no source documents are provided.
    """
    if not source_documents:
        return ""

    unique_sources: dict[str, dict[str, Any]] = {}
    for doc in source_documents:
        source_name = doc.metadata.get("source", "Unknown Document")
        page_number = doc.metadata.get("page", "N/A")
        dedup_key = f"{source_name}-{page_number}"
        if dedup_key not in unique_sources:
            unique_sources[dedup_key] = {
                "source": source_name,
                "page": page_number,
            }

    if not unique_sources:
        return ""

    citation_text = "\n\n\nSources:\n"
    for index, source_info in enumerate(unique_sources.values(), start=1):
        citation_text += (
            f"{index}. {source_info['source']}, "
            f"page {source_info['page']}\n"
        )
    return citation_text


def format_response_with_citations(
    answer: str,
    source_documents: list[Document] | None,
    question: str,
) -> str:
    """Clean up an LLM answer and append source citations.

    Processing pipeline:
        1. If the response is the exact "no information" boilerplate,
           return it unchanged.
        2. If the response *contains* a "no information" segment mixed
           with substantive content, split and clean each part.
        3. Remove any leading question echo from the response.
        4. Append deduplicated source citations.

    Args:
        answer: Raw LLM-generated answer text.
        source_documents: Retrieved chunks with source metadata (or None).
        question: The original user question (for echo detection).

    Returns:
        A clean answer string followed by formatted source citations.
    """
    # Step 1 — verbatim "no information" reply
    if answer.strip() == _NO_INFO_EXACT:
        return answer

    # Step 2 — partial "no information" mixed with substantive content
    if _NO_INFO_MARKER in answer.lower():
        parts = _NO_INFO_SPLIT_PATTERN.split(answer)
        if len(parts) > 1:
            cleaned_parts: list[str] = []
            for part in parts:
                stripped = part.strip()
                if not stripped:
                    continue
                if _NO_INFO_MARKER in stripped.lower():
                    cleaned_parts.append(stripped)
                else:
                    cleaned = remove_question_repetition(stripped, question)
                    if cleaned.strip():
                        cleaned_parts.append(cleaned.strip())
            answer = " ".join(cleaned_parts)

    # Step 3 — drop leading question echo
    answer = remove_question_repetition(answer, question)

    # Step 4 — deduplicate and format source citations
    if source_documents:
        answer += format_citations(source_documents)

    return answer


def extract_source_metadata(source_documents: list[Document]) -> list[dict]:
    """Extract unique source metadata dicts from retrieved documents.

    Args:
        source_documents: Retrieved ``Document`` objects.

    Returns:
        A deduplicated list of dicts with keys ``"source"`` (filename),
        ``"page"`` (int), and ``"content"`` (preview snippet).
    """
    seen: set[str] = set()
    results: list[dict] = []

    for doc in source_documents:
        source_name = doc.metadata.get("source", "Unknown Document")
        page_number = doc.metadata.get("page", "N/A")
        dedup_key = f"{source_name}-{page_number}"

        if dedup_key not in seen:
            seen.add(dedup_key)
            content_preview = doc.page_content[:SOURCE_CONTENT_PREVIEW_LENGTH]
            results.append({
                "source": source_name,
                "page": page_number,
                "content": content_preview,
            })

    return results
