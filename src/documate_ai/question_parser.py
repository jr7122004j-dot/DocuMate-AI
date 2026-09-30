"""Smart question parsing utilities.

Provides:
    1. Multi-question detection — splits compound user prompts like
       "What is X? And what about Y?" into individual sub-questions.
    2. Question repetition removal — strips the LLM's tendency to echo
       the user's question before answering.
"""

from __future__ import annotations

import re

from documate_ai.config import (
    COMMON_STOP_WORDS,
    MIN_MEANINGFUL_WORD_LENGTH,
    QUESTION_INDICATOR_WORDS,
    QUESTION_OVERLAP_THRESHOLD,
)


def detect_multiple_questions(text: str) -> tuple[bool, list[str]]:
    """Detect whether the user's input contains multiple questions.

    The heuristic splits on ``?`` characters and retains any fragment
    whose words overlap with a predefined set of English question-
    indicator words (e.g. *what*, *why*, *how*).

    Args:
        text: The raw user input string.

    Returns:
        A tuple of ``(is_multi, questions)`` where ``is_multi`` is
        ``True`` if two or more questions were detected, and
        ``questions`` is the list of individual question strings.
        When only one (or zero) questions are found, the original
        *text* is returned as the sole list element.
    """
    fragments = re.split(r"\?", text)
    questions: list[str] = []

    for fragment in fragments:
        stripped = fragment.strip()
        if not stripped:
            continue
        words = stripped.lower().split()
        if any(word in QUESTION_INDICATOR_WORDS for word in words):
            questions.append(stripped + "?")

    if len(questions) > 1:
        return True, questions
    return False, [text]


def remove_question_repetition(response: str, question: str) -> str:
    """Remove a leading sentence that merely restates the user's question.

    Many LLMs echo the question before answering. This function detects
    that pattern by measuring word-level overlap between the first
    sentence of *response* and the original *question*. If the overlap
    ratio exceeds ``QUESTION_OVERLAP_THRESHOLD``, the first sentence
    is dropped.

    Only "meaningful" words — those longer than
    ``MIN_MEANINGFUL_WORD_LENGTH`` characters and not in
    ``COMMON_STOP_WORDS`` — are considered when computing overlap.

    Args:
        response: The LLM-generated response text.
        question: The original user question.

    Returns:
        The response with any leading question echo removed, or the
        original text unchanged if no repetition was detected.
    """
    question_words = {
        word
        for word in re.findall(r"\b\w+\b", question.lower())
        if len(word) > MIN_MEANINGFUL_WORD_LENGTH and word not in COMMON_STOP_WORDS
    }

    sentences = re.split(r"(?<=[.!?])\s+", response)
    if not sentences or len(sentences) <= 1:
        return response

    first_sentence_words = set(re.findall(r"\b\w+\b", sentences[0].lower()))
    overlap_ratio = len(question_words & first_sentence_words) / max(
        1,
        len(question_words),
    )

    if overlap_ratio > QUESTION_OVERLAP_THRESHOLD:
        return " ".join(sentences[1:])

    return response
