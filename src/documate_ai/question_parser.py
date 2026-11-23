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

# Common leading conjunctions / polite words that precede a question word
_LEADING_PREFIXES = frozenset({"and", "or", "so", "also", "then", "but", "please"})


def _starts_with_question_word(fragment: str) -> bool:
    """Check if a text fragment begins with a question indicator word.

    Permits conversational prefixes (e.g. "and", "or", "also", "please")
    preceding the question word.

    Args:
        fragment: The text fragment to evaluate.

    Returns:
        True if the fragment starts with a question word, False otherwise.
    """
    cleaned = re.sub(r"^[^\w]+", "", fragment).strip()
    if not cleaned:
        return False

    words = [re.sub(r"[^\w]", "", w).lower() for w in cleaned.split()]
    words = [w for w in words if w]
    if not words:
        return False

    first_word = words[0]
    if first_word in QUESTION_INDICATOR_WORDS:
        return True

    if first_word in _LEADING_PREFIXES and len(words) > 1 and words[1] in QUESTION_INDICATOR_WORDS:
        return True

    return False


def detect_multiple_questions(text: str) -> tuple[bool, list[str]]:
    """Detect whether the user's input contains multiple questions.

    Splits input on ``?`` characters and verifies that each question fragment
    starts with a question-indicator word (e.g. *what*, *why*, *how*). Also
    handles trailing fragments that lack a terminal ``?`` if they start with
    a question word.

    Args:
        text: The raw user input string.

    Returns:
        A tuple of ``(is_multi, questions)`` where ``is_multi`` is
        ``True`` if two or more valid questions were detected, and
        ``questions`` is the list of individual question strings with trailing
        ``?``. When only one (or zero) questions are found, the original
        *text* is returned as the sole list element with ``is_multi=False``.
    """
    if not text or not text.strip():
        return False, [text]

    fragments = re.split(r"\?", text)
    questions: list[str] = []

    for fragment in fragments:
        stripped = fragment.strip()
        if not stripped:
            continue
        if _starts_with_question_word(stripped):
            questions.append(stripped.rstrip("?").strip() + "?")

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
    if not response or not question:
        return response

    question_words = {
        word
        for word in re.findall(r"\b\w+\b", question.lower())
        if len(word) > MIN_MEANINGFUL_WORD_LENGTH and word not in COMMON_STOP_WORDS
    }
    if not question_words:
        return response

    sentences = re.split(r"(?<=[.!?])\s+", response)
    if not sentences or len(sentences) <= 1:
        return response

    first_sentence_words = set(re.findall(r"\b\w+\b", sentences[0].lower()))
    overlap_ratio = len(question_words & first_sentence_words) / max(
        1,
        len(question_words),
    )

    if overlap_ratio > QUESTION_OVERLAP_THRESHOLD:
        return " ".join(sentences[1:]).strip()

    return response
