"""Unit tests for the question parser module.

Tests multi-question detection heuristics and question-echo removal.
"""

from __future__ import annotations

import pytest

from documate_ai.question_parser import (
    detect_multiple_questions,
    remove_question_repetition,
)


class TestDetectMultipleQuestions:
    """Tests for detect_multiple_questions."""

    def test_single_question_with_mark(self):
        text = "What is the capital of France?"
        is_multi, questions = detect_multiple_questions(text)
        assert is_multi is False
        assert questions == [text]

    def test_single_question_without_mark(self):
        text = "What is the capital of France"
        is_multi, questions = detect_multiple_questions(text)
        assert is_multi is False
        assert questions == [text]

    def test_two_questions_both_with_marks(self):
        text = "What is machine learning? How does deep learning differ?"
        is_multi, questions = detect_multiple_questions(text)
        assert is_multi is True
        assert len(questions) == 2
        assert questions == [
            "What is machine learning?",
            "How does deep learning differ?",
        ]

    def test_three_questions(self):
        text = "What is A? Why is B? Where is C?"
        is_multi, questions = detect_multiple_questions(text)
        assert is_multi is True
        assert len(questions) == 3
        assert questions == ["What is A?", "Why is B?", "Where is C?"]

    def test_second_question_with_conjunction_prefix(self):
        text = "What is the revenue for 2023? And what about the net income?"
        is_multi, questions = detect_multiple_questions(text)
        assert is_multi is True
        assert len(questions) == 2
        assert questions == [
            "What is the revenue for 2023?",
            "And what about the net income?",
        ]

    def test_last_question_lacks_trailing_question_mark(self):
        text = "What is Python? How does garbage collection work"
        is_multi, questions = detect_multiple_questions(text)
        assert is_multi is True
        assert len(questions) == 2
        assert questions == [
            "What is Python?",
            "How does garbage collection work?",
        ]

    def test_statement_containing_indicator_word_is_not_a_question(self):
        """Statements that merely contain 'is' or 'can' must not count as questions."""
        text = "This is a statement. The weather is nice today."
        is_multi, questions = detect_multiple_questions(text)
        assert is_multi is False
        assert questions == [text]

    def test_question_followed_by_statement(self):
        """A single question followed by a non-question statement is not multi-question."""
        text = "What is the interest rate? Please answer as quickly as possible."
        is_multi, questions = detect_multiple_questions(text)
        assert is_multi is False
        assert questions == [text]

    def test_two_questions_with_trailing_non_question_statement(self):
        text = "What is X? How is Y? Thank you very much."
        is_multi, questions = detect_multiple_questions(text)
        assert is_multi is True
        assert len(questions) == 2
        assert questions == ["What is X?", "How is Y?"]

    def test_empty_string(self):
        is_multi, questions = detect_multiple_questions("")
        assert is_multi is False
        assert questions == [""]

    def test_whitespace_string(self):
        is_multi, questions = detect_multiple_questions("   ")
        assert is_multi is False
        assert questions == ["   "]

    def test_modal_verb_questions(self):
        text = "Can we refund this order? Should I contact support?"
        is_multi, questions = detect_multiple_questions(text)
        assert is_multi is True
        assert questions == [
            "Can we refund this order?",
            "Should I contact support?",
        ]


class TestRemoveQuestionRepetition:
    """Tests for remove_question_repetition."""

    def test_removes_exact_question_echo(self):
        question = "What is the total revenue for 2023?"
        response = (
            "What is the total revenue for 2023? "
            "The total revenue for 2023 was 5 million dollars."
        )
        cleaned = remove_question_repetition(response, question)
        assert cleaned == "The total revenue for 2023 was 5 million dollars."

    def test_removes_paraphrased_echo(self):
        question = "What is the total revenue reported by Acme Corp in 2023?"
        response = (
            "The total revenue reported by Acme Corp in 2023 was as follows. "
            "Acme Corp generated 50 million dollars."
        )
        cleaned = remove_question_repetition(response, question)
        assert cleaned == "Acme Corp generated 50 million dollars."

    def test_keeps_substantive_response_when_no_echo(self):
        question = "What is the total revenue for 2023?"
        response = "The company reported 5 million dollars in total earnings."
        cleaned = remove_question_repetition(response, question)
        assert cleaned == response

    def test_single_sentence_response_is_not_truncated(self):
        question = "What is Python?"
        response = "Python is a high-level programming language."
        cleaned = remove_question_repetition(response, question)
        assert cleaned == response

    def test_empty_response(self):
        assert remove_question_repetition("", "What is this?") == ""

    def test_empty_question(self):
        response = "This is a response."
        assert remove_question_repetition(response, "") == response

    def test_stop_words_only_question(self):
        """Questions consisting only of stop words should not trigger false positive removal."""
        question = "What is it?"
        response = "It is an automated document analysis tool. It runs locally."
        cleaned = remove_question_repetition(response, question)
        assert cleaned == response
