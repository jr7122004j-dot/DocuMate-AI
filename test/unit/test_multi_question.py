"""Unit tests for multi-question query processing and history isolation (T14).

Tests:
    1. Single-question fallback (with and without streaming, history updated).
    2. Multi-question answering in order with fake LLM.
    3. History isolation: sub-questions share the same history snapshot,
       and exactly one combined turn is appended to history.
    4. Citations are extracted per sub-question and deduplicated in the combined list.
    5. Live streaming via TokenSink (headings, tokens, sub-citations streamed;
       sink is never finalized inside the function).
    6. Max sub-questions cap from Settings and per-call override.
    7. Backwards-compatible stream_handler parameter.
"""

from __future__ import annotations

import os

# Ensure telemetry is disabled before any chromadb import
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.vectorstores import InMemoryVectorStore

from documate_ai.chat_history import ChatHistory
from documate_ai.config import Settings
from documate_ai.conversation import ConversationalRAG, process_multi_question_query
from documate_ai.schemas import ChatResponse, Citation
from documate_ai.stream_handler import CollectingSink, NullSink, StreamlitTokenSink
from test.conftest import FakeContainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=384)


@pytest.fixture
def sample_docs() -> list[Document]:
    return [
        Document(
            page_content="Acme Corp generated $50 million in revenue in 2023.",
            metadata={"source": "annual_report.pdf", "page": 4},
        ),
        Document(
            page_content="Net profit for Acme Corp reached $8 million in 2023.",
            metadata={"source": "annual_report.pdf", "page": 5},
        ),
        Document(
            page_content="Acme Corp had 500 employees as of December 2023.",
            metadata={"source": "factsheet.pdf", "page": 1},
        ),
    ]


@pytest.fixture
def vector_store(
    fake_embeddings: DeterministicFakeEmbedding,
    sample_docs: list[Document],
) -> InMemoryVectorStore:
    store = InMemoryVectorStore(embedding=fake_embeddings)
    store.add_documents(sample_docs)
    return store


# ---------------------------------------------------------------------------
# Single-Question Path Tests
# ---------------------------------------------------------------------------


class TestSingleQuestionPath:
    def test_single_question_without_sink(self, vector_store: InMemoryVectorStore) -> None:
        """Single question without sink returns answer, citations, and updates history."""
        llm = FakeListChatModel(responses=["The revenue was $50 million."])
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        rag = ConversationalRAG(retriever=retriever, llm=llm)
        history = ChatHistory(max_turns=3)

        result = process_multi_question_query(
            chain=rag,
            prompt="What was the revenue of Acme Corp?",
            history=history,
        )

        assert "The revenue was $50 million." in result["answer"]
        assert len(result["source_documents"]) > 0
        assert len(result["citations"]) > 0
        assert result["sub_questions"] == ["What was the revenue of Acme Corp?"]
        assert len(result["sub_answers"]) == 1

        # History must contain exactly 1 turn (2 messages)
        assert len(history) == 2
        msgs = history.get_messages()
        assert msgs[0].content == "What was the revenue of Acme Corp?"
        assert msgs[1].content == result["answer"]

    def test_single_question_with_sink_streams_tokens(
        self,
        vector_store: InMemoryVectorStore,
    ) -> None:
        """Single question streams tokens via TokenSink and never calls finalize()."""
        llm = FakeListChatModel(responses=["Net profit was $8 million."])
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        rag = ConversationalRAG(retriever=retriever, llm=llm)
        sink = CollectingSink()

        result = process_multi_question_query(
            chain=rag,
            prompt="What was the net profit?",
            sink=sink,
        )

        # Tokens reached sink
        assert "Net profit was $8 million." in sink.text
        # finalize() was NOT called inside process_multi_question_query
        assert sink.is_finalized is False

        # Caller can finalize
        final = sink.finalize(result["answer"])
        assert sink.is_finalized is True
        assert final == result["answer"]


# ---------------------------------------------------------------------------
# Multi-Question Evaluation & Ordering Tests
# ---------------------------------------------------------------------------


class TestMultiQuestionProcessing:
    def test_two_questions_answered_in_order(
        self,
        vector_store: InMemoryVectorStore,
    ) -> None:
        """Two questions produce two answers in order under markdown headings."""
        llm = FakeListChatModel(
            responses=[
                "Acme revenue was $50 million.",
                "Net profit was $8 million.",
            ]
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        rag = ConversationalRAG(retriever=retriever, llm=llm)

        prompt = "What was the revenue? And what was the net profit?"
        result = process_multi_question_query(chain=rag, prompt=prompt)

        assert len(result["sub_questions"]) == 2
        assert "What was the revenue?" in result["sub_questions"][0]
        assert "what was the net profit?" in result["sub_questions"][1].lower()

        # Both answers present in order under headings
        answer = result["answer"]
        idx_q1 = answer.find("### What was the revenue?")
        idx_ans1 = answer.find("Acme revenue was $50 million.")
        idx_q2 = answer.find("### And what was the net profit?")
        idx_ans2 = answer.find("Net profit was $8 million.")

        assert idx_q1 != -1
        assert idx_ans1 != -1
        assert idx_q2 != -1
        assert idx_ans2 != -1
        assert idx_q1 < idx_ans1 < idx_q2 < idx_ans2

        assert len(result["sub_answers"]) == 2
        assert "Acme revenue was $50 million." in result["sub_answers"][0]["answer"]
        assert "Net profit was $8 million." in result["sub_answers"][1]["answer"]


# ---------------------------------------------------------------------------
# History Isolation & Snapshot Tests
# ---------------------------------------------------------------------------


class TestHistoryIsolation:
    def test_sub_questions_evaluated_against_same_history_snapshot(
        self,
        vector_store: InMemoryVectorStore,
    ) -> None:
        """Both sub-questions see the initial history snapshot without cross-pollution.

        If history was polluted, sub-question 2 would see sub-question 1's answer
        in history during condensation.
        """
        # When history is non-empty, ConversationalRAG calls the condense chain
        # for each sub-question before the QA chain.
        # Order of LLM calls:
        #   1. Condense sub-question 1 -> "Standalone Q1"
        #   2. QA for sub-question 1 -> "Answer 1"
        #   3. Condense sub-question 2 -> "Standalone Q2"
        #   4. QA for sub-question 2 -> "Answer 2"
        llm = FakeListChatModel(
            responses=[
                "What was the 2023 revenue of Acme Corp?",
                "Revenue was $50 million.",
                "What was the 2023 net profit of Acme Corp?",
                "Net profit was $8 million.",
            ]
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        rag = ConversationalRAG(retriever=retriever, llm=llm)

        history = ChatHistory(max_turns=5)
        history.add_turn("Who is Acme Corp?", "A manufacturing company.")

        prompt = "What was their revenue? And what was their net profit?"
        result = process_multi_question_query(
            chain=rag,
            prompt=prompt,
            history=history,
        )

        # Standalone questions recorded in sub_answers
        assert (
            result["sub_answers"][0]["standalone_question"]
            == "What was the 2023 revenue of Acme Corp?"
        )
        assert (
            result["sub_answers"][1]["standalone_question"]
            == "What was the 2023 net profit of Acme Corp?"
        )

        # History must have gained EXACTLY ONE combined turn (from 2 messages to 4)
        assert len(history) == 4
        msgs = history.get_messages()
        assert msgs[0].content == "Who is Acme Corp?"
        assert msgs[1].content == "A manufacturing company."
        assert msgs[2].content == prompt
        assert msgs[3].content == result["answer"]

    def test_empty_history_does_not_condense_either_sub_question(
        self,
        vector_store: InMemoryVectorStore,
    ) -> None:
        """On empty history, neither sub-question 1 nor sub-question 2 condenses."""
        # Only TWO responses are provided: one QA response per sub-question.
        # If sub-question 2 condensed because of sub-question 1 polluting history,
        # FakeListChatModel would run out of responses and raise IndexError.
        llm = FakeListChatModel(
            responses=[
                "Revenue is $50 million.",
                "Net profit is $8 million.",
            ]
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        rag = ConversationalRAG(retriever=retriever, llm=llm)

        history = ChatHistory(max_turns=3)
        assert history.is_empty is True

        prompt = "What is the revenue? And what is the net profit?"
        result = process_multi_question_query(
            chain=rag,
            prompt=prompt,
            history=history,
        )

        # Exactly 1 turn added to the initially empty history
        assert len(history) == 2
        assert history.get_messages()[0].content == prompt
        assert history.get_messages()[1].content == result["answer"]


# ---------------------------------------------------------------------------
# Citation Deduplication Tests
# ---------------------------------------------------------------------------


class TestCitationDeduplication:
    def test_citations_deduplicated_across_sub_questions(
        self,
        fake_embeddings: DeterministicFakeEmbedding,
    ) -> None:
        """Citations across sub-questions are deduplicated in the top-level list."""
        # Doc 1 is retrieved by both sub-questions; Doc 2 by Q1 only; Doc 3 by Q2 only
        doc1 = Document(
            page_content="Overview of revenue and profits.",
            metadata={"source": "report.pdf", "page": 4},
        )
        doc2 = Document(
            page_content="Detailed revenue breakdown.",
            metadata={"source": "report.pdf", "page": 5},
        )
        doc3 = Document(
            page_content="Employee count and headcount details.",
            metadata={"source": "factsheet.pdf", "page": 1},
        )

        store = InMemoryVectorStore(embedding=fake_embeddings)
        store.add_documents([doc1, doc2, doc3])
        retriever = store.as_retriever(search_kwargs={"k": 3})

        llm = FakeListChatModel(
            responses=[
                "Answer to Q1.",
                "Answer to Q2.",
            ]
        )
        rag = ConversationalRAG(retriever=retriever, llm=llm)

        prompt = "What is the revenue? And what is the employee count?"
        result = process_multi_question_query(chain=rag, prompt=prompt)

        # Top-level citations must have unique (source, page) entries
        citations: list[Citation] = result["citations"]
        keys = [(c.source, c.page) for c in citations]
        assert len(keys) == len(set(keys))

        # Check source documents are also deduplicated
        source_docs: list[Document] = result["source_documents"]
        doc_keys = [(d.metadata.get("source"), d.metadata.get("page")) for d in source_docs]
        assert len(doc_keys) == len(set(doc_keys))


# ---------------------------------------------------------------------------
# Streaming with TokenSink Tests
# ---------------------------------------------------------------------------


class TestStreamingWithTokenSink:
    def test_stream_headings_and_answers_live_through_collecting_sink(
        self,
        vector_store: InMemoryVectorStore,
    ) -> None:
        """Headings, answers, and sub-citations are streamed live to TokenSink."""
        llm = FakeListChatModel(
            responses=[
                "Revenue: $50 million.",
                "Profit: $8 million.",
            ]
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        rag = ConversationalRAG(retriever=retriever, llm=llm)

        sink = CollectingSink()
        prompt = "What is the revenue? And what is the net profit?"
        result = process_multi_question_query(
            chain=rag,
            prompt=prompt,
            sink=sink,
        )

        streamed = sink.text
        # Headings streamed
        assert "### What is the revenue?" in streamed
        assert "### And what is the net profit?" in streamed
        # Answers streamed
        assert "Revenue: $50 million." in streamed
        assert "Profit: $8 million." in streamed
        # Sub-citations streamed
        assert "Sources:" in streamed
        assert "annual_report.pdf, page 4" in streamed

        # sink.finalize() was NOT called internally
        assert sink.is_finalized is False

        # Caller finalizes
        final = sink.finalize(result["answer"])
        assert sink.is_finalized is True
        assert final == result["answer"]

    def test_stream_with_fake_container(
        self,
        vector_store: InMemoryVectorStore,
        fake_container: FakeContainer,
    ) -> None:
        """StreamlitTokenSink receives live updates and renders properly."""
        llm = FakeListChatModel(
            responses=[
                "Ans 1",
                "Ans 2",
            ]
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        rag = ConversationalRAG(retriever=retriever, llm=llm)

        sink = StreamlitTokenSink(container=fake_container, render_interval=0.0)
        prompt = "What is the revenue? And what is the net profit?"
        result = process_multi_question_query(
            chain=rag,
            prompt=prompt,
            sink=sink,
        )

        assert not sink.is_finalized
        # Live updates rendered in fake container
        assert len(fake_container.markdown_calls) > 0

        # Caller finalizes
        sink.finalize(result["answer"])
        assert sink.is_finalized
        assert fake_container.markdown_calls[-1] == result["answer"]

    def test_backwards_compatible_stream_handler_kwarg(
        self,
        vector_store: InMemoryVectorStore,
    ) -> None:
        """stream_handler keyword argument is mapped to sink for backwards compatibility."""
        llm = FakeListChatModel(responses=["Revenue: $50 million."])
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        rag = ConversationalRAG(retriever=retriever, llm=llm)

        sink = CollectingSink()
        result = process_multi_question_query(
            chain=rag,
            prompt="What is the revenue?",
            stream_handler=sink,
        )
        assert "Revenue: $50 million." in sink.text
        assert result["answer"]


# ---------------------------------------------------------------------------
# Max Sub-Questions Cap Tests
# ---------------------------------------------------------------------------


class TestMaxSubQuestionsCap:
    def test_max_sub_questions_caps_cost(
        self,
        vector_store: InMemoryVectorStore,
    ) -> None:
        """Compound query with 3 sub-questions is capped when max_sub_questions=2."""
        # Provide only 2 responses; if 3 questions ran, it would fail
        llm = FakeListChatModel(
            responses=[
                "Answer 1.",
                "Answer 2.",
            ]
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        rag = ConversationalRAG(retriever=retriever, llm=llm)

        prompt = "What is the revenue? What is the profit? And what is the headcount?"
        result = process_multi_question_query(
            chain=rag,
            prompt=prompt,
            max_sub_questions=2,
        )

        assert len(result["sub_questions"]) == 2
        assert len(result["sub_answers"]) == 2

    def test_settings_max_sub_questions_applied(
        self,
        vector_store: InMemoryVectorStore,
    ) -> None:
        """Settings.max_sub_questions is used when per-call parameter is None."""
        llm = FakeListChatModel(responses=["Answer 1."])
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        settings = Settings(max_sub_questions=1)
        rag = ConversationalRAG(retriever=retriever, llm=llm, settings=settings)

        prompt = "What is the revenue? And what is the net profit?"
        result = process_multi_question_query(chain=rag, prompt=prompt)

        assert len(result["sub_questions"]) == 1


# ---------------------------------------------------------------------------
# ChatResponse Schema Compatibility Tests
# ---------------------------------------------------------------------------


class TestChatResponseCompatibility:
    def test_result_can_construct_chat_response(
        self,
        vector_store: InMemoryVectorStore,
    ) -> None:
        """The result dict can directly construct a valid ChatResponse."""
        llm = FakeListChatModel(
            responses=[
                "Answer 1.",
                "Answer 2.",
            ]
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        rag = ConversationalRAG(retriever=retriever, llm=llm)

        prompt = "What is the revenue? And what is the net profit?"
        result = process_multi_question_query(chain=rag, prompt=prompt)

        chat_resp = ChatResponse(
            answer=result["answer"],
            citations=result["citations"],
            sub_questions=result["sub_questions"],
        )
        assert chat_resp.answer == result["answer"]
        assert chat_resp.citations == result["citations"]
        assert len(chat_resp.sub_questions) == 2

        resp_dict = chat_resp.to_dict()
        assert resp_dict["answer"] == result["answer"]
        assert len(resp_dict["citations"]) > 0
