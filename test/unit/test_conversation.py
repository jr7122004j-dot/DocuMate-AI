"""Unit tests for the LCEL conversational RAG pipeline and ChatHistory.

Tests first-turn condense skipping, follow-up condensing, context formatting,
streaming answer tokens, and chat history window enforcement.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore

from documate_ai.chat_history import ChatHistory
from documate_ai.conversation import (
    ConversationalRAG,
    ask_question,
    create_conversational_chain,
    format_docs_for_context,
    stream_answer,
)


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
    ]


@pytest.fixture
def vector_store(fake_embeddings, sample_docs) -> InMemoryVectorStore:
    store = InMemoryVectorStore(embedding=fake_embeddings)
    store.add_documents(sample_docs)
    return store


# ---------------------------------------------------------------------------
# ChatHistory Tests
# ---------------------------------------------------------------------------


class TestChatHistory:
    def test_init_defaults(self):
        history = ChatHistory(max_turns=3)
        assert history.max_turns == 3
        assert history.is_empty is True
        assert len(history) == 0

    def test_add_turn(self):
        history = ChatHistory(max_turns=5)
        history.add_turn("What is revenue?", "$50M")
        assert history.is_empty is False
        assert len(history) == 2
        msgs = history.get_messages()
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == "What is revenue?"
        assert isinstance(msgs[1], AIMessage)
        assert msgs[1].content == "$50M"

    def test_max_turns_window_enforced(self):
        """When turns exceed max_turns, only the most recent max_turns are kept."""
        history = ChatHistory(max_turns=2)

        history.add_turn("Q1", "A1")
        history.add_turn("Q2", "A2")
        assert len(history) == 4

        # Adding a 3rd turn must drop the 1st turn
        history.add_turn("Q3", "A3")
        assert len(history) == 4
        msgs = history.get_messages()
        assert [m.content for m in msgs] == ["Q2", "A2", "Q3", "A3"]

    def test_clear(self):
        history = ChatHistory(max_turns=3)
        history.add_turn("Q1", "A1")
        assert len(history) == 2
        history.clear()
        assert history.is_empty is True
        assert len(history) == 0

    def test_add_individual_messages(self):
        history = ChatHistory()
        history.add_user_message("Hello")
        history.add_ai_message("Hi there")
        assert len(history) == 2


# ---------------------------------------------------------------------------
# Context Formatting Tests
# ---------------------------------------------------------------------------


class TestFormatDocsForContext:
    def test_empty_docs(self):
        assert format_docs_for_context([]) == "No documents retrieved."

    def test_formats_with_source_and_page_headers(self, sample_docs):
        formatted = format_docs_for_context(sample_docs)
        assert "[annual_report.pdf, page 4]" in formatted
        assert "Acme Corp generated $50 million in revenue in 2023." in formatted
        assert "[annual_report.pdf, page 5]" in formatted
        assert "Net profit for Acme Corp reached $8 million in 2023." in formatted


# ---------------------------------------------------------------------------
# ConversationalRAG Tests
# ---------------------------------------------------------------------------


class TestConversationalRAG:
    def test_first_turn_skips_condensing(self, vector_store):
        """On an empty history, the question must not be sent to a condensing LLM."""
        # Only ONE LLM response is provided; if condensing ran, this would raise
        # an error because FakeListChatModel would run out of responses.
        llm = FakeListChatModel(responses=["The revenue was $50 million."])
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})

        rag = ConversationalRAG(retriever=retriever, llm=llm)
        result = rag.invoke("What was the revenue?")

        assert result["standalone_question"] == "What was the revenue?"
        assert result["answer"] == "The revenue was $50 million."
        assert len(result["source_documents"]) > 0

    def test_follow_up_turn_condenses_question(self, vector_store):
        """When history is present, the condense chain rephrases the question."""
        # Two responses: 1st for condense step, 2nd for QA step
        llm = FakeListChatModel(
            responses=[
                "What was the net profit of Acme Corp in 2023?",
                "Net profit was $8 million.",
            ]
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})

        rag = ConversationalRAG(retriever=retriever, llm=llm)
        history = ChatHistory(max_turns=3)
        history.add_turn("What was Acme Corp revenue in 2023?", "$50M")

        result = rag.invoke("And what about net profit?", history=history)

        assert result["standalone_question"] == (
            "What was the net profit of Acme Corp in 2023?"
        )
        assert result["answer"] == "Net profit was $8 million."
        assert len(result["source_documents"]) > 0

    def test_stream_answer_yields_only_qa_tokens(self, vector_store):
        """stream_answer yields tokens from the QA step and populates metadata."""
        llm = FakeListChatModel(
            responses=[
                "Standalone Question for retrieval",
                "TokenA TokenB TokenC",
            ]
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        rag = ConversationalRAG(retriever=retriever, llm=llm)

        history = ChatHistory(max_turns=3)
        history.add_turn("Q1", "A1")

        out_meta: dict = {}
        streamed_chunks = list(
            rag.stream_answer("Follow-up Q2?", history=history, out_result=out_meta)
        )

        # Streamed chunks must contain the QA tokens, NOT the condense question
        full_streamed = "".join(streamed_chunks)
        assert "TokenA TokenB TokenC" in full_streamed
        assert "Standalone Question" not in full_streamed

        assert out_meta["answer"] == "TokenA TokenB TokenC"
        assert len(out_meta["source_documents"]) > 0
        assert out_meta["standalone_question"] == "Standalone Question for retrieval"

    def test_convenience_helpers(self, vector_store):
        llm = FakeListChatModel(responses=["Grounded Answer."])
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})

        chain = create_conversational_chain(retriever, llm)
        res = ask_question(chain, "Question?")
        assert res["answer"] == "Grounded Answer."

        llm2 = FakeListChatModel(responses=["Stream Token"])
        chain2 = create_conversational_chain(retriever, llm2)
        tokens = list(stream_answer(chain2, "Question?"))
        assert "".join(tokens) == "Stream Token"
