"""Unit tests for token streaming sinks and handlers.

Tests TokenSink protocol implementations, throttling, cursor management,
idempotent finalization, replacement of streamed text with formatted content,
and integration with conversation.stream_answer.
"""

from __future__ import annotations

import time

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.vectorstores import InMemoryVectorStore

from documate_ai.chat_history import ChatHistory
from documate_ai.conversation import ConversationalRAG
from documate_ai.stream_handler import (
    CollectingSink,
    NullSink,
    StreamHandler,
    StreamlitTokenSink,
    TokenSink,
)
from test.conftest import FakeContainer


# ---------------------------------------------------------------------------
# Protocol Conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_sinks_implement_token_sink_protocol(self, fake_container: FakeContainer):
        null_sink = NullSink()
        collecting_sink = CollectingSink()
        streamlit_sink = StreamlitTokenSink(container=fake_container)
        handler = StreamHandler(container=fake_container)

        assert isinstance(null_sink, TokenSink)
        assert isinstance(collecting_sink, TokenSink)
        assert isinstance(streamlit_sink, TokenSink)
        assert isinstance(handler, TokenSink)


# ---------------------------------------------------------------------------
# NullSink Tests
# ---------------------------------------------------------------------------


class TestNullSink:
    def test_null_sink_discards_tokens(self):
        sink = NullSink()
        sink.on_token("hello")
        sink.on_token("world")
        assert sink.finalize() == ""

    def test_null_sink_with_final_text(self):
        sink = NullSink()
        sink.on_token("discards")
        assert sink.finalize("final content") == "final content"


# ---------------------------------------------------------------------------
# CollectingSink Tests
# ---------------------------------------------------------------------------


class TestCollectingSink:
    def test_collects_tokens_in_memory(self):
        sink = CollectingSink()
        sink.on_token("Hello")
        sink.on_token(" ")
        sink.on_token("world!")

        assert sink.tokens == ["Hello", " ", "world!"]
        assert sink.text == "Hello world!"

        final = sink.finalize()
        assert final == "Hello world!"
        assert sink.text == "Hello world!"

    def test_finalize_with_replacement_text(self):
        sink = CollectingSink()
        sink.on_token("Streamed text")

        final = sink.finalize("Streamed text\n\nSources:\n1. doc.pdf, page 1")
        assert final == "Streamed text\n\nSources:\n1. doc.pdf, page 1"
        assert sink.text == "Streamed text\n\nSources:\n1. doc.pdf, page 1"

    def test_idempotent_finalize(self):
        sink = CollectingSink()
        sink.on_token("Content")

        res1 = sink.finalize("First")
        res2 = sink.finalize("Second")  # Subsequent call must not overwrite

        assert res1 == "First"
        assert res2 == "First"
        assert sink.text == "First"

    def test_tokens_after_finalize_ignored(self):
        sink = CollectingSink()
        sink.on_token("Before")
        sink.finalize()
        sink.on_token("After")
        assert sink.text == "Before"


# ---------------------------------------------------------------------------
# StreamlitTokenSink Tests
# ---------------------------------------------------------------------------


class TestStreamlitTokenSink:
    def test_renders_cursor_during_streaming(self, fake_container: FakeContainer):
        sink = StreamlitTokenSink(container=fake_container, render_interval=0.0)

        sink.on_token("Hello")
        assert fake_container.last_content == "Hello▌"

        sink.on_token(" world")
        assert fake_container.last_content == "Hello world▌"

    def test_throttling_avoids_excessive_renders(self, fake_container: FakeContainer):
        # Set a 1-second render interval
        sink = StreamlitTokenSink(container=fake_container, render_interval=1.0)

        sink.on_token("Token1")
        initial_render_count = len(fake_container.markdown_calls)
        assert initial_render_count == 1

        # Rapid succession within interval should not trigger additional renders
        sink.on_token("Token2")
        sink.on_token("Token3")
        assert len(fake_container.markdown_calls) == initial_render_count

        # Finalize should immediately render the complete text without cursor
        final = sink.finalize()
        assert final == "Token1Token2Token3"
        assert fake_container.last_content == "Token1Token2Token3"
        assert len(fake_container.markdown_calls) == initial_render_count + 1

    def test_cursor_removed_after_finalize(self, fake_container: FakeContainer):
        sink = StreamlitTokenSink(container=fake_container, render_interval=0.0)
        sink.on_token("Generating")
        assert "▌" in str(fake_container.last_content)

        sink.finalize()
        assert fake_container.last_content == "Generating"
        assert "▌" not in str(fake_container.last_content)

    def test_finalize_replaces_streamed_text_with_formatted_content(
        self, fake_container: FakeContainer
    ):
        sink = StreamlitTokenSink(container=fake_container, render_interval=0.0)
        sink.on_token("Raw answer text")

        post_processed = "Cleaned answer text\n\nSources:\n1. doc.pdf, page 2"
        final = sink.finalize(post_processed)

        assert final == post_processed
        assert fake_container.last_content == post_processed
        assert sink.text == post_processed

    def test_finalize_is_idempotent(self, fake_container: FakeContainer):
        sink = StreamlitTokenSink(container=fake_container, render_interval=0.0)
        sink.on_token("Text")

        sink.finalize("Final text")
        render_count = len(fake_container.markdown_calls)

        # Second finalize call must return the result without re-rendering
        res2 = sink.finalize("Different text")
        assert res2 == "Final text"
        assert fake_container.last_content == "Final text"
        assert len(fake_container.markdown_calls) == render_count

    def test_tokens_after_finalize_are_ignored(self, fake_container: FakeContainer):
        sink = StreamlitTokenSink(container=fake_container, render_interval=0.0)
        sink.on_token("Start")
        sink.finalize("Finished")

        render_count = len(fake_container.markdown_calls)
        sink.on_token("Late token")

        assert len(fake_container.markdown_calls) == render_count
        assert sink.text == "Finished"


# ---------------------------------------------------------------------------
# StreamHandler Backwards-Compatibility Tests
# ---------------------------------------------------------------------------


class TestStreamHandlerCompatibility:
    def test_on_llm_new_token_string(self, fake_container: FakeContainer):
        handler = StreamHandler(container=fake_container, render_interval=0.0)
        handler.on_llm_new_token("Chunk 1")
        handler.on_llm_new_token(" Chunk 2")
        final = handler.finalize()

        assert final == "Chunk 1 Chunk 2"
        assert fake_container.last_content == "Chunk 1 Chunk 2"

    def test_on_llm_new_token_list(self, fake_container: FakeContainer):
        handler = StreamHandler(container=fake_container, render_interval=0.0)
        handler.on_llm_new_token(["Part A", " Part B"])
        handler.on_llm_new_token([{"text": " Part C"}])
        final = handler.finalize()

        assert final == "Part A Part B Part C"


# ---------------------------------------------------------------------------
# Integration with conversation.stream_answer
# ---------------------------------------------------------------------------


class TestConversationStreamAnswerWithSink:
    def test_only_answer_tokens_reach_sink(self, fake_embeddings, fake_container):
        docs = [
            Document(
                page_content="Content of document.",
                metadata={"source": "report.pdf", "page": 1},
            )
        ]
        store = InMemoryVectorStore(embedding=fake_embeddings)
        store.add_documents(docs)
        retriever = store.as_retriever(search_kwargs={"k": 1})

        # 1st response: condense step (should NOT reach the sink)
        # 2nd response: QA tokens (MUST reach the sink)
        llm = FakeListChatModel(
            responses=[
                "Standalone Condensed Question",
                "AnswerToken1 AnswerToken2",
            ]
        )
        rag = ConversationalRAG(retriever=retriever, llm=llm)

        history = ChatHistory(max_turns=3)
        history.add_turn("Q1", "A1")

        sink = StreamlitTokenSink(container=fake_container, render_interval=0.0)
        out_meta: dict = {}

        tokens = list(
            rag.stream_answer(
                "Follow up question?",
                history=history,
                out_result=out_meta,
                sink=sink,
            )
        )

        assert "".join(tokens) == "AnswerToken1 AnswerToken2"
        # The sink's accumulated text must only be the answer, never the condense question!
        assert "Standalone Condensed Question" not in sink.text
        assert sink.text == "AnswerToken1 AnswerToken2"

        # Finalize with citations replaces the content in the container
        final_with_sources = (
            f"{sink.text}\n\nSources:\n1. report.pdf, page 1"
        )
        sink.finalize(final_with_sources)
        assert fake_container.last_content == final_with_sources
