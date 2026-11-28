"""Unit and integration tests for the chat UI panel (T18).

Tests:
    1. Pre-ingest state: guidance banner and disabled chat input before ingestion.
    2. Sources expander: formatting of Citation objects and raw dictionaries.
    3. Debug panel: timings metrics and execution metadata rendering.
    4. Chat history: replay of user and assistant messages with badges and citations.
    5. Live QA execution: token streaming via StreamlitTokenSink and history persistence.
    6. Semantic cache hit: badge caption rendering and cache metadata.
    7. Error handling: typed exceptions (EngineNotReady, Ollama, Processing, generic)
       display friendly errors without polluting chat history.
    8. AppTest integration: end-to-end verification of question answering, cache hits,
       and error handling via Streamlit AppTest.
"""

from __future__ import annotations

import os

# Disable OpenTelemetry telemetry before importing any chromadb-dependent modules
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from documate_ai.config import Settings
from documate_ai.errors import (
    DocuMateError,
    DocumentProcessingError,
    EngineNotReadyError,
    ModelNotFoundError,
    OllamaUnavailableError,
)
from documate_ai.schemas import ChatResponse, Citation
from documate_ai.session_state import add_chat_message, initialise_session_state
from documate_ai.ui.chat import (
    render_chat,
    render_chat_message,
    render_citations,
    render_debug_expander,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_session_state():
    """Ensure clean session state before and after each test."""
    st.session_state.clear()
    yield
    st.session_state.clear()


@pytest.fixture
def fake_settings() -> Settings:
    """Settings instance for chat UI tests."""
    return Settings(
        similarity_threshold=0.85,
        stream_render_interval_seconds=0.0,
    )


@pytest.fixture
def mock_engine() -> MagicMock:
    """Lightweight mock DocuMateEngine with default ready status."""
    engine = MagicMock()
    engine.is_ready = True
    engine.ask.return_value = ChatResponse(
        answer="The capital of France is Paris.",
        citations=[
            Citation(source="geography.pdf", page=1, snippet="Paris is the capital."),
        ],
        from_cache=False,
        similarity=None,
        timings={"generation_ms": 50.0, "total_ms": 60.0},
    )
    return engine


# ---------------------------------------------------------------------------
# Citations Expander Tests
# ---------------------------------------------------------------------------


class TestRenderCitations:
    def test_render_citations_empty_returns_early(self) -> None:
        """Calling render_citations with an empty list immediately returns."""
        with patch("streamlit.expander") as mock_expander:
            render_citations([])
            mock_expander.assert_not_called()

    def test_render_assistant_historical_message_with_citations(self) -> None:
        """Historical assistant message renders its attached citations."""
        msg = {
            "role": "assistant",
            "content": "Answer with citations.",
            "citations": [Citation(source="manual.pdf", page=1, snippet="Step 1.")],
        }
        with patch("streamlit.chat_message") as mock_cm, \
             patch("documate_ai.ui.chat.render_citations") as mock_citations:
            mock_cm.return_value.__enter__.return_value = MagicMock()
            render_chat_message(msg)

            mock_citations.assert_called_once_with(msg["citations"])

    def test_render_citations_skips_invalid_item(self) -> None:
        """Non-Citation non-dict items in citations list are safely skipped."""
        citations = ["invalid_str", 12345]
        with patch("streamlit.expander") as mock_expander, patch("streamlit.markdown") as mock_md:
            mock_expander.return_value.__enter__.return_value = MagicMock()
            render_citations(citations)

            mock_expander.assert_called_once()
            mock_md.assert_not_called()

    def test_render_assistant_cached_message_without_similarity_score(self) -> None:
        """Cached message with similarity=None renders plain cached caption."""
        msg = {
            "role": "assistant",
            "content": "Cached text",
            "from_cache": True,
            "similarity": None,
        }
        with patch("streamlit.chat_message") as mock_cm, \
             patch("streamlit.caption") as mock_caption:
            mock_cm.return_value.__enter__.return_value = MagicMock()
            render_chat_message(msg)

            mock_caption.assert_called_once_with("⚡ *Cached answer*")

    def test_render_citations_with_citation_objects(self) -> None:
        """Citation dataclass objects are rendered with source, page, and snippet."""
        citations = [
            Citation(source="doc_a.pdf", page=3, snippet="Snippet from A"),
            Citation(source="doc_b.pdf", page=7, snippet="Snippet from B"),
        ]
        with patch("streamlit.expander") as mock_expander, patch("streamlit.markdown") as mock_md:
            mock_expander.return_value.__enter__.return_value = MagicMock()
            render_citations(citations)

            mock_expander.assert_called_once_with("📚 Sources", expanded=False)
            rendered_calls = [c[0][0] for c in mock_md.call_args_list]
            assert any("doc_a.pdf" in text and "Page 3" in text for text in rendered_calls)
            assert any("Snippet from A" in text for text in rendered_calls)
            assert any("doc_b.pdf" in text and "Page 7" in text for text in rendered_calls)

    def test_render_citations_with_dictionaries(self) -> None:
        """Dictionary citations (e.g. from cache) are parsed and rendered correctly."""
        citations = [
            {"source": "cached_report.pdf", "page": 12, "snippet": "Cached text"},
            {"source": "fallback.pdf", "content": "Fallback content snippet"},
        ]
        with patch("streamlit.expander") as mock_expander, patch("streamlit.markdown") as mock_md:
            mock_expander.return_value.__enter__.return_value = MagicMock()
            render_citations(citations)

            mock_expander.assert_called_once()
            rendered_calls = [c[0][0] for c in mock_md.call_args_list]
            assert any("cached_report.pdf" in text for text in rendered_calls)
            assert any("Cached text" in text for text in rendered_calls)
            assert any("fallback.pdf" in text for text in rendered_calls)
            assert any("Fallback content snippet" in text for text in rendered_calls)


# ---------------------------------------------------------------------------
# Debug Expander Tests
# ---------------------------------------------------------------------------


class TestRenderDebugExpander:
    def test_render_debug_expander_with_timings_and_metadata(self) -> None:
        """Debug expander renders timing metrics and structured JSON metadata."""
        metadata: dict[str, Any] = {
            "timings": {"cache_ms": 2.1, "retrieval_ms": 15.3, "total_ms": 120.0},
            "from_cache": True,
            "similarity": 0.9654,
            "matched_question": "What is AI?",
            "standalone_question": "What is artificial intelligence?",
            "sub_questions": ["Define AI", "History of AI"],
        }
        with patch("streamlit.expander") as mock_expander, \
             patch("streamlit.columns") as mock_columns, \
             patch("streamlit.json") as mock_json:
            mock_expander.return_value.__enter__.return_value = MagicMock()
            mock_col1, mock_col2, mock_col3 = MagicMock(), MagicMock(), MagicMock()
            mock_columns.return_value = [mock_col1, mock_col2, mock_col3]

            render_debug_expander(metadata)

            mock_expander.assert_called_once_with("🐛 Debug Information", expanded=False)
            mock_columns.assert_called_once()
            mock_json.assert_called_once()
            json_arg = mock_json.call_args[0][0]
            assert json_arg["from_cache"] is True
            assert json_arg["similarity"] == 0.9654
            assert json_arg["matched_cached_question"] == "What is AI?"
            assert json_arg["standalone_question"] == "What is artificial intelligence?"
            assert json_arg["sub_questions"] == ["Define AI", "History of AI"]


# ---------------------------------------------------------------------------
# Chat Message Rendering Tests
# ---------------------------------------------------------------------------


class TestRenderChatMessage:
    def test_render_user_message(self) -> None:
        """User message is rendered directly into st.chat_message('user')."""
        msg = {"role": "user", "content": "Hello DocuMate"}
        with patch("streamlit.chat_message") as mock_cm, patch("streamlit.markdown") as mock_md:
            mock_cm.return_value.__enter__.return_value = MagicMock()
            render_chat_message(msg)

            mock_cm.assert_called_once_with("user")
            mock_md.assert_called_once_with("Hello DocuMate")

    def test_render_assistant_fresh_message(self) -> None:
        """Fresh assistant message renders content without cache badge."""
        msg = {
            "role": "assistant",
            "content": "Fresh response",
            "from_cache": False,
            "citations": [],
        }
        with patch("streamlit.chat_message") as mock_cm, \
             patch("streamlit.markdown") as mock_md, \
             patch("streamlit.caption") as mock_caption:
            mock_cm.return_value.__enter__.return_value = MagicMock()
            render_chat_message(msg)

            mock_cm.assert_called_once_with("assistant")
            mock_md.assert_called_once_with("Fresh response")
            mock_caption.assert_not_called()

    def test_render_assistant_cached_message_shows_badge(self) -> None:
        """Cached assistant message displays the similarity badge."""
        msg = {
            "role": "assistant",
            "content": "Cached response",
            "from_cache": True,
            "similarity": 0.976,
            "citations": [],
        }
        with patch("streamlit.chat_message") as mock_cm, \
             patch("streamlit.markdown") as mock_md, \
             patch("streamlit.caption") as mock_caption:
            mock_cm.return_value.__enter__.return_value = MagicMock()
            render_chat_message(msg)

            mock_caption.assert_called_once()
            caption_text = mock_caption.call_args[0][0]
            assert "Cached answer" in caption_text
            assert "0.98" in caption_text

    def test_render_assistant_message_with_debug_mode_enabled(self) -> None:
        """When debug_mode=True, debug expander is rendered."""
        msg = {
            "role": "assistant",
            "content": "Answer with debug info",
            "timings": {"total_ms": 80.0},
        }
        with patch("streamlit.chat_message") as mock_cm, \
             patch("documate_ai.ui.chat.render_debug_expander") as mock_debug:
            mock_cm.return_value.__enter__.return_value = MagicMock()
            render_chat_message(msg, debug_mode=True)

            mock_debug.assert_called_once_with(msg)

    def test_render_assistant_message_with_debug_mode_disabled(self) -> None:
        """When debug_mode=False, debug expander is omitted."""
        msg = {
            "role": "assistant",
            "content": "Answer without debug",
            "timings": {"total_ms": 80.0},
        }
        with patch("streamlit.chat_message") as mock_cm, \
             patch("documate_ai.ui.chat.render_debug_expander") as mock_debug:
            mock_cm.return_value.__enter__.return_value = MagicMock()
            render_chat_message(msg, debug_mode=False)

            mock_debug.assert_not_called()


# ---------------------------------------------------------------------------
# Main Chat Panel Tests
# ---------------------------------------------------------------------------


class TestRenderChat:
    def test_pre_ingest_state_shows_info_banner(
        self,
        mock_engine: MagicMock,
        fake_settings: Settings,
    ) -> None:
        """When engine is not ready, guidance banner is displayed and input disabled."""
        mock_engine.is_ready = False

        with patch("streamlit.info") as mock_info, \
             patch("streamlit.chat_input") as mock_input, \
             patch("streamlit.title") as mock_title:
            mock_input.return_value = None
            render_chat(engine=mock_engine, settings=fake_settings)

            mock_title.assert_called_once()
            mock_info.assert_called_once()
            assert "Please upload and process PDF documents" in mock_info.call_args[0][0]
            mock_input.assert_called_once_with(
                "Ask a question about your documents...",
                disabled=True,
            )

    def test_ready_state_hides_info_banner_and_enables_input(
        self,
        mock_engine: MagicMock,
        fake_settings: Settings,
    ) -> None:
        """When engine is ready, guidance banner is hidden and input enabled."""
        mock_engine.is_ready = True

        with patch("streamlit.info") as mock_info, \
             patch("streamlit.chat_input") as mock_input:
            mock_input.return_value = None
            render_chat(engine=mock_engine, settings=fake_settings)

            mock_info.assert_not_called()
            mock_input.assert_called_once_with(
                "Ask a question about your documents...",
                disabled=False,
            )

    def test_renders_existing_history(
        self,
        mock_engine: MagicMock,
        fake_settings: Settings,
    ) -> None:
        """Existing messages in session_state are rendered upon panel load."""
        mock_engine.is_ready = True
        add_chat_message("user", "What is the policy?")
        add_chat_message("assistant", "The policy is standard.")

        with patch("documate_ai.ui.chat.render_chat_message") as mock_render_msg, \
             patch("streamlit.chat_input") as mock_input:
            mock_input.return_value = None
            render_chat(engine=mock_engine, settings=fake_settings)

            assert mock_render_msg.call_count == 2
            assert mock_render_msg.call_args_list[0][0][0]["content"] == "What is the policy?"
            assert mock_render_msg.call_args_list[1][0][0]["content"] == "The policy is standard."

    def test_user_input_triggers_engine_ask_and_persists_messages(
        self,
        mock_engine: MagicMock,
        fake_settings: Settings,
    ) -> None:
        """Submitting a question streams tokens and appends both turns to state."""
        mock_engine.is_ready = True
        mock_engine.ask.return_value = ChatResponse(
            answer="Quantum computing uses qubits.",
            citations=[
                Citation(source="physics.pdf", page=5, snippet="Qubits exhibit superposition."),
            ],
            from_cache=False,
            similarity=None,
            timings={"generation_ms": 110.0},
        )

        with patch("streamlit.chat_input", return_value="Explain qubits"), \
             patch("streamlit.chat_message") as mock_cm, \
             patch("streamlit.empty") as mock_empty:
            mock_container = MagicMock()
            mock_empty.return_value = mock_container
            mock_cm.return_value.__enter__.return_value = MagicMock()

            render_chat(engine=mock_engine, settings=fake_settings)

            mock_engine.ask.assert_called_once()
            call_kwargs = mock_engine.ask.call_args[1]
            assert mock_engine.ask.call_args[0][0] == "Explain qubits"
            assert call_kwargs["use_cache"] is True
            assert call_kwargs["threshold"] == fake_settings.similarity_threshold

            # Session state updated
            messages = st.session_state["chat_messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "Explain qubits"
            assert messages[1]["role"] == "assistant"
            assert messages[1]["content"] == "Quantum computing uses qubits."
            assert len(messages[1]["citations"]) == 1

            # Backwards-compatible tuple history synced
            history = st.session_state["chat_history"]
            assert len(history) == 2
            assert history[0] == ("user", "Explain qubits")
            assert history[1] == ("assistant", "Quantum computing uses qubits.")

    def test_render_chat_defaults_to_session_state_engine(
        self,
        mock_engine: MagicMock,
    ) -> None:
        """render_chat uses session_state engine when engine parameter is None."""
        st.session_state["engine"] = mock_engine
        mock_engine.is_ready = True

        with patch("streamlit.chat_input", return_value=None):
            render_chat()

            assert mock_engine.is_ready is True

    def test_live_query_with_debug_mode_renders_debug_expander(
        self,
        mock_engine: MagicMock,
        fake_settings: Settings,
    ) -> None:
        """Live QA query renders debug expander when debug_mode is active."""
        st.session_state["debug_mode"] = True
        mock_engine.is_ready = True
        mock_engine.ask.return_value = ChatResponse(
            answer="Debugged answer.",
            citations=[],
            timings={"generation_ms": 100.0},
        )

        with patch("streamlit.chat_input", return_value="Show debug info"), \
             patch("streamlit.chat_message") as mock_cm, \
             patch("documate_ai.ui.chat.render_debug_expander") as mock_render_debug:
            mock_cm.return_value.__enter__.return_value = MagicMock()
            render_chat(engine=mock_engine, settings=fake_settings)

            mock_render_debug.assert_called_once()
            call_meta = mock_render_debug.call_args[0][0]
            assert call_meta["answer"] == "Debugged answer."

    def test_live_query_cached_without_similarity(
        self,
        mock_engine: MagicMock,
        fake_settings: Settings,
    ) -> None:
        """Live cache hit with similarity=None renders plain cached answer caption."""
        mock_engine.is_ready = True
        mock_engine.ask.return_value = ChatResponse(
            answer="Cached text without sim score.",
            citations=[],
            from_cache=True,
            similarity=None,
        )

        with patch("streamlit.chat_input", return_value="Question?"), \
             patch("streamlit.chat_message") as mock_cm, \
             patch("streamlit.caption") as mock_caption:
            mock_cm.return_value.__enter__.return_value = MagicMock()
            render_chat(engine=mock_engine, settings=fake_settings)

            mock_caption.assert_called_once_with("⚡ *Cached answer*")

    def test_guard_against_input_when_engine_not_ready(
        self,
        mock_engine: MagicMock,
        fake_settings: Settings,
    ) -> None:
        """If prompt is submitted when not ready, warning is shown and ask is not called."""
        mock_engine.is_ready = False

        with patch("streamlit.chat_input", return_value="Early question"), \
             patch("streamlit.warning") as mock_warning:
            render_chat(engine=mock_engine, settings=fake_settings)

            mock_warning.assert_called_once()
            assert "Please upload and process PDF documents first" in mock_warning.call_args[0][0]
            mock_engine.ask.assert_not_called()
            assert len(st.session_state.get("chat_messages", [])) == 0


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


class TestChatErrorHandling:
    @pytest.mark.parametrize(
        ("exc", "expected_err_substring"),
        [
            (
                EngineNotReadyError("Engine not ready."),
                "Engine is not ready",
            ),
            (
                OllamaUnavailableError("Ollama is not running at http://localhost:11434"),
                "Ollama server error",
            ),
            (
                ModelNotFoundError("Model phi4-mini not found"),
                "Ollama server error",
            ),
            (
                DocumentProcessingError("Corrupted PDF data"),
                "Document processing error",
            ),
            (
                DocuMateError("Internal pipeline fault"),
                "Application error",
            ),
            (
                RuntimeError("Unexpected connection reset"),
                "An unexpected error occurred",
            ),
        ],
    )
    def test_typed_exceptions_display_error_and_protect_history(
        self,
        mock_engine: MagicMock,
        fake_settings: Settings,
        exc: Exception,
        expected_err_substring: str,
    ) -> None:
        """Typed exceptions during QA show friendly st.error and do not corrupt history."""
        mock_engine.is_ready = True
        mock_engine.ask.side_effect = exc

        with patch("streamlit.chat_input", return_value="What caused the error?"), \
             patch("streamlit.chat_message") as mock_cm, \
             patch("streamlit.error") as mock_error, \
             patch("streamlit.empty") as mock_empty:
            mock_container = MagicMock()
            mock_empty.return_value = mock_container
            mock_cm.return_value.__enter__.return_value = MagicMock()

            render_chat(engine=mock_engine, settings=fake_settings)

            mock_error.assert_called_once()
            assert expected_err_substring in mock_error.call_args[0][0]

            # Container was emptied so partial streamed tokens are wiped
            mock_container.empty.assert_called_once()

            # User message was recorded, but assistant message was NOT recorded
            messages = st.session_state["chat_messages"]
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            assert not any(m["role"] == "assistant" for m in messages)


# ---------------------------------------------------------------------------
# Streamlit AppTest Integration Tests
# ---------------------------------------------------------------------------


class TestChatAppTest:
    def test_apptest_pre_ingest_state(self) -> None:
        """AppTest renders guidance banner and disabled chat input before ingest."""
        def app_func():
            from unittest.mock import MagicMock
            from documate_ai.ui.chat import render_chat
            mock_eng = MagicMock()
            mock_eng.is_ready = False
            render_chat(engine=mock_eng)

        at = AppTest.from_function(app_func)
        at.run()

        assert len(at.info) == 1
        assert "Please upload and process PDF documents" in at.info[0].value
        assert len(at.chat_input) == 1
        assert at.chat_input[0].disabled is True

    def test_apptest_question_answering_with_sources(self) -> None:
        """AppTest renders answer, user message, and Sources expander upon question."""
        def app_func():
            from unittest.mock import MagicMock
            from documate_ai.schemas import ChatResponse, Citation
            from documate_ai.ui.chat import render_chat

            mock_eng = MagicMock()
            mock_eng.is_ready = True
            mock_eng.ask.return_value = ChatResponse(
                answer="LangChain orchestrates LLM chains.",
                citations=[
                    Citation(
                        source="langchain.pdf",
                        page=2,
                        snippet="Chains link components together.",
                    ),
                ],
                from_cache=False,
            )
            render_chat(engine=mock_eng)

        at = AppTest.from_function(app_func)
        at.run()
        assert at.chat_input[0].disabled is False

        at.chat_input[0].set_value("What is LangChain?").run()

        assert len(at.chat_message) == 2
        assert at.chat_message[0].name == "user"
        assert at.chat_message[1].name == "assistant"
        assert len(at.expander) >= 1
        assert "Sources" in at.expander[0].label

    def test_apptest_cache_hit_shows_badge(self) -> None:
        """AppTest displays cached answer caption on semantic cache hit."""
        def app_func():
            from unittest.mock import MagicMock
            from documate_ai.schemas import ChatResponse
            from documate_ai.ui.chat import render_chat

            mock_eng = MagicMock()
            mock_eng.is_ready = True
            mock_eng.ask.return_value = ChatResponse(
                answer="Cached answer about retrieval.",
                citations=[],
                from_cache=True,
                similarity=0.98,
            )
            render_chat(engine=mock_eng)

        at = AppTest.from_function(app_func)
        at.run()

        at.chat_input[0].set_value("Repeat question?").run()

        assert len(at.caption) >= 1
        assert any("Cached answer" in c.value and "0.98" in c.value for c in at.caption)

    def test_apptest_error_path_shows_st_error(self) -> None:
        """AppTest displays friendly st.error when QA engine encounters an error."""
        def app_func():
            from unittest.mock import MagicMock
            from documate_ai.errors import OllamaUnavailableError
            from documate_ai.ui.chat import render_chat

            mock_eng = MagicMock()
            mock_eng.is_ready = True
            mock_eng.ask.side_effect = OllamaUnavailableError("Ollama daemon is stopped.")
            render_chat(engine=mock_eng)

        at = AppTest.from_function(app_func)
        at.run()

        at.chat_input[0].set_value("Will this fail?").run()

        assert len(at.error) >= 1
        assert any("Ollama server error" in err.value for err in at.error)


# ---------------------------------------------------------------------------
# App Main Entry Point Test
# ---------------------------------------------------------------------------


class TestAppMain:
    def test_app_main_coordinates_sidebar_and_chat(self) -> None:
        """app.main configures logging, session state, and invokes components."""
        from documate_ai.app import main

        mock_eng = MagicMock()
        with patch("documate_ai.app.configure_logging") as mock_log, \
             patch("documate_ai.app.initialise_session_state") as mock_init, \
             patch("documate_ai.app.get_engine", return_value=mock_eng), \
             patch("documate_ai.app.render_sidebar") as mock_sidebar, \
             patch("documate_ai.app.render_chat") as mock_chat:
            main()

            mock_log.assert_called_once()
            mock_init.assert_called_once()
            mock_sidebar.assert_called_once()
            mock_chat.assert_called_once()
