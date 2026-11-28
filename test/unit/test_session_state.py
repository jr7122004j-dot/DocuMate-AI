"""Unit tests for Streamlit session state and resource caching (T16).

Tests:
    1. initialise_session_state sets default keys (engine, chat_messages,
       cache_enabled, similarity_threshold, debug_mode, health_report).
    2. initialise_session_state is idempotent and preserves user-configured values.
    3. get_engine retrieves the active engine or initialises one if absent.
    4. add_chat_message appends structured dicts and syncs chat_history.
    5. clear_chat empties UI messages and calls engine.clear_history().
    6. reset_all calls engine.reset() and restores UI defaults.
    7. ui/resources.py resource factories and create_engine.
"""

from __future__ import annotations

import os

# Ensure telemetry is disabled before any chromadb import
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import streamlit as st
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from documate_ai.cache_manager import CacheManager
from documate_ai.config import Settings
from documate_ai.engine import DocuMateEngine
from documate_ai.models import HealthReport
from documate_ai.session_state import (
    add_chat_message,
    clear_chat,
    get_engine,
    initialise_session_state,
    reset_all,
)
from documate_ai.ui.resources import (
    create_engine,
    get_cache_manager,
    get_embeddings,
    get_llm,
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
def fake_engine(tmp_path: Path) -> DocuMateEngine:
    """Create a lightweight DocuMateEngine instance for session state testing."""
    settings = Settings(data_dir=tmp_path)
    embeddings = DeterministicFakeEmbedding(size=384)
    llm = FakeListChatModel(responses=["Test response."])
    cache_mgr = CacheManager(cache_dir=tmp_path / "cache", settings=settings)
    return DocuMateEngine(
        embeddings=embeddings,
        llm=llm,
        cache_manager=cache_mgr,
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Session State Initialisation Tests
# ---------------------------------------------------------------------------


class TestSessionStateInitialisation:
    def test_initialise_session_state_defaults(self, fake_engine: DocuMateEngine) -> None:
        """initialise_session_state sets all expected default keys."""
        settings = Settings(similarity_threshold=0.88)
        initialise_session_state(settings=settings, engine=fake_engine)

        assert "engine" in st.session_state
        assert st.session_state["engine"] is fake_engine
        assert st.session_state["chat_messages"] == []
        assert st.session_state["chat_history"] == []
        assert st.session_state["cache_enabled"] is True
        assert st.session_state["similarity_threshold"] == 0.88
        assert st.session_state["debug_mode"] is False
        assert st.session_state["health_report"] is None

    def test_initialise_session_state_is_idempotent(
        self,
        fake_engine: DocuMateEngine,
    ) -> None:
        """Existing session state values are not overwritten on subsequent calls."""
        st.session_state["similarity_threshold"] = 0.95
        st.session_state["debug_mode"] = True
        st.session_state["cache_enabled"] = False
        st.session_state["chat_messages"] = [{"role": "user", "content": "Hello"}]

        initialise_session_state(engine=fake_engine)

        assert st.session_state["similarity_threshold"] == 0.95
        assert st.session_state["debug_mode"] is True
        assert st.session_state["cache_enabled"] is False
        assert len(st.session_state["chat_messages"]) == 1

    def test_get_engine_retrieves_or_initialises(
        self,
        fake_engine: DocuMateEngine,
    ) -> None:
        """get_engine returns the active engine or initialises state if absent."""
        st.session_state["engine"] = fake_engine
        retrieved = get_engine()
        assert retrieved is fake_engine

        st.session_state.clear()
        engine2 = get_engine()
        assert isinstance(engine2, DocuMateEngine)
        assert st.session_state["engine"] is engine2


# ---------------------------------------------------------------------------
# Chat Message Management Tests
# ---------------------------------------------------------------------------


class TestChatMessageManagement:
    def test_add_chat_message_structured_and_syncs_history(self) -> None:
        """add_chat_message records structured metadata and syncs chat_history tuple."""
        citations = [{"source": "report.pdf", "page": 4}]
        timings = {"cache_ms": 1.2, "generation_ms": 150.0}

        msg = add_chat_message(
            role="assistant",
            content="Revenue was $50 million.",
            citations=citations,
            from_cache=True,
            similarity=0.96,
            timings=timings,
        )

        assert msg["role"] == "assistant"
        assert msg["content"] == "Revenue was $50 million."
        assert msg["from_cache"] is True
        assert msg["similarity"] == 0.96
        assert msg["citations"] == citations
        assert msg["timings"] == timings

        # Check session state
        assert st.session_state["chat_messages"][-1] == msg
        assert st.session_state["chat_history"][-1] == ("assistant", "Revenue was $50 million.")

    def test_add_chat_message_with_debug_metadata(self) -> None:
        """add_chat_message preserves matched, standalone, and sub-questions."""
        msg = add_chat_message(
            role="assistant",
            content="Answer text",
            matched_question="Cached Q?",
            standalone_question="Condensed Q?",
            sub_questions=["Sub Q1?", "Sub Q2?"],
        )

        assert msg["matched_question"] == "Cached Q?"
        assert msg["standalone_question"] == "Condensed Q?"
        assert msg["sub_questions"] == ["Sub Q1?", "Sub Q2?"]
        assert msg["citations"] == []
        assert msg["timings"] == {}

    def test_clear_chat_empties_ui_and_engine_history(
        self,
        fake_engine: DocuMateEngine,
    ) -> None:
        """clear_chat empties UI message lists and calls engine.clear_history()."""
        initialise_session_state(engine=fake_engine)
        fake_engine.history.add_turn("User question", "Assistant answer")
        add_chat_message("user", "User question")
        add_chat_message("assistant", "Assistant answer")

        assert len(st.session_state["chat_messages"]) == 2
        assert len(st.session_state["chat_history"]) == 2
        assert len(fake_engine.history) == 2

        clear_chat()

        assert len(st.session_state["chat_messages"]) == 0
        assert len(st.session_state["chat_history"]) == 0
        assert len(fake_engine.history) == 0


# ---------------------------------------------------------------------------
# Reset All Tests
# ---------------------------------------------------------------------------


class TestResetAll:
    def test_reset_all_clears_engine_and_restores_defaults(
        self,
        fake_engine: DocuMateEngine,
    ) -> None:
        """reset_all calls engine.reset() and restores UI default values."""
        initialise_session_state(engine=fake_engine)

        mock_reset = MagicMock()
        fake_engine.reset = mock_reset  # type: ignore[method-assign]

        # Mutate UI state
        st.session_state["chat_messages"] = [{"role": "user", "content": "hi"}]
        st.session_state["chat_history"] = [("user", "hi")]
        st.session_state["cache_enabled"] = False
        st.session_state["similarity_threshold"] = 0.55
        st.session_state["debug_mode"] = True
        st.session_state["health_report"] = HealthReport(server_reachable=True)

        reset_all(delete_store=True)

        # Engine reset was called with delete_store
        mock_reset.assert_called_once_with(delete_store=True)

        # UI state restored to defaults
        assert st.session_state["chat_messages"] == []
        assert st.session_state["chat_history"] == []
        assert st.session_state["cache_enabled"] is True
        assert st.session_state["similarity_threshold"] == 0.85
        assert st.session_state["debug_mode"] is False
        assert st.session_state["health_report"] is None


# ---------------------------------------------------------------------------
# UI Resources Tests
# ---------------------------------------------------------------------------


class TestUIResources:
    def test_get_embeddings_cached(self) -> None:
        """get_embeddings returns an OllamaEmbeddings instance."""
        emb1 = get_embeddings(model_name="all-minilm:l6-v2")
        emb2 = get_embeddings(model_name="all-minilm:l6-v2")
        # Streamlit cache_resource caches the returned instance
        assert emb1 is emb2

    def test_get_llm_cached(self) -> None:
        """get_llm returns a ChatOllama instance."""
        llm1 = get_llm(model_name="phi4-mini")
        llm2 = get_llm(model_name="phi4-mini")
        assert llm1 is llm2

    def test_get_cache_manager_cached(self, tmp_path: Path) -> None:
        """get_cache_manager returns a CacheManager instance."""
        c1 = get_cache_manager(cache_dir=str(tmp_path), threshold=0.85)
        c2 = get_cache_manager(cache_dir=str(tmp_path), threshold=0.85)
        assert c1 is c2

    def test_create_engine_with_custom_components(self, tmp_path: Path) -> None:
        """create_engine wires provided components into DocuMateEngine."""
        settings = Settings(data_dir=tmp_path)
        embeddings = DeterministicFakeEmbedding(size=384)
        llm = FakeListChatModel(responses=["Test"])
        cache_mgr = CacheManager(cache_dir=tmp_path / "cache", settings=settings)

        engine = create_engine(
            settings=settings,
            embeddings=embeddings,
            llm=llm,
            cache_manager=cache_mgr,
        )

        assert isinstance(engine, DocuMateEngine)
        assert engine.embeddings is embeddings
        assert engine.llm is llm
        assert engine.cache_manager is cache_mgr
