"""Streamlit session state management for DocuMate-AI.

Maintains per-session state (DocuMateEngine, structured chat messages,
cache settings, debug toggle, and health status) decoupled from application
engine logic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import streamlit as st

from documate_ai.config import Settings, get_settings
from documate_ai.ui.resources import create_engine

if TYPE_CHECKING:
    from documate_ai.engine import DocuMateEngine
    from documate_ai.models import HealthReport

logger = logging.getLogger(__name__)


def initialise_session_state(
    settings: Settings | None = None,
    engine: DocuMateEngine | None = None,
) -> None:
    """Initialise default values for all Streamlit session state keys.

    Idempotent: will not overwrite values that already exist in session state.

    Session state keys managed:
        - ``engine``: The active per-session DocuMateEngine instance
        - ``chat_messages``: List of structured message dicts (role, content,
          citations, from_cache, similarity, timings)
        - ``chat_history``: Backwards-compatible list of (role, content) tuples
        - ``cache_enabled``: Boolean toggle for semantic caching
        - ``similarity_threshold``: Similarity threshold slider value
        - ``debug_mode``: Boolean toggle for debug metadata
        - ``health_report``: Cached Ollama server health status
    """
    if settings is None:
        settings = get_settings()

    if "engine" not in st.session_state or st.session_state["engine"] is None:
        st.session_state["engine"] = engine or create_engine(settings)

    defaults: dict[str, Any] = {
        "chat_messages": [],
        "chat_history": [],
        "cache_enabled": True,
        "similarity_threshold": settings.similarity_threshold,
        "debug_mode": False,
        "health_report": None,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def get_engine() -> DocuMateEngine:
    """Retrieve the active DocuMateEngine from session state, initializing if needed."""
    if "engine" not in st.session_state or st.session_state["engine"] is None:
        initialise_session_state()
    engine: DocuMateEngine = st.session_state["engine"]
    return engine


def add_chat_message(
    role: str,
    content: str,
    *,
    citations: list[Any] | None = None,
    from_cache: bool = False,
    similarity: float | None = None,
    timings: dict[str, float] | None = None,
    matched_question: str | None = None,
    standalone_question: str | None = None,
    sub_questions: list[str] | None = None,
) -> dict[str, Any]:
    """Add a structured message dict to chat_messages in session state.

    Args:
        role: Message author ('user' or 'assistant').
        content: Message markdown text.
        citations: List of structured Citation objects or dicts.
        from_cache: Whether the answer was returned from the cache.
        similarity: Cosine similarity score if matched from semantic cache.
        timings: Execution timing breakdown.
        matched_question: The cached question matched if from cache.
        standalone_question: Condensed standalone question if follow-up.
        sub_questions: List of split sub-questions if multi-question.

    Returns:
        The created message dictionary.
    """
    msg: dict[str, Any] = {
        "role": role,
        "content": content,
        "citations": citations or [],
        "from_cache": from_cache,
        "similarity": similarity,
        "timings": timings or {},
        "matched_question": matched_question,
        "standalone_question": standalone_question,
        "sub_questions": sub_questions or [],
    }

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    st.session_state["chat_messages"].append(msg)

    # Sync backwards-compatible chat_history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    st.session_state["chat_history"].append((role, content))

    return msg


def clear_chat() -> None:
    """Clear both UI messages and the underlying engine's chat history."""
    if "chat_messages" in st.session_state:
        st.session_state["chat_messages"] = []
    if "chat_history" in st.session_state:
        st.session_state["chat_history"] = []

    engine = st.session_state.get("engine")
    if engine is not None and hasattr(engine, "clear_history"):
        engine.clear_history()

    logger.info("Chat cleared in UI session state and DocuMateEngine.")


def reset_all(*, delete_store: bool = True) -> None:
    """Reset the application session state, clearing the engine and vector store.

    Args:
        delete_store: Whether to delete the underlying Chroma collection from disk.
    """
    engine = st.session_state.get("engine")
    if engine is not None and hasattr(engine, "reset"):
        engine.reset(delete_store=delete_store)

    settings = get_settings()
    st.session_state["chat_messages"] = []
    st.session_state["chat_history"] = []
    st.session_state["cache_enabled"] = True
    st.session_state["similarity_threshold"] = settings.similarity_threshold
    st.session_state["debug_mode"] = False
    st.session_state["health_report"] = None

    logger.info("Application session state and engine reset.")
