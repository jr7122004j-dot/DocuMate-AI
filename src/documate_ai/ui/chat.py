"""Chat interface UI component for DocuMate-AI.

Renders the main conversation panel, past message history with collapsible
source citations and cache indicators, execution debug metadata, and handles
streaming user questions via DocuMateEngine.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import streamlit as st

from documate_ai.config import Settings, get_settings
from documate_ai.errors import (
    DocuMateError,
    DocumentProcessingError,
    EngineNotReadyError,
    ModelNotFoundError,
    OllamaUnavailableError,
)
from documate_ai.session_state import add_chat_message, get_engine
from documate_ai.stream_handler import StreamlitTokenSink

if TYPE_CHECKING:
    from documate_ai.engine import DocuMateEngine

logger = logging.getLogger(__name__)


def render_citations(citations: list[Any]) -> None:
    """Render structured source citations in a collapsible expander.

    Args:
        citations: List of Citation dataclasses or dictionaries containing
            source, page, and snippet keys.
    """
    if not citations:
        return

    with st.expander("📚 Sources", expanded=False):
        for idx, citation in enumerate(citations, start=1):
            if hasattr(citation, "source"):
                source = citation.source
                page = citation.page
                snippet = citation.snippet
            elif isinstance(citation, dict):
                source = citation.get("source", "Unknown Document")
                page = citation.get("page", 1)
                snippet = citation.get("snippet", citation.get("content", ""))
            else:
                continue

            st.markdown(f"**Source {idx}:** `{source}` (Page {page})")
            if snippet:
                st.markdown(f"> {snippet}")


def render_debug_expander(metadata: dict[str, Any]) -> None:
    """Render execution metadata and timing breakdown in a collapsible expander.

    Args:
        metadata: Dictionary containing timings, similarity, matched question,
            standalone question, or sub-questions.
    """
    with st.expander("🐛 Debug Information", expanded=False):
        timings = metadata.get("timings", {})
        if timings:
            st.markdown("**⏱️ Timing Breakdown (ms):**")
            cols = st.columns(min(len(timings), 4))
            for col, (metric_name, val) in zip(cols, timings.items()):
                col.metric(label=metric_name, value=f"{val:.1f}")

        details: dict[str, Any] = {}
        if metadata.get("from_cache"):
            details["from_cache"] = True
            if (sim := metadata.get("similarity")) is not None:
                details["similarity"] = round(sim, 4)
            if matched_q := metadata.get("matched_question"):
                details["matched_cached_question"] = matched_q

        if standalone_q := metadata.get("standalone_question"):
            details["standalone_question"] = standalone_q

        if sub_qs := metadata.get("sub_questions"):
            details["sub_questions"] = sub_qs

        if details:
            st.json(details)


def render_chat_message(msg: dict[str, Any], *, debug_mode: bool = False) -> None:
    """Render a single historical or generated chat message.

    Args:
        msg: Message dictionary containing role, content, citations,
            from_cache, similarity, and timings.
        debug_mode: Whether to display execution metadata for assistant turns.
    """
    role = msg.get("role", "user")
    content = msg.get("content", "")

    with st.chat_message(role):
        st.markdown(content)

        if role == "assistant":
            if msg.get("from_cache"):
                sim = msg.get("similarity")
                if sim is not None:
                    st.caption(f"⚡ *Cached answer (similarity: {sim:.2f})*")
                else:
                    st.caption("⚡ *Cached answer*")

            if citations := msg.get("citations"):
                render_citations(citations)

            if debug_mode:
                render_debug_expander(msg)


def render_chat(
    engine: DocuMateEngine | None = None,
    settings: Settings | None = None,
) -> None:
    """Render the full chat interface: title, history, guidance, and input.

    Args:
        engine: The active DocuMateEngine service instance. Defaults to the
            session state engine.
        settings: Application settings. Defaults to current cached settings.
    """
    if settings is None:
        settings = get_settings()
    if engine is None:
        engine = get_engine()

    st.title("💬 Chat with your Documents")

    # 1. Guidance banner when no documents are ingested yet
    if not engine.is_ready:
        st.info("👈 Please upload and process PDF documents in the sidebar to begin chatting.")

    # 2. Render existing message history
    debug_mode = bool(st.session_state.get("debug_mode", False))
    for msg in st.session_state.get("chat_messages", []):
        render_chat_message(msg, debug_mode=debug_mode)

    # 3. User chat input (disabled before document ingestion)
    prompt = st.chat_input(
        "Ask a question about your documents...",
        disabled=not engine.is_ready,
    )

    if not prompt:
        return

    # Guard against queries before ingestion if input was somehow submitted
    if not engine.is_ready:
        st.warning("⚠️ Please upload and process PDF documents first.")
        return

    # Render and store user turn
    with st.chat_message("user"):
        st.markdown(prompt)
    add_chat_message("user", prompt)

    # Execute QA query and stream assistant turn
    with st.chat_message("assistant"):
        placeholder = st.empty()
        sink = StreamlitTokenSink(
            container=placeholder,
            render_interval=settings.stream_render_interval_seconds,
        )

        cache_enabled = bool(st.session_state.get("cache_enabled", True))
        similarity_threshold = float(
            st.session_state.get("similarity_threshold", settings.similarity_threshold)
        )

        try:
            response = engine.ask(
                prompt,
                sink=sink,
                use_cache=cache_enabled,
                threshold=similarity_threshold,
            )
        except EngineNotReadyError as exc:
            placeholder.empty()
            logger.warning("Engine not ready for query: %s", exc)
            st.error("Engine is not ready. Please upload and process documents first.")
            return
        except (OllamaUnavailableError, ModelNotFoundError) as exc:
            placeholder.empty()
            logger.error("Ollama service error during question answering: %s", exc)
            st.error(f"Ollama server error: {exc}")
            return
        except DocumentProcessingError as exc:
            placeholder.empty()
            logger.error("Document processing error during question answering: %s", exc)
            st.error(f"Document processing error: {exc}")
            return
        except DocuMateError as exc:
            placeholder.empty()
            logger.error("DocuMate application error during question answering: %s", exc)
            st.error(f"Application error: {exc}")
            return
        except Exception as exc:
            placeholder.empty()
            logger.exception("Unexpected error during question answering: %s", exc)
            st.error(f"An unexpected error occurred: {exc}")
            return

        # Render cache badge if from cache
        if response.from_cache:
            if response.similarity is not None:
                st.caption(f"⚡ *Cached answer (similarity: {response.similarity:.2f})*")
            else:
                st.caption("⚡ *Cached answer*")

        # Render citations expander
        if response.citations:
            render_citations(response.citations)

        # Render debug expander if debug mode is active
        if debug_mode:
            render_debug_expander(response.to_dict())

        # Persist structured assistant message to session state
        add_chat_message(
            "assistant",
            response.answer,
            citations=response.citations,
            from_cache=response.from_cache,
            similarity=response.similarity,
            timings=response.timings,
            matched_question=response.matched_question,
            standalone_question=response.standalone_question,
            sub_questions=response.sub_questions,
        )
