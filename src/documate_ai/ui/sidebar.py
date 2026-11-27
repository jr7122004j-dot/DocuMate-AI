"""Sidebar UI component for DocuMate-AI.

Provides document uploading, Ollama service health monitoring, semantic cache
controls, loaded document statistics, and action buttons (Clear Chat, Reset All).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import streamlit as st

from documate_ai.config import Settings, get_settings
from documate_ai.errors import DocumentProcessingError
from documate_ai.models import HealthReport, check_ollama_health
from documate_ai.schemas import IngestReport, UploadedPdf
from documate_ai.session_state import clear_chat, get_engine, reset_all

if TYPE_CHECKING:
    from documate_ai.engine import DocuMateEngine

logger = logging.getLogger(__name__)


def render_health_banner(settings: Settings) -> HealthReport:
    """Render the Ollama server and model health banner.

    Displays an actionable error banner with ``ollama pull <model>`` commands
    when required models are missing or when the server is unreachable.

    Args:
        settings: Application settings.

    Returns:
        The current HealthReport object.
    """
    report: HealthReport | None = st.session_state.get("health_report")

    # Fetch report if not yet cached in session state
    if report is None:
        report = check_ollama_health(settings)
        st.session_state["health_report"] = report

    if not report.server_reachable:
        st.error(
            f"🔴 **Ollama Unavailable**\n\n"
            f"Cannot reach server at `{settings.ollama_base_url}`.\n\n"
            "Ensure Ollama is running (`ollama serve`)."
        )
    elif report.missing_models:
        st.error("⚠️ **Missing Model(s)**\n\nPull the required model(s) to continue:")
        for missing in report.missing_models:
            st.code(f"ollama pull {missing}", language="bash")
    else:
        st.success("🟢 **Ollama Ready**")

    # Re-check button
    if st.button("🔄 Re-check Health", width="stretch", help="Re-scan Ollama server"):
        st.session_state["health_report"] = check_ollama_health(settings)
        st.rerun()

    return report


def render_document_uploader(
    engine: DocuMateEngine,
    health: HealthReport,
) -> None:
    """Render the PDF uploader and document ingestion trigger.

    Args:
        engine: Active DocuMateEngine instance.
        health: Current Ollama server HealthReport.
    """
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Select PDF files to upload",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to analyse.",
    )

    can_process = (
        bool(uploaded_files)
        and health.server_reachable
        and not bool(health.missing_models)
    )

    help_msg = (
        "Process uploaded PDFs into the vector store."
        if can_process
        else "Upload PDFs and ensure Ollama models are ready to enable processing."
    )

    process_clicked = st.button(
        "⚙️ Process PDFs",
        width="stretch",
        disabled=not can_process,
        help=help_msg,
    )

    if process_clicked and uploaded_files:
        with st.status("Processing PDF documents...", expanded=True) as status:
            file_objs: list[UploadedPdf] = []
            for item in uploaded_files:
                name = getattr(item, "name", "document.pdf")
                content = item.getvalue() if hasattr(item, "getvalue") else b""
                file_objs.append(UploadedPdf(name=name, data=content))

            st.write(f"Extracting and chunking {len(file_objs)} PDF file(s)...")

            try:
                report: IngestReport = engine.ingest(file_objs, reset_history=True)
                st.session_state["last_ingest_report"] = report
                clear_chat()

                status.update(
                    label=f"Processed {len(report.files)} PDF(s) in {report.elapsed_seconds}s!",
                    state="complete",
                    expanded=False,
                )
                st.success(
                    f"Successfully ingested {report.pages} pages "
                    f"({report.chunks} chunks) across {len(report.files)} file(s)."
                )

                if report.skipped:
                    for filename, reason in report.skipped.items():
                        st.warning(f"Skipped `{filename}`: {reason}")

            except DocumentProcessingError as exc:
                status.update(label="Document processing failed!", state="error")
                st.error(str(exc))
            except Exception as exc:
                status.update(label="Processing failed unexpectedly!", state="error")
                st.error(f"Error processing files: {exc}")


def render_loaded_documents(engine: DocuMateEngine) -> None:
    """Render details of currently loaded documents.

    Args:
        engine: Active DocuMateEngine instance.
    """
    if not engine.is_ready:
        return

    st.subheader("📚 Loaded Documents")
    last_report: IngestReport | None = st.session_state.get("last_ingest_report")

    if last_report is not None:
        st.caption(
            f"Active set: {last_report.pages} pages, {last_report.chunks} chunks"
        )
        for idx, filename in enumerate(last_report.files, start=1):
            st.markdown(f"**{idx}.** `{filename}`")
    else:
        for idx, filename in enumerate(engine.loaded_documents, start=1):
            st.markdown(f"**{idx}.** `{filename}`")


def render_cache_settings(engine: DocuMateEngine, settings: Settings) -> None:
    """Render semantic cache configuration controls and statistics.

    Args:
        engine: Active DocuMateEngine instance.
        settings: Application settings for slider bounds.
    """
    st.subheader("⚡ Cache Settings")

    st.session_state["cache_enabled"] = st.toggle(
        "Enable semantic caching",
        value=bool(st.session_state.get("cache_enabled", True)),
        help="Check and save answers to the local semantic cache.",
    )

    if st.session_state["cache_enabled"]:
        current_threshold = float(
            st.session_state.get(
                "similarity_threshold", settings.similarity_threshold
            )
        )
        st.session_state["similarity_threshold"] = st.slider(
            "Similarity threshold",
            min_value=settings.similarity_slider_min,
            max_value=settings.similarity_slider_max,
            value=current_threshold,
            step=settings.similarity_slider_step,
            help="Higher = stricter matching. Lower = more cache hits.",
        )

    # Cache statistics and clear cache action
    cache_mgr = engine.cache_manager
    count = cache_mgr.entry_count if cache_mgr is not None else 0
    st.caption(f"Cached answers in store: **{count}**")

    if st.button("🧹 Clear Cache", width="stretch", help="Clear cached answers"):
        cleared = engine.clear_cache(all_documents=False)
        st.success(f"Cleared {cleared} cached answer(s).")
        st.rerun()


def render_action_buttons(engine: DocuMateEngine) -> None:
    """Render Clear Chat and Reset All actions.

    Args:
        engine: Active DocuMateEngine instance.
    """
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "🗑️ Clear Chat",
            width="stretch",
            disabled=not engine.is_ready,
            help="Clear current conversation history.",
        ):
            clear_chat()
            st.rerun()

    with col2:
        if st.button(
            "🔄 Reset All",
            width="stretch",
            help="Reset session, clear history, and purge vector store.",
        ):
            reset_all(delete_store=True)
            st.session_state.pop("last_ingest_report", None)
            st.rerun()


def render_debug_toggle() -> None:
    """Render the debug mode checkbox."""
    st.session_state["debug_mode"] = st.checkbox(
        "🐛 Debug mode",
        value=bool(st.session_state.get("debug_mode", False)),
        help="Display timing breakdowns, similarity scores, and retrieved chunk metadata.",
    )


def render_sidebar(
    engine: DocuMateEngine | None = None,
    settings: Settings | None = None,
) -> None:
    """Render the complete DocuMate-AI sidebar layout.

    Coordinates health banner, document uploading, loaded document statistics,
    cache controls, action buttons, and debug toggle.

    Args:
        engine: Optional DocuMateEngine. If None, retrieved from session state.
        settings: Optional application settings. If None, loaded via get_settings().
    """
    if settings is None:
        settings = get_settings()
    if engine is None:
        engine = get_engine()

    with st.sidebar:
        st.title("📄 DocuMate-AI")
        st.caption("Intelligent, on-device PDF analysis")
        st.divider()

        # 1. Ollama Health Banner
        health = render_health_banner(settings)
        st.divider()

        # 2. PDF Upload & Ingest
        render_document_uploader(engine, health)
        st.divider()

        # 3. Loaded Documents List
        render_loaded_documents(engine)
        if engine.is_ready:
            st.divider()

        # 4. Cache Controls
        render_cache_settings(engine, settings)
        st.divider()

        # 5. Clear Chat and Reset All Actions
        render_action_buttons(engine)

        # 6. Debug Toggle
        render_debug_toggle()
