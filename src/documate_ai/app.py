"""DocuMate-AI — Streamlit application entry point.

Run with:
    streamlit run src/documate_ai/app.py

UI Layout:
    ┌────────────────────┬──────────────────────────────────┐
    │  SIDEBAR           │  MAIN PANEL                      │
    │                    │                                  │
    │  🏥 Ollama Health  │  💬 Chat History                 │
    │  📄 PDF Upload     │     User messages                │
    │  ⚙️ Process PDFs   │     AI responses with citations  │
    │  📋 Loaded PDFs    │     ⚡ Cache indicators           │
    │  🗑️ Clear / Reset  │     🐛 Debug metadata            │
    │  ⚡ Cache Settings  │                                  │
    │  🐛 Debug Toggle   │  ⌨️ User Input Box               │
    └────────────────────┴──────────────────────────────────┘
"""

from __future__ import annotations

import streamlit as st

from documate_ai.config import get_settings
from documate_ai.logging_config import configure_logging
from documate_ai.session_state import get_engine, initialise_session_state
from documate_ai.ui.chat import render_chat
from documate_ai.ui.sidebar import render_sidebar

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit command)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DocuMate-AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Main application loop."""
    settings = get_settings()
    configure_logging(level=settings.log_level)

    # Ensure all session state keys exist
    initialise_session_state(settings=settings)

    # Retrieve engine
    engine = get_engine()

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    render_sidebar(engine=engine, settings=settings)

    # ------------------------------------------------------------------
    # Main chat panel
    # ------------------------------------------------------------------
    render_chat(engine=engine, settings=settings)


if __name__ == "__main__":
    main()
