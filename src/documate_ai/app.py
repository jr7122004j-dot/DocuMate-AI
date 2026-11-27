"""DocuMate-AI — Streamlit application entry point.

This is the main file that ties all modules together. Run with:
    streamlit run src/documate_ai/app.py

UI Layout:
    ┌────────────────────┬──────────────────────────────────┐
    │  SIDEBAR           │  MAIN PANEL                      │
    │                    │                                  │
    │  📄 PDF Upload     │  💬 Chat History                 │
    │  ⚙️ Process PDFs   │     User messages                │
    │  📋 Loaded PDFs    │     AI responses with citations  │
    │  🗑️ Clear / Reset  │                                  │
    │  ⚡ Cache Settings  │  ⌨️ User Input Box               │
    │  🐛 Debug Toggle   │                                  │
    └────────────────────┴──────────────────────────────────┘
"""

from __future__ import annotations

import streamlit as st

from documate_ai.config import get_settings
from documate_ai.logging_config import configure_logging
from documate_ai.session_state import initialise_session_state
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
    initialise_session_state()

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    render_sidebar(settings=settings)

    # ------------------------------------------------------------------
    # Main chat panel
    # ------------------------------------------------------------------
    st.title("💬 Chat with your Documents")

    # Render chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add to history
        st.session_state.chat_history.append(("user", prompt))

        # Check if PDFs are loaded
        if st.session_state.conversation_chain is None:
            with st.chat_message("assistant"):
                st.warning("Please upload and process PDF documents first.")
            st.session_state.chat_history.append(
                ("assistant", "⚠️ Please upload and process PDF documents first.")
            )
        else:
            # Generate response
            with st.chat_message("assistant"):
                response_container = st.empty()

                # TODO: Check cache first (if enabled)
                # TODO: If cache miss, run conversation chain with StreamHandler
                # TODO: Format response with citations
                # TODO: Save to cache

                response_container.markdown("_Response generation not yet implemented._")

            st.session_state.chat_history.append(
                ("assistant", "_Response generation not yet implemented._")
            )


if __name__ == "__main__":
    main()
