"""Streamlit session state initialisation.

Declares all session state keys with their default values. This module
is called once at app startup to ensure every key exists before any
other module tries to access it.
"""

from __future__ import annotations

import streamlit as st


def initialise_session_state() -> None:
    """Set default values for all Streamlit session state keys.

    This function is idempotent — calling it multiple times will not
    overwrite values that have already been set by user interactions.

    Session state keys managed:
        - ``chat_history``: List of (role, message) tuples
        - ``uploaded_pdfs``: List of uploaded PDF file dicts
        - ``vector_store``: The active ChromaDB vector store (or None)
        - ``conversation_chain``: The active RAG chain (or None)
        - ``embeddings``: The cached embedding model (or None)
        - ``llm``: The cached LLM (or None)
        - ``cache_manager``: The CacheManager instance (or None)
        - ``cache_enabled``: Whether semantic caching is on
        - ``similarity_threshold``: Current similarity threshold slider value
        - ``processed_pdf_names``: Set of already-processed PDF filenames
        - ``debug_mode``: Whether to show debug info in the UI
    """
    defaults: dict = {
        "chat_history": [],
        "uploaded_pdfs": [],
        "vector_store": None,
        "conversation_chain": None,
        "embeddings": None,
        "llm": None,
        "cache_manager": None,
        "cache_enabled": True,
        "similarity_threshold": 0.95,
        "processed_pdf_names": set(),
        "debug_mode": False,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
