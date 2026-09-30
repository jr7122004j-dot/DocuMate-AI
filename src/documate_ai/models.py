"""LLM and embedding model initialisation.

Provides factory functions for creating Ollama-backed language models
and embedding models. In the old code these used Streamlit's
``@st.cache_resource``; here they are plain functions so the caching
strategy can be decided by the caller (Streamlit UI layer or otherwise).
"""

from __future__ import annotations

from langchain_ollama import ChatOllama, OllamaEmbeddings

from documate_ai.config import (
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    LLM_REPEAT_PENALTY,
    LLM_TEMPERATURE,
    LLM_TOP_K,
    LLM_TOP_P,
)


def create_embeddings(
    model_name: str = EMBEDDING_MODEL_NAME,
) -> OllamaEmbeddings:
    """Create an ``OllamaEmbeddings`` instance.

    Args:
        model_name: Name of the Ollama embedding model to load.

    Returns:
        A configured ``OllamaEmbeddings`` object ready to embed text.
    """
    return OllamaEmbeddings(model=model_name)


def create_llm(
    model_name: str = LLM_MODEL_NAME,
    temperature: float = LLM_TEMPERATURE,
    top_k: int = LLM_TOP_K,
    top_p: float = LLM_TOP_P,
    repeat_penalty: float = LLM_REPEAT_PENALTY,
) -> ChatOllama:
    """Create a ``ChatOllama`` instance for chat-based generation.

    Args:
        model_name: Name of the Ollama chat model.
        temperature: Sampling temperature — lower is more deterministic.
        top_k: Highest-probability tokens considered during sampling.
        top_p: Nucleus-sampling cumulative probability mass.
        repeat_penalty: Penalty for repeated tokens in context.

    Returns:
        A configured ``ChatOllama`` object ready for inference.
    """
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
    )
