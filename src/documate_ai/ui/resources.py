"""Streamlit resource caching for DocuMate-AI.

Provides global, shared resources (embeddings model, chat language model,
semantic cache manager) cached across user sessions using Streamlit's
``@st.cache_resource`` decorator.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import streamlit as st
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from documate_ai.cache_manager import CacheManager
from documate_ai.config import Settings, get_settings
from documate_ai.engine import DocuMateEngine
from documate_ai.models import create_embeddings, create_llm

if TYPE_CHECKING:
    from langchain_ollama import ChatOllama, OllamaEmbeddings

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def get_embeddings(
    model_name: str | None = None,
    base_url: str | None = None,
) -> OllamaEmbeddings:
    """Retrieve or build a globally cached OllamaEmbeddings model.

    Streamlit caches the returned instance across user sessions keyed by
    ``model_name`` and ``base_url``. If either changes, a new model is built.

    Args:
        model_name: Name of the Ollama embedding model.
        base_url: Base URL of the Ollama server.

    Returns:
        Configured ``OllamaEmbeddings`` instance.
    """
    settings = get_settings()
    effective_model = model_name or settings.embedding_model_name
    return create_embeddings(
        settings=settings,
        model_name=effective_model,
    )


@st.cache_resource(show_spinner=False)
def get_llm(
    model_name: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    repeat_penalty: float | None = None,
    num_ctx: int | None = None,
) -> ChatOllama:
    """Retrieve or build a globally cached ChatOllama language model.

    Streamlit caches the returned instance across user sessions keyed by
    model arguments. If any parameter changes, a new model is built.

    Args:
        model_name: Name of the Ollama chat model.
        base_url: Base URL of the Ollama server.
        temperature: Sampling temperature.
        top_k: Top-k sampling limit.
        top_p: Top-p nucleus sampling mass.
        repeat_penalty: Repetition penalty.
        num_ctx: Context window size in tokens.

    Returns:
        Configured ``ChatOllama`` instance.
    """
    settings = get_settings()
    effective_model = model_name or settings.llm_model_name
    return create_llm(
        settings=settings,
        model_name=effective_model,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        num_ctx=num_ctx,
    )


@st.cache_resource(show_spinner=False)
def get_cache_manager(
    cache_dir: str | None = None,
    threshold: float | None = None,
) -> CacheManager:
    """Retrieve or build a globally cached CacheManager instance.

    Streamlit caches the returned CacheManager across user sessions keyed
    by ``cache_dir`` and ``threshold``.

    Args:
        cache_dir: Path to disk cache directory as a string.
        threshold: Default similarity threshold score.

    Returns:
        Configured ``CacheManager`` instance.
    """
    settings = get_settings()
    c_dir = Path(cache_dir) if cache_dir is not None else settings.cache_dir
    c_threshold = threshold if threshold is not None else settings.similarity_threshold
    return CacheManager(
        cache_dir=c_dir,
        threshold=c_threshold,
        settings=settings,
    )


def create_engine(
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
    llm: BaseChatModel | None = None,
    cache_manager: CacheManager | None = None,
) -> DocuMateEngine:
    """Instantiate a new per-session DocuMateEngine backed by cached shared resources.

    Args:
        settings: Application settings.
        embeddings: Optional pre-configured or test Embeddings instance.
        llm: Optional pre-configured or test BaseChatModel instance.
        cache_manager: Optional pre-configured or test CacheManager instance.

    Returns:
        A new ``DocuMateEngine`` configured for a user session.
    """
    if settings is None:
        settings = get_settings()

    effective_embeddings = (
        embeddings
        if embeddings is not None
        else get_embeddings(
            model_name=settings.embedding_model_name,
            base_url=settings.ollama_base_url,
        )
    )

    effective_llm = (
        llm
        if llm is not None
        else get_llm(
            model_name=settings.llm_model_name,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
            top_k=settings.llm_top_k,
            top_p=settings.llm_top_p,
            repeat_penalty=settings.llm_repeat_penalty,
            num_ctx=settings.llm_num_ctx,
        )
    )

    effective_cache = (
        cache_manager
        if cache_manager is not None
        else get_cache_manager(
            cache_dir=str(settings.cache_dir) if settings.cache_dir else None,
            threshold=settings.similarity_threshold,
        )
    )

    return DocuMateEngine(
        embeddings=effective_embeddings,
        llm=effective_llm,
        cache_manager=effective_cache,
        settings=settings,
    )
