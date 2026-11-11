"""LLM and embedding model initialisation.

Provides factory functions for creating Ollama-backed language models
and embedding models, along with an Ollama server health check. In the old
code these used Streamlit's ``@st.cache_resource``; here they are plain
functions so the caching strategy can be decided by the caller (Streamlit
UI layer or otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import ollama
from langchain_ollama import ChatOllama, OllamaEmbeddings

from documate_ai.config import (
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    LLM_REPEAT_PENALTY,
    LLM_TEMPERATURE,
    LLM_TOP_K,
    LLM_TOP_P,
    Settings,
    get_settings,
)
from documate_ai.errors import ModelNotFoundError, OllamaUnavailableError


@dataclass
class HealthReport:
    """Ollama server health and model availability status."""

    server_reachable: bool
    available_models: list[str] = field(default_factory=list)
    missing_models: list[str] = field(default_factory=list)
    error_message: str | None = None


def _model_matches(required: str, available: str) -> bool:
    """Check if an available Ollama model matches the required model identifier."""
    if required == available:
        return True
    if available == f"{required}:latest":
        return True
    if required == f"{available}:latest":
        return True
    return False


def _extract_model_name(model_obj: Any) -> str:
    """Extract model name string from an Ollama list response entry."""
    if isinstance(model_obj, str):
        return model_obj
    if isinstance(model_obj, dict):
        val = model_obj.get("model") or model_obj.get("name")
        return str(val) if val is not None else ""
    model_attr = getattr(model_obj, "model", None)
    if isinstance(model_attr, str):
        return model_attr
    name_attr = getattr(model_obj, "name", None)
    if isinstance(name_attr, str):
        return name_attr
    return str(model_obj)


def _validate_model_ready(base_url: str, model_name: str) -> None:
    """Validate that Ollama is reachable and has the requested model available."""
    try:
        client = ollama.Client(host=base_url, timeout=5.0)
        resp = client.list()
        raw_models = getattr(resp, "models", None)
        if raw_models is None and isinstance(resp, dict):
            raw_models = resp.get("models", [])
        available = [_extract_model_name(m) for m in (raw_models or [])]
    except Exception as exc:
        raise OllamaUnavailableError(
            f"Ollama server at '{base_url}' is unavailable: {exc}",
            base_url=base_url,
        ) from exc

    if not any(_model_matches(model_name, a) for a in available):
        raise ModelNotFoundError(
            f"Requested model '{model_name}' was not found on Ollama server.",
            model_name=model_name,
        )


def check_ollama_health(
    settings: Settings | None = None,
    *,
    strict: bool = False,
    validate: bool = False,
) -> HealthReport:
    """Check if the Ollama server is running and required models are pulled.

    Args:
        settings: Application settings. If None, loaded via ``get_settings()``.
        strict: If True, raises ``OllamaUnavailableError`` or ``ModelNotFoundError``.
        validate: Alias for strict validation.

    Returns:
        A ``HealthReport`` indicating server reachability and model status.

    Raises:
        OllamaUnavailableError: If strict validation is requested and Ollama is unreachable.
        ModelNotFoundError: If strict validation is requested and required models are missing.
    """
    if settings is None:
        settings = get_settings()

    base_url = settings.ollama_base_url
    required_models = [settings.llm_model_name, settings.embedding_model_name]

    try:
        client = ollama.Client(host=base_url, timeout=5.0)
        resp = client.list()
        raw_models = getattr(resp, "models", None)
        if raw_models is None and isinstance(resp, dict):
            raw_models = resp.get("models", [])
        available = [_extract_model_name(m) for m in (raw_models or [])]
        missing = [
            req
            for req in required_models
            if not any(_model_matches(req, avail) for avail in available)
        ]
        report = HealthReport(
            server_reachable=True,
            available_models=available,
            missing_models=missing,
            error_message=None,
        )
    except Exception as exc:
        report = HealthReport(
            server_reachable=False,
            available_models=[],
            missing_models=[],
            error_message=str(exc),
        )

    if strict or validate:
        if not report.server_reachable:
            raise OllamaUnavailableError(
                f"Ollama server at '{base_url}' is unavailable: {report.error_message}",
                base_url=base_url,
            )
        if report.missing_models:
            raise ModelNotFoundError(
                f"Required model(s) not found on Ollama server: {', '.join(report.missing_models)}",
                model_name=report.missing_models[0],
            )

    return report


def create_embeddings(
    settings: Settings | None = None,
    model_name: str | None = None,
    *,
    validate: bool = False,
    strict: bool = False,
) -> OllamaEmbeddings:
    """Create an ``OllamaEmbeddings`` instance.

    Args:
        settings: Application settings. If None, loaded via ``get_settings()``.
        model_name: Name of the Ollama embedding model. If None, uses settings default.
        validate: If True, verifies Ollama reachability and model availability.
        strict: Alias for validate.

    Returns:
        A configured ``OllamaEmbeddings`` object ready to embed text.

    Raises:
        OllamaUnavailableError: If validation is requested and Ollama is unreachable.
        ModelNotFoundError: If validation is requested and model is missing.
    """
    if isinstance(settings, str):
        model_name = settings
        settings = None

    if settings is None:
        settings = get_settings()

    effective_model = model_name or settings.embedding_model_name
    base_url = settings.ollama_base_url

    if validate or strict:
        _validate_model_ready(base_url, effective_model)

    return OllamaEmbeddings(
        model=effective_model,
        base_url=base_url,
    )


def create_llm(
    settings: Settings | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    repeat_penalty: float | None = None,
    num_ctx: int | None = None,
    *,
    validate: bool = False,
    strict: bool = False,
) -> ChatOllama:
    """Create a ``ChatOllama`` instance for chat-based generation.

    Args:
        settings: Application settings. If None, loaded via ``get_settings()``.
        model_name: Name of the Ollama chat model.
        temperature: Sampling temperature — lower is more deterministic.
        top_k: Highest-probability tokens considered during sampling.
        top_p: Nucleus-sampling cumulative probability mass.
        repeat_penalty: Penalty for repeated tokens in context.
        num_ctx: Context window size in tokens.
        validate: If True, verifies Ollama reachability and model availability.
        strict: Alias for validate.

    Returns:
        A configured ``ChatOllama`` object ready for inference.

    Raises:
        OllamaUnavailableError: If validation is requested and Ollama is unreachable.
        ModelNotFoundError: If validation is requested and model is missing.
    """
    if isinstance(settings, str):
        model_name = settings
        settings = None

    if settings is None:
        settings = get_settings()

    effective_model = model_name or settings.llm_model_name
    effective_temp = temperature if temperature is not None else settings.llm_temperature
    effective_top_k = top_k if top_k is not None else settings.llm_top_k
    effective_top_p = top_p if top_p is not None else settings.llm_top_p
    effective_repeat = (
        repeat_penalty if repeat_penalty is not None else settings.llm_repeat_penalty
    )
    effective_num_ctx = (
        num_ctx
        if num_ctx is not None
        else getattr(settings, "num_ctx", getattr(settings, "llm_num_ctx", 4096))
    )
    base_url = settings.ollama_base_url

    if validate or strict:
        _validate_model_ready(base_url, effective_model)

    return ChatOllama(
        model=effective_model,
        base_url=base_url,
        num_ctx=effective_num_ctx,
        temperature=effective_temp,
        top_k=effective_top_k,
        top_p=effective_top_p,
        repeat_penalty=effective_repeat,
    )
