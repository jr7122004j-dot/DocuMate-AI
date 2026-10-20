"""Unit tests for configuration, settings validation, errors, and logging."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from documate_ai.config import (
    CONDENSE_QUESTION_HUMAN_PROMPT,
    CONDENSE_QUESTION_SYSTEM_PROMPT,
    CONDENSE_QUESTION_TEMPLATE,
    NO_INFO_RESPONSE,
    QA_HUMAN_PROMPT,
    QA_PROMPT_TEMPLATE,
    QA_SYSTEM_PROMPT,
    Settings,
    get_settings,
)
from documate_ai.errors import (
    DocuMateError,
    DocumentProcessingError,
    EngineNotReadyError,
    ModelNotFoundError,
    OllamaUnavailableError,
)
from documate_ai.logging_config import configure_logging, set_debug_mode


def test_default_settings() -> None:
    """Settings should initialize with documented default values."""
    settings = Settings()

    assert settings.llm_model_name == "phi4-mini"
    assert settings.embedding_model_name == "all-minilm:l6-v2"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.llm_temperature == 0.15
    assert settings.llm_top_k == 10
    assert settings.llm_top_p == 0.9
    assert settings.llm_repeat_penalty == 1.1
    assert settings.llm_num_ctx == 4096

    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 100
    assert settings.retriever_result_count == 3
    assert settings.max_pdf_pages == 500
    assert settings.max_upload_size_mb == 50

    assert settings.data_dir == Path("./data")
    assert settings.cache_dir in (Path("data/cache"), Path("./data/cache"))
    assert settings.chroma_dir in (Path("data/chroma_db"), Path("./data/chroma_db"))

    assert settings.similarity_threshold == 0.85
    assert settings.similarity_slider_min == 0.50
    assert settings.similarity_slider_max == 0.99
    assert settings.similarity_slider_step == 0.01

    assert settings.log_level == "INFO"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """DOCUMATE_ prefixed environment variables should override default settings."""
    monkeypatch.setenv("DOCUMATE_LLM_MODEL_NAME", "llama3.2")
    monkeypatch.setenv("DOCUMATE_EMBEDDING_MODEL_NAME", "bge-m3")
    monkeypatch.setenv("DOCUMATE_CHUNK_SIZE", "800")
    monkeypatch.setenv("DOCUMATE_CHUNK_OVERLAP", "150")
    monkeypatch.setenv("DOCUMATE_SIMILARITY_THRESHOLD", "0.92")
    monkeypatch.setenv("DOCUMATE_LOG_LEVEL", "DEBUG")

    settings = Settings()
    assert settings.llm_model_name == "llama3.2"
    assert settings.embedding_model_name == "bge-m3"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 150
    assert settings.similarity_threshold == 0.92
    assert settings.log_level == "DEBUG"


def test_validation_chunk_overlap_ge_chunk_size() -> None:
    """chunk_overlap must be strictly less than chunk_size."""
    with pytest.raises(ValidationError, match="chunk_overlap .* must be strictly less than"):
        Settings(chunk_size=200, chunk_overlap=200)

    with pytest.raises(ValidationError, match="chunk_overlap .* must be strictly less than"):
        Settings(chunk_size=200, chunk_overlap=250)


def test_validation_chunk_size_positive() -> None:
    """chunk_size must be positive and overlap non-negative."""
    with pytest.raises(ValidationError, match="chunk_size must be positive"):
        Settings(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValidationError, match="chunk_overlap must be non-negative"):
        Settings(chunk_size=100, chunk_overlap=-10)


def test_validation_similarity_threshold_bounds() -> None:
    """similarity_threshold must be within (0, 1]."""
    with pytest.raises(ValidationError, match="similarity_threshold .* must be between 0 and 1"):
        Settings(similarity_threshold=0.0)

    with pytest.raises(ValidationError, match="similarity_threshold .* must be between 0 and 1"):
        Settings(similarity_threshold=1.5)


def test_validation_similarity_slider_bounds() -> None:
    """similarity_threshold must fall within slider min and max."""
    with pytest.raises(ValidationError, match="similarity_slider_min .* must be less than"):
        Settings(similarity_slider_min=0.90, similarity_slider_max=0.50)

    with pytest.raises(ValidationError, match="must be between"):
        Settings(
            similarity_slider_min=0.60,
            similarity_slider_max=0.90,
            similarity_threshold=0.55,
        )


def test_path_resolution_relative_to_data_dir() -> None:
    """Relative cache and chroma directories should resolve under data_dir."""
    custom_data = Path("./test_data_dir")
    settings = Settings(
        data_dir=custom_data,
        cache_dir=Path("custom_cache"),
        chroma_dir=Path("custom_chroma"),
    )

    assert settings.cache_dir == custom_data / "custom_cache"
    assert settings.chroma_dir == custom_data / "custom_chroma"


def test_get_settings_cached() -> None:
    """get_settings() should return a cached singleton instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_error_hierarchy() -> None:
    """All custom exceptions should inherit from DocuMateError."""
    assert issubclass(OllamaUnavailableError, DocuMateError)
    assert issubclass(ModelNotFoundError, DocuMateError)
    assert issubclass(DocumentProcessingError, DocuMateError)
    assert issubclass(EngineNotReadyError, DocuMateError)

    err = ModelNotFoundError("Missing model", model_name="test-model")
    assert err.model_name == "test-model"
    assert str(err) == "Missing model"

    server_err = OllamaUnavailableError(base_url="http://localhost:11434")
    assert server_err.base_url == "http://localhost:11434"

    doc_err = DocumentProcessingError("Corrupted file", filename="bad.pdf")
    assert doc_err.filename == "bad.pdf"


def test_logging_configuration() -> None:
    """configure_logging and set_debug_mode should configure log levels without error."""
    configure_logging(level="WARNING")
    app_logger = logging.getLogger("documate_ai")
    assert app_logger.level == logging.WARNING

    set_debug_mode(True)
    assert app_logger.level == logging.DEBUG

    set_debug_mode(False)
    assert app_logger.level == logging.INFO


def test_prompt_templates_and_sentinel() -> None:
    """Prompt templates should contain required placeholders and the sentinel string."""
    assert "{context}" in QA_PROMPT_TEMPLATE
    assert "{question}" in QA_PROMPT_TEMPLATE
    assert NO_INFO_RESPONSE in QA_PROMPT_TEMPLATE

    assert "{context}" in QA_HUMAN_PROMPT
    assert "{question}" in QA_HUMAN_PROMPT
    assert NO_INFO_RESPONSE in QA_SYSTEM_PROMPT

    assert "{chat_history}" in CONDENSE_QUESTION_TEMPLATE
    assert "{question}" in CONDENSE_QUESTION_TEMPLATE
    assert "{chat_history}" in CONDENSE_QUESTION_HUMAN_PROMPT
    assert "{question}" in CONDENSE_QUESTION_HUMAN_PROMPT
    assert CONDENSE_QUESTION_SYSTEM_PROMPT
