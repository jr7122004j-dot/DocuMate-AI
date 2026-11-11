"""Unit tests for Ollama model factories and health check functionality."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from documate_ai.config import Settings
from documate_ai.errors import ModelNotFoundError, OllamaUnavailableError
from documate_ai.models import (
    HealthReport,
    check_ollama_health,
    create_embeddings,
    create_llm,
)


@pytest.fixture
def mock_settings() -> Settings:
    """Fixture providing a deterministic test Settings instance."""
    return Settings(
        ollama_base_url="http://mock-ollama:11434",
        llm_model_name="phi4-mini",
        embedding_model_name="all-minilm:l6-v2",
        llm_temperature=0.2,
        llm_top_k=15,
        llm_top_p=0.85,
        llm_repeat_penalty=1.15,
        llm_num_ctx=4096,
    )


# ---------------------------------------------------------------------------
# check_ollama_health scenarios
# ---------------------------------------------------------------------------


def test_check_ollama_health_scenario_1_all_models_present(
    mock_settings: Settings,
) -> None:
    """Scenario 1: Ollama reachable and all required models present."""
    mock_client = MagicMock()
    mock_client.list.return_value = MagicMock(
        models=[
            MagicMock(model="phi4-mini"),
            MagicMock(model="all-minilm:l6-v2"),
        ]
    )

    with patch("ollama.Client", return_value=mock_client):
        report = check_ollama_health(mock_settings)

    mock_client.list.assert_called_once()
    assert isinstance(report, HealthReport)
    assert report.server_reachable is True
    assert report.missing_models == []
    assert "phi4-mini" in report.available_models
    assert "all-minilm:l6-v2" in report.available_models
    assert report.error_message is None


def test_check_ollama_health_scenario_2_model_missing(
    mock_settings: Settings,
) -> None:
    """Scenario 2: Ollama reachable but embedding or LLM model missing."""
    mock_client = MagicMock()
    # Only embedding model is available, LLM model phi4-mini is missing
    mock_client.list.return_value = {
        "models": [{"model": "all-minilm:l6-v2"}]
    }

    with patch("ollama.Client", return_value=mock_client):
        report = check_ollama_health(mock_settings)

    assert report.server_reachable is True
    assert report.missing_models == ["phi4-mini"]
    assert "all-minilm:l6-v2" in report.available_models
    assert report.error_message is None


def test_check_ollama_health_scenario_3_server_unreachable(
    mock_settings: Settings,
) -> None:
    """Scenario 3: Ollama unreachable / connection refused."""
    mock_client = MagicMock()
    mock_client.list.side_effect = ConnectionRefusedError("Connection refused by host")

    with patch("ollama.Client", return_value=mock_client):
        report = check_ollama_health(mock_settings)

    assert report.server_reachable is False
    assert report.available_models == []
    assert report.error_message is not None
    assert "Connection refused" in report.error_message


def test_check_ollama_health_matches_tagged_models(
    mock_settings: Settings,
) -> None:
    """Check that models tagged with :latest or without tags match correctly."""
    mock_client = MagicMock()
    mock_client.list.return_value = {
        "models": [
            {"model": "phi4-mini:latest"},
            {"name": "all-minilm:l6-v2:latest"},
        ]
    }

    with patch("ollama.Client", return_value=mock_client):
        report = check_ollama_health(mock_settings)

    assert report.server_reachable is True
    assert report.missing_models == []


def test_check_ollama_health_strict_mode(
    mock_settings: Settings,
) -> None:
    """Strict mode raises appropriate exceptions on failure."""
    # When server unreachable
    mock_client_unreachable = MagicMock()
    mock_client_unreachable.list.side_effect = TimeoutError("Timed out")
    with patch("ollama.Client", return_value=mock_client_unreachable):
        with pytest.raises(OllamaUnavailableError) as exc_info:
            check_ollama_health(mock_settings, strict=True)
        assert exc_info.value.base_url == mock_settings.ollama_base_url

    # When model missing
    mock_client_missing = MagicMock()
    mock_client_missing.list.return_value = {"models": []}
    with patch("ollama.Client", return_value=mock_client_missing):
        with pytest.raises(ModelNotFoundError):
            check_ollama_health(mock_settings, strict=True)


# ---------------------------------------------------------------------------
# create_embeddings tests
# ---------------------------------------------------------------------------


def test_create_embeddings_honors_settings(mock_settings: Settings) -> None:
    """create_embeddings correctly passes base_url and model from settings."""
    embeddings = create_embeddings(mock_settings)
    assert embeddings.model == "all-minilm:l6-v2"
    assert embeddings.base_url == "http://mock-ollama:11434"


def test_create_embeddings_custom_model_override(mock_settings: Settings) -> None:
    """create_embeddings allows overriding model_name."""
    embeddings = create_embeddings(mock_settings, model_name="nomic-embed-text")
    assert embeddings.model == "nomic-embed-text"
    assert embeddings.base_url == "http://mock-ollama:11434"


def test_create_embeddings_validation_mode(mock_settings: Settings) -> None:
    """create_embeddings with validate=True raises on missing model."""
    mock_client = MagicMock()
    mock_client.list.return_value = {"models": [{"model": "other-model"}]}
    with patch("ollama.Client", return_value=mock_client):
        with pytest.raises(ModelNotFoundError) as exc_info:
            create_embeddings(mock_settings, validate=True)
        assert exc_info.value.model_name == mock_settings.embedding_model_name


# ---------------------------------------------------------------------------
# create_llm tests
# ---------------------------------------------------------------------------


def test_create_llm_honors_settings(mock_settings: Settings) -> None:
    """create_llm correctly passes all parameters and context window from settings."""
    llm = create_llm(mock_settings)
    assert llm.model == "phi4-mini"
    assert llm.base_url == "http://mock-ollama:11434"
    assert llm.num_ctx == 4096
    assert llm.temperature == 0.2
    assert llm.top_k == 15
    assert llm.top_p == 0.85
    assert llm.repeat_penalty == 1.15


def test_create_llm_custom_parameter_overrides(mock_settings: Settings) -> None:
    """create_llm allows overriding model name, context window, and sampling params."""
    llm = create_llm(
        mock_settings,
        model_name="llama3.2:1b",
        temperature=0.8,
        top_k=25,
        top_p=0.92,
        repeat_penalty=1.05,
        num_ctx=8192,
    )
    assert llm.model == "llama3.2:1b"
    assert llm.base_url == "http://mock-ollama:11434"
    assert llm.num_ctx == 8192
    assert llm.temperature == 0.8
    assert llm.top_k == 25
    assert llm.top_p == 0.92
    assert llm.repeat_penalty == 1.05


def test_create_llm_validation_mode_unreachable(mock_settings: Settings) -> None:
    """create_llm with validate=True raises OllamaUnavailableError when server unreachable."""
    mock_client = MagicMock()
    mock_client.list.side_effect = ConnectionResetError("Reset by peer")
    with patch("ollama.Client", return_value=mock_client):
        with pytest.raises(OllamaUnavailableError):
            create_llm(mock_settings, validate=True)

