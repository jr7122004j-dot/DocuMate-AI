"""Unit tests for the sidebar UI component (T17).

Tests:
    1. Ollama health banner (ready status, unreachable server, missing models, re-check).
    2. Document uploader (disabled states, ingestion trigger, error handling, skipped files).
    3. Loaded documents list (hidden when engine not ready, rendered when ready).
    4. Cache settings controls (toggle, slider bounds, clear cache action, stats).
    5. Action buttons (Clear Chat, Reset All).
    6. Debug toggle.
    7. Full render_sidebar coordination.
"""

from __future__ import annotations

import os

# Ensure telemetry is disabled before any chromadb import
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from documate_ai.cache_manager import CacheManager
from documate_ai.config import Settings
from documate_ai.engine import DocuMateEngine
from documate_ai.errors import DocumentProcessingError
from documate_ai.models import HealthReport
from documate_ai.schemas import IngestReport, UploadedPdf
from documate_ai.ui.sidebar import (
    render_action_buttons,
    render_cache_settings,
    render_debug_toggle,
    render_document_uploader,
    render_health_banner,
    render_loaded_documents,
    render_sidebar,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_session_state():
    """Ensure clean session state before and after each test."""
    st.session_state.clear()
    yield
    st.session_state.clear()


@pytest.fixture
def fake_settings(tmp_path: Path) -> Settings:
    """Settings configured with temp directory and custom slider bounds."""
    return Settings(
        data_dir=tmp_path,
        similarity_threshold=0.85,
        similarity_slider_min=0.50,
        similarity_slider_max=0.99,
        similarity_slider_step=0.01,
    )


@pytest.fixture
def mock_engine(tmp_path: Path, fake_settings: Settings) -> DocuMateEngine:
    """Create a mock DocuMateEngine for testing sidebar component interactions."""
    embeddings = DeterministicFakeEmbedding(size=384)
    llm = FakeListChatModel(responses=["Test"])
    cache_mgr = CacheManager(cache_dir=tmp_path / "cache", settings=fake_settings)
    engine = DocuMateEngine(
        embeddings=embeddings,
        llm=llm,
        cache_manager=cache_mgr,
        settings=fake_settings,
    )
    return engine


class DummyUploadedFile:
    """Simulates a Streamlit UploadedFile object."""

    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


# ---------------------------------------------------------------------------
# Health Banner Tests
# ---------------------------------------------------------------------------


class TestHealthBanner:
    @patch("documate_ai.ui.sidebar.check_ollama_health")
    def test_health_banner_ready(
        self,
        mock_check: MagicMock,
        fake_settings: Settings,
    ) -> None:
        """When Ollama is running and all models are present, shows success."""
        mock_check.return_value = HealthReport(
            server_reachable=True,
            available_models=["phi4-mini", "all-minilm:l6-v2"],
            missing_models=[],
        )

        with patch("streamlit.success") as mock_success:
            report = render_health_banner(fake_settings)

            assert report.server_reachable is True
            assert report.missing_models == []
            mock_success.assert_called_once()
            assert "Ollama Ready" in mock_success.call_args[0][0]

    @patch("documate_ai.ui.sidebar.check_ollama_health")
    def test_health_banner_server_unreachable(
        self,
        mock_check: MagicMock,
        fake_settings: Settings,
    ) -> None:
        """When Ollama is unreachable, shows actionable error with server URL."""
        mock_check.return_value = HealthReport(
            server_reachable=False,
            available_models=[],
            missing_models=[],
            error_message="Connection refused",
        )

        with patch("streamlit.error") as mock_error:
            report = render_health_banner(fake_settings)

            assert report.server_reachable is False
            mock_error.assert_called_once()
            assert "Ollama Unavailable" in mock_error.call_args[0][0]
            assert fake_settings.ollama_base_url in mock_error.call_args[0][0]

    @patch("documate_ai.ui.sidebar.check_ollama_health")
    def test_health_banner_missing_models(
        self,
        mock_check: MagicMock,
        fake_settings: Settings,
    ) -> None:
        """When models are missing, shows error and ollama pull command code blocks."""
        mock_check.return_value = HealthReport(
            server_reachable=True,
            available_models=[],
            missing_models=["phi4-mini", "all-minilm:l6-v2"],
        )

        with patch("streamlit.error") as mock_error, patch("streamlit.code") as mock_code:
            report = render_health_banner(fake_settings)

            assert len(report.missing_models) == 2
            mock_error.assert_called_once()
            assert "Missing Model(s)" in mock_error.call_args[0][0]
            assert mock_code.call_count == 2
            mock_code.assert_any_call("ollama pull phi4-mini", language="bash")
            mock_code.assert_any_call("ollama pull all-minilm:l6-v2", language="bash")

    @patch("documate_ai.ui.sidebar.check_ollama_health")
    def test_health_banner_recheck_button(
        self,
        mock_check: MagicMock,
        fake_settings: Settings,
    ) -> None:
        """Clicking Re-check Health triggers a fresh check and st.rerun()."""
        mock_check.return_value = HealthReport(server_reachable=True)

        with patch("streamlit.button", return_value=True), patch("streamlit.rerun") as mock_rerun:
            render_health_banner(fake_settings)
            mock_rerun.assert_called_once()


# ---------------------------------------------------------------------------
# Document Uploader Tests
# ---------------------------------------------------------------------------


class TestDocumentUploader:
    def test_process_button_disabled_when_no_files(
        self,
        mock_engine: DocuMateEngine,
    ) -> None:
        """Process button is disabled when no files are uploaded."""
        health = HealthReport(server_reachable=True, missing_models=[])

        with patch("streamlit.file_uploader", return_value=[]), patch(
            "streamlit.button"
        ) as mock_button:
            render_document_uploader(mock_engine, health)
            mock_button.assert_called_once()
            assert mock_button.call_args[1]["disabled"] is True

    def test_process_button_disabled_when_health_unready(
        self,
        mock_engine: DocuMateEngine,
    ) -> None:
        """Process button is disabled when Ollama is unreachable or models missing."""
        unhealthy = HealthReport(server_reachable=False)
        files = [DummyUploadedFile("doc.pdf", b"%PDF-1.4")]

        with patch("streamlit.file_uploader", return_value=files), patch(
            "streamlit.button"
        ) as mock_button:
            render_document_uploader(mock_engine, unhealthy)
            mock_button.assert_called_once()
            assert mock_button.call_args[1]["disabled"] is True

    def test_process_button_ingests_and_updates_state(
        self,
        mock_engine: DocuMateEngine,
    ) -> None:
        """Clicking Process button ingests files, sets last_ingest_report, and clears chat."""
        health = HealthReport(server_reachable=True, missing_models=[])
        files = [DummyUploadedFile("doc1.pdf", b"%PDF-1.4 content")]

        mock_report = IngestReport(
            files=["doc1.pdf"],
            pages=2,
            chunks=4,
            skipped={"blank.pdf": "No text"},
            fingerprint="abc12345",
            elapsed_seconds=1.23,
        )
        mock_engine.ingest = MagicMock(return_value=mock_report)  # type: ignore[method-assign]

        status_mock = MagicMock()
        with patch("streamlit.file_uploader", return_value=files), patch(
            "streamlit.button", return_value=True
        ), patch("streamlit.status", return_value=status_mock), patch(
            "documate_ai.ui.sidebar.clear_chat"
        ) as mock_clear, patch(
            "streamlit.warning"
        ) as mock_warn:
            render_document_uploader(mock_engine, health)

            mock_engine.ingest.assert_called_once()
            mock_clear.assert_called_once()
            assert st.session_state["last_ingest_report"] is mock_report
            mock_warn.assert_called_once_with("Skipped `blank.pdf`: No text")

    def test_process_button_handles_ingestion_error(
        self,
        mock_engine: DocuMateEngine,
    ) -> None:
        """Ingestion error renders error message cleanly."""
        health = HealthReport(server_reachable=True, missing_models=[])
        files = [DummyUploadedFile("bad.pdf", b"corrupt")]

        mock_engine.ingest = MagicMock(  # type: ignore[method-assign]
            side_effect=DocumentProcessingError("Corrupted PDF document", filename="bad.pdf")
        )

        status_mock = MagicMock()
        with patch("streamlit.file_uploader", return_value=files), patch(
            "streamlit.button", return_value=True
        ), patch("streamlit.status", return_value=status_mock), patch(
            "streamlit.error"
        ) as mock_error:
            render_document_uploader(mock_engine, health)
            mock_error.assert_called_once()
            assert "Corrupted PDF document" in mock_error.call_args[0][0]


# ---------------------------------------------------------------------------
# Loaded Documents List Tests
# ---------------------------------------------------------------------------


class TestLoadedDocuments:
    def test_loaded_documents_hidden_when_not_ready(
        self,
        mock_engine: DocuMateEngine,
    ) -> None:
        """When engine is not ready, loaded documents section is not drawn."""
        assert mock_engine.is_ready is False
        with patch("streamlit.subheader") as mock_sub:
            render_loaded_documents(mock_engine)
            mock_sub.assert_not_called()

    def test_loaded_documents_rendered_when_ready(
        self,
        mock_engine: DocuMateEngine,
    ) -> None:
        """When engine is ready, displays document count and file names."""
        mock_engine.set_rag(MagicMock(), loaded_documents=["doc_a.pdf", "doc_b.pdf"])
        assert mock_engine.is_ready is True

        st.session_state["last_ingest_report"] = IngestReport(
            files=["doc_a.pdf", "doc_b.pdf"],
            pages=5,
            chunks=10,
        )

        with patch("streamlit.subheader") as mock_sub, patch("streamlit.markdown") as mock_md:
            render_loaded_documents(mock_engine)
            mock_sub.assert_called_once_with("📚 Loaded Documents")
            assert mock_md.call_count == 2


# ---------------------------------------------------------------------------
# Cache Settings Tests
# ---------------------------------------------------------------------------


class TestCacheSettings:
    def test_cache_settings_toggle_and_slider(
        self,
        mock_engine: DocuMateEngine,
        fake_settings: Settings,
    ) -> None:
        """Renders toggle, slider using settings bounds, and cache count."""
        with patch("streamlit.toggle", return_value=True) as mock_toggle, patch(
            "streamlit.slider", return_value=0.90
        ) as mock_slider, patch("streamlit.caption") as mock_caption:
            render_cache_settings(mock_engine, fake_settings)

            mock_toggle.assert_called_once()
            mock_slider.assert_called_once_with(
                "Similarity threshold",
                min_value=fake_settings.similarity_slider_min,
                max_value=fake_settings.similarity_slider_max,
                value=fake_settings.similarity_threshold,
                step=fake_settings.similarity_slider_step,
                help="Higher = stricter matching. Lower = more cache hits.",
            )
            assert st.session_state["cache_enabled"] is True
            assert st.session_state["similarity_threshold"] == 0.90
            mock_caption.assert_called_once()

    def test_clear_cache_button(
        self,
        mock_engine: DocuMateEngine,
        fake_settings: Settings,
    ) -> None:
        """Clicking Clear Cache calls engine.clear_cache() and st.rerun()."""
        mock_engine.clear_cache = MagicMock(return_value=3)  # type: ignore[method-assign]

        with patch("streamlit.button", return_value=True), patch(
            "streamlit.success"
        ) as mock_success, patch("streamlit.rerun") as mock_rerun:
            render_cache_settings(mock_engine, fake_settings)
            mock_engine.clear_cache.assert_called_once_with(all_documents=False)
            mock_success.assert_called_once()
            assert "Cleared 3" in mock_success.call_args[0][0]
            mock_rerun.assert_called_once()


# ---------------------------------------------------------------------------
# Action Buttons & Debug Toggle Tests
# ---------------------------------------------------------------------------


class TestActionButtonsAndDebug:
    def test_clear_chat_button(self, mock_engine: DocuMateEngine) -> None:
        """Clicking Clear Chat calls clear_chat() and st.rerun()."""
        mock_engine.set_rag(MagicMock())  # make engine ready
        assert mock_engine.is_ready is True

        # Simulate button 1 (Clear Chat) clicked
        with patch("streamlit.button", side_effect=[True, False]), patch(
            "documate_ai.ui.sidebar.clear_chat"
        ) as mock_clear, patch("streamlit.rerun") as mock_rerun:
            render_action_buttons(mock_engine)
            mock_clear.assert_called_once()
            mock_rerun.assert_called_once()

    def test_reset_all_button(self, mock_engine: DocuMateEngine) -> None:
        """Clicking Reset All calls reset_all(delete_store=True) and st.rerun()."""
        st.session_state["last_ingest_report"] = IngestReport()

        # Simulate button 2 (Reset All) clicked
        with patch("streamlit.button", side_effect=[False, True]), patch(
            "documate_ai.ui.sidebar.reset_all"
        ) as mock_reset, patch("streamlit.rerun") as mock_rerun:
            render_action_buttons(mock_engine)
            mock_reset.assert_called_once_with(delete_store=True)
            assert "last_ingest_report" not in st.session_state
            mock_rerun.assert_called_once()

    def test_debug_toggle(self) -> None:
        """Checkbox sets st.session_state['debug_mode']."""
        with patch("streamlit.checkbox", return_value=True):
            render_debug_toggle()
            assert st.session_state["debug_mode"] is True


# ---------------------------------------------------------------------------
# Full Sidebar Layout Test
# ---------------------------------------------------------------------------


class TestRenderSidebarLayout:
    @patch("documate_ai.ui.sidebar.check_ollama_health")
    def test_render_sidebar_coordination(
        self,
        mock_check: MagicMock,
        mock_engine: DocuMateEngine,
        fake_settings: Settings,
    ) -> None:
        """render_sidebar coordinates all sub-components without error."""
        mock_check.return_value = HealthReport(server_reachable=True)

        render_sidebar(engine=mock_engine, settings=fake_settings)

        assert "health_report" in st.session_state
        assert "cache_enabled" in st.session_state
        assert "debug_mode" in st.session_state
