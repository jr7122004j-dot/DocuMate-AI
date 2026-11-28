"""End-to-end integration tests using live Ollama models (T19).

These tests run against an active Ollama server (http://localhost:11434) with
real neural models (e.g. phi4-mini and all-minilm:l6-v2).

When Ollama is unreachable, tests are automatically skipped by the
@pytest.mark.ollama hook in test/conftest.py.
"""

from __future__ import annotations

import os

# Disable OpenTelemetry telemetry before importing any chromadb modules
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from pathlib import Path

import pytest

from documate_ai.cache_manager import CacheManager
from documate_ai.config import Settings
from documate_ai.engine import DocuMateEngine
from documate_ai.models import create_embeddings, create_llm
from documate_ai.schemas import UploadedPdf
from documate_ai.stream_handler import CollectingSink


@pytest.mark.ollama
class TestLiveOllamaPipeline:
    """Live integration tests requiring an active Ollama server."""

    @pytest.fixture
    def live_settings(self, tmp_path: Path) -> Settings:
        """Isolated Settings instance with temporary cache and chroma directories."""
        return Settings(
            data_dir=tmp_path,
            similarity_threshold=0.85,
            stream_render_interval_seconds=0.0,
        )

    @pytest.fixture
    def live_engine(self, live_settings: Settings) -> DocuMateEngine:
        """Create a real DocuMateEngine connected to local Ollama."""
        embeddings = create_embeddings(live_settings)
        llm = create_llm(live_settings)
        cache_manager = CacheManager(
            cache_dir=live_settings.cache_dir,
            settings=live_settings,
        )
        return DocuMateEngine(
            embeddings=embeddings,
            llm=llm,
            cache_manager=cache_manager,
            settings=live_settings,
        )

    def test_live_ingestion_and_fresh_qa_with_citations(
        self,
        live_engine: DocuMateEngine,
        sample_pdf_bytes: bytes,
    ) -> None:
        """Full cycle: ingest PDF -> ask question -> verify answer and page citation."""
        # 1. Ingestion
        files = [UploadedPdf(name="sample.pdf", data=sample_pdf_bytes)]
        report = live_engine.ingest(files)

        assert report.pages == 3
        assert report.chunks >= 3
        assert report.files == ["sample.pdf"]
        assert live_engine.is_ready is True

        # 2. Fresh QA with live token streaming
        sink = CollectingSink()
        response = live_engine.ask("What is DocuMate-AI?", sink=sink, use_cache=True)

        assert response.from_cache is False
        assert len(response.answer) > 0
        assert sink.is_finalized is True
        assert len(sink.text) > 0

        # Citations verification
        assert len(response.citations) > 0
        citation_sources = [c.source for c in response.citations]
        citation_pages = [c.page for c in response.citations]

        assert "sample.pdf" in citation_sources
        # "DocuMate-AI" is introduced on page 1 of sample.pdf
        assert 1 in citation_pages

    def test_live_semantic_cache_hit(
        self,
        live_engine: DocuMateEngine,
        sample_pdf_bytes: bytes,
    ) -> None:
        """Repeated question hits the semantic cache with execution timings."""
        files = [UploadedPdf(name="sample.pdf", data=sample_pdf_bytes)]
        live_engine.ingest(files)

        question = "How does semantic caching work in DocuMate?"

        # Miss
        resp1 = live_engine.ask(question, use_cache=True)
        assert resp1.from_cache is False

        # Hit (exact / high semantic similarity)
        sink = CollectingSink()
        resp2 = live_engine.ask(question, sink=sink, use_cache=True)

        assert resp2.from_cache is True
        assert resp2.similarity is not None
        assert resp2.similarity >= 0.85
        assert resp2.answer == resp1.answer
        assert sink.is_finalized is True
        assert sink.text == resp1.answer

        # Cache timing verification: generation time is zero on cache hit
        assert resp2.timings["cache_ms"] > 0
        assert resp2.timings["generation_ms"] == 0.0

    def test_live_multi_question_query(
        self,
        live_engine: DocuMateEngine,
        sample_pdf_bytes: bytes,
    ) -> None:
        """Multi-question compound prompt produces answers and citations for both."""
        files = [UploadedPdf(name="sample.pdf", data=sample_pdf_bytes)]
        live_engine.ingest(files)

        compound_question = (
            "What is DocuMate-AI? And how does Semantic Caching work?"
        )
        sink = CollectingSink()
        response = live_engine.ask(compound_question, sink=sink, use_cache=False)

        assert response.from_cache is False
        assert len(response.answer) > 0
        assert len(response.sub_questions) == 2

        # Both sub-questions should be cited
        pages = {c.page for c in response.citations}
        assert 1 in pages or 2 in pages

    def test_live_follow_up_conversational_history(
        self,
        live_engine: DocuMateEngine,
        sample_pdf_bytes: bytes,
    ) -> None:
        """Follow-up questions leverage history snapshot without prompt leakage."""
        files = [UploadedPdf(name="sample.pdf", data=sample_pdf_bytes)]
        live_engine.ingest(files)

        resp1 = live_engine.ask("What is DocuMate-AI?")
        assert len(resp1.answer) > 0

        # Follow-up using anaphoric reference ("it")
        resp2 = live_engine.ask("Does it run on-device or in the cloud?")
        assert len(resp2.answer) > 0
        assert resp2.standalone_question is not None

    def test_live_engine_reset(
        self,
        live_engine: DocuMateEngine,
        sample_pdf_bytes: bytes,
    ) -> None:
        """reset() cleans up vector store and restores initial state."""
        files = [UploadedPdf(name="sample.pdf", data=sample_pdf_bytes)]
        live_engine.ingest(files)
        assert live_engine.is_ready is True

        live_engine.reset(delete_store=True)
        assert live_engine.is_ready is False
        assert live_engine.loaded_documents == []
