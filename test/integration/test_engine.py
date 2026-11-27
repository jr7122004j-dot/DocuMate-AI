"""Integration tests for DocuMateEngine service layer (T15).

Tests end-to-end execution across ingestion, semantic caching, multi-turn QA,
multi-question queries, streaming TokenSinks, and state reset without Streamlit or Ollama:
    1. EngineNotReadyError before document ingestion.
    2. Document ingestion and IngestReport generation.
    3. Cache miss -> Exact cache hit -> Semantic cache hit.
    4. Multi-question compound query flow.
    5. Live token streaming through TokenSink with idempotent finalization.
    6. Conversation history clearing and persistence.
    7. Reset and vector store deletion.
"""

from __future__ import annotations

import os

# Ensure telemetry is disabled before any chromadb import
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from documate_ai.cache_manager import CacheManager
from documate_ai.config import Settings
from documate_ai.engine import DocuMateEngine
from documate_ai.errors import DocumentProcessingError, EngineNotReadyError
from documate_ai.schemas import ChatResponse, IngestReport, UploadedPdf
from documate_ai.stream_handler import CollectingSink, StreamlitTokenSink
from test.conftest import ControllableEmbeddings, FakeContainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=384)


@pytest.fixture
def temp_settings(tmp_path: Path) -> Settings:
    """Create isolated settings with data_dir pointing to a temporary path."""
    return Settings(
        data_dir=tmp_path,
        chunk_size=300,
        chunk_overlap=50,
        similarity_threshold=0.85,
    )


# ---------------------------------------------------------------------------
# Engine Readiness & Ingestion Tests
# ---------------------------------------------------------------------------


class TestEngineReadinessAndIngest:
    def test_ask_before_ingest_raises_not_ready_error(
        self,
        fake_embeddings: DeterministicFakeEmbedding,
        temp_settings: Settings,
    ) -> None:
        """Calling ask() on an uninitialized engine raises EngineNotReadyError."""
        llm = FakeListChatModel(responses=["Not used"])
        engine = DocuMateEngine(
            embeddings=fake_embeddings,
            llm=llm,
            settings=temp_settings,
        )

        assert engine.is_ready is False
        assert engine.loaded_documents == []
        assert engine.document_fingerprint is None

        with pytest.raises(EngineNotReadyError, match="Engine is not ready for queries"):
            engine.ask("What is the company revenue?")

    def test_ingest_empty_files_raises_processing_error(
        self,
        fake_embeddings: DeterministicFakeEmbedding,
        temp_settings: Settings,
    ) -> None:
        """Ingesting an empty list of files raises DocumentProcessingError."""
        llm = FakeListChatModel(responses=["Not used"])
        engine = DocuMateEngine(
            embeddings=fake_embeddings,
            llm=llm,
            settings=temp_settings,
        )

        with pytest.raises(DocumentProcessingError, match="No PDF files provided"):
            engine.ingest([])

    def test_ingest_valid_pdf_builds_engine(
        self,
        sample_pdf_bytes: bytes,
        fake_embeddings: DeterministicFakeEmbedding,
        temp_settings: Settings,
    ) -> None:
        """Ingesting a valid PDF produces IngestReport and transitions engine to ready state."""
        llm = FakeListChatModel(responses=["Not used"])
        engine = DocuMateEngine(
            embeddings=fake_embeddings,
            llm=llm,
            settings=temp_settings,
        )

        uploaded = UploadedPdf(name="sample.pdf", data=sample_pdf_bytes)
        report: IngestReport = engine.ingest([uploaded])

        assert isinstance(report, IngestReport)
        assert report.files == ["sample.pdf"]
        assert report.pages == 3
        assert report.chunks > 0
        assert report.fingerprint != ""
        assert report.elapsed_seconds >= 0.0

        assert engine.is_ready is True
        assert engine.loaded_documents == ["sample.pdf"]
        assert engine.document_fingerprint == report.fingerprint


# ---------------------------------------------------------------------------
# End-to-End Q&A Flow (Cache Miss -> Exact Hit -> Semantic Hit)
# ---------------------------------------------------------------------------


class TestEngineQAAndCaching:
    def test_qa_flow_cache_miss_then_exact_hit(
        self,
        sample_pdf_bytes: bytes,
        fake_embeddings: DeterministicFakeEmbedding,
        temp_settings: Settings,
    ) -> None:
        """A new question causes a cache miss, then the same question produces an exact hit."""
        # 1st call uses response 1; 2nd call should hit cache and NOT call LLM
        llm = FakeListChatModel(responses=["Acme generated $50 million in 2023."])
        engine = DocuMateEngine(
            embeddings=fake_embeddings,
            llm=llm,
            settings=temp_settings,
        )

        engine.ingest([UploadedPdf(name="sample.pdf", data=sample_pdf_bytes)])

        # Turn 1: Cache Miss
        question = "What was the revenue of Acme?"
        res1 = engine.ask(question)

        assert isinstance(res1, ChatResponse)
        assert res1.from_cache is False
        assert "Acme generated $50 million in 2023." in res1.answer
        assert len(res1.citations) > 0
        assert "generation_ms" in res1.timings
        assert "total_ms" in res1.timings
        assert len(engine.history) == 2

        # Turn 2: Exact Cache Hit (same question)
        # FakeListChatModel has no more responses; if LLM was called, an error would be raised
        res2 = engine.ask(question)

        assert isinstance(res2, ChatResponse)
        assert res2.from_cache is True
        assert res2.similarity == 1.0
        assert res2.matched_question == question
        assert "Acme generated $50 million in 2023." in res2.answer
        assert len(res2.citations) > 0

        # Cache hit must also append the turn to history so follow-ups work
        assert len(engine.history) == 4
        msgs = engine.history_messages
        assert msgs[2].content == question
        assert msgs[3].content == res2.answer

    def test_semantic_cache_hit_above_threshold(
        self,
        sample_pdf_bytes: bytes,
        temp_settings: Settings,
    ) -> None:
        """Semantically similar question above threshold returns a cached answer."""
        dim = 384
        vec_q1 = [1.0] + [0.0] * (dim - 1)
        # Vector with cosine similarity ~ 0.95 with vec_q1
        vec_q2 = [0.95, 0.3122498999] + [0.0] * (dim - 2)

        controllable_emb = ControllableEmbeddings(
            default_vector=vec_q1,
            size=dim,
        )
        controllable_emb.register("what is the company revenue", vec_q1)
        controllable_emb.register("tell me the company revenue", vec_q2)

        # Only 1 LLM response provided
        llm = FakeListChatModel(responses=["Revenue was $50 million."])
        engine = DocuMateEngine(
            embeddings=controllable_emb,
            llm=llm,
            settings=temp_settings,
        )

        engine.ingest([UploadedPdf(name="sample.pdf", data=sample_pdf_bytes)])

        # Question 1: Cache Miss
        res1 = engine.ask("What is the company revenue?")
        assert res1.from_cache is False

        # Question 2: Semantic Hit (similarity ~ 0.95 >= threshold 0.85)
        res2 = engine.ask("Tell me the company revenue?", threshold=0.85)
        assert res2.from_cache is True
        assert res2.similarity is not None
        assert res2.similarity >= 0.85
        assert "Revenue was $50 million." in res2.answer

    def test_use_cache_false_bypasses_cache(
        self,
        sample_pdf_bytes: bytes,
        fake_embeddings: DeterministicFakeEmbedding,
        temp_settings: Settings,
    ) -> None:
        """When use_cache=False, lookup is skipped and LLM is invoked."""
        llm = FakeListChatModel(
            responses=[
                "First answer.",
                "Standalone question 2.",
                "Second answer from LLM.",
            ]
        )
        engine = DocuMateEngine(
            embeddings=fake_embeddings,
            llm=llm,
            settings=temp_settings,
        )
        engine.ingest([UploadedPdf(name="sample.pdf", data=sample_pdf_bytes)])

        res1 = engine.ask("What is revenue?", use_cache=False)
        assert res1.from_cache is False

        res2 = engine.ask("What is revenue?", use_cache=False)
        assert res2.from_cache is False
        assert "Second answer from LLM." in res2.answer


# ---------------------------------------------------------------------------
# Multi-Question Flow Tests
# ---------------------------------------------------------------------------


class TestEngineMultiQuestionFlow:
    def test_multi_question_processed_and_history_updated_once(
        self,
        sample_pdf_bytes: bytes,
        fake_embeddings: DeterministicFakeEmbedding,
        temp_settings: Settings,
    ) -> None:
        """Compound multi-question prompt answers each question and adds 1 turn to history."""
        llm = FakeListChatModel(
            responses=[
                "Revenue: $50 million.",
                "Profit: $8 million.",
            ]
        )
        engine = DocuMateEngine(
            embeddings=fake_embeddings,
            llm=llm,
            settings=temp_settings,
        )
        engine.ingest([UploadedPdf(name="sample.pdf", data=sample_pdf_bytes)])

        prompt = "What was the revenue? And what was the net profit?"
        res = engine.ask(prompt)

        assert isinstance(res, ChatResponse)
        assert res.from_cache is False
        assert len(res.sub_questions) == 2
        assert "### What was the revenue?" in res.answer
        assert "### And what was the net profit?" in res.answer
        assert len(res.citations) > 0

        # History must contain exactly 1 turn (2 messages) for the whole compound query
        assert len(engine.history) == 2
        assert engine.history_messages[0].content == prompt
        assert engine.history_messages[1].content == res.answer


# ---------------------------------------------------------------------------
# Streaming Integration with TokenSink Tests
# ---------------------------------------------------------------------------


class TestEngineStreamingIntegration:
    def test_streaming_with_collecting_sink(
        self,
        sample_pdf_bytes: bytes,
        fake_embeddings: DeterministicFakeEmbedding,
        temp_settings: Settings,
    ) -> None:
        """Answer tokens are streamed to sink and sink is finalized by the engine."""
        llm = FakeListChatModel(responses=["Streamed answer text."])
        engine = DocuMateEngine(
            embeddings=fake_embeddings,
            llm=llm,
            settings=temp_settings,
        )
        engine.ingest([UploadedPdf(name="sample.pdf", data=sample_pdf_bytes)])

        sink = CollectingSink()
        res = engine.ask("What is the revenue?", sink=sink)

        assert sink.is_finalized is True
        assert "Streamed answer text." in sink.text
        assert sink.text == res.answer

    def test_streaming_with_streamlit_token_sink(
        self,
        sample_pdf_bytes: bytes,
        fake_embeddings: DeterministicFakeEmbedding,
        fake_container: FakeContainer,
        temp_settings: Settings,
    ) -> None:
        """StreamlitTokenSink receives markdown updates and is finalized with clean text."""
        llm = FakeListChatModel(responses=["Streamed to fake container."])
        engine = DocuMateEngine(
            embeddings=fake_embeddings,
            llm=llm,
            settings=temp_settings,
        )
        engine.ingest([UploadedPdf(name="sample.pdf", data=sample_pdf_bytes)])

        sink = StreamlitTokenSink(container=fake_container, render_interval=0.0)
        res = engine.ask("What is the revenue?", sink=sink)

        assert sink.is_finalized is True
        assert len(fake_container.markdown_calls) > 0
        assert fake_container.markdown_calls[-1] == res.answer

    def test_cached_answer_streams_to_sink(
        self,
        sample_pdf_bytes: bytes,
        fake_embeddings: DeterministicFakeEmbedding,
        temp_settings: Settings,
    ) -> None:
        """A cache hit properly streams into the sink and finalizes."""
        llm = FakeListChatModel(responses=["Answer once."])
        engine = DocuMateEngine(
            embeddings=fake_embeddings,
            llm=llm,
            settings=temp_settings,
        )
        engine.ingest([UploadedPdf(name="sample.pdf", data=sample_pdf_bytes)])

        # Miss
        engine.ask("What is revenue?")

        # Hit with sink
        sink = CollectingSink()
        res = engine.ask("What is revenue?", sink=sink)

        assert res.from_cache is True
        assert sink.is_finalized is True
        assert "Answer once." in sink.text


# ---------------------------------------------------------------------------
# History, Cache & Reset Management Tests
# ---------------------------------------------------------------------------


class TestEngineLifecycleManagement:
    def test_clear_history(
        self,
        sample_pdf_bytes: bytes,
        fake_embeddings: DeterministicFakeEmbedding,
        temp_settings: Settings,
    ) -> None:
        """clear_history() empties chat history."""
        llm = FakeListChatModel(responses=["Answer."])
        engine = DocuMateEngine(
            embeddings=fake_embeddings,
            llm=llm,
            settings=temp_settings,
        )
        engine.ingest([UploadedPdf(name="sample.pdf", data=sample_pdf_bytes)])
        engine.ask("Question?")
        assert len(engine.history) == 2

        engine.clear_history()
        assert len(engine.history) == 0
        assert len(engine.history_messages) == 0

    def test_clear_cache(
        self,
        sample_pdf_bytes: bytes,
        fake_embeddings: DeterministicFakeEmbedding,
        temp_settings: Settings,
    ) -> None:
        """clear_cache() removes cache entries."""
        llm = FakeListChatModel(responses=["Answer."])
        engine = DocuMateEngine(
            embeddings=fake_embeddings,
            llm=llm,
            settings=temp_settings,
        )
        engine.ingest([UploadedPdf(name="sample.pdf", data=sample_pdf_bytes)])
        engine.ask("Question?")

        cleared = engine.clear_cache()
        assert cleared >= 1

    def test_reset_engine_state(
        self,
        sample_pdf_bytes: bytes,
        fake_embeddings: DeterministicFakeEmbedding,
        temp_settings: Settings,
    ) -> None:
        """reset(delete_store=True) unloads engine and purges vector store."""
        llm = FakeListChatModel(responses=["Answer."])
        engine = DocuMateEngine(
            embeddings=fake_embeddings,
            llm=llm,
            settings=temp_settings,
        )
        engine.ingest([UploadedPdf(name="sample.pdf", data=sample_pdf_bytes)])
        assert engine.is_ready is True

        engine.reset(delete_store=True)
        assert engine.is_ready is False
        assert engine.loaded_documents == []
        assert engine.document_fingerprint is None
        assert len(engine.history) == 0

        with pytest.raises(EngineNotReadyError):
            engine.ask("Question after reset?")
