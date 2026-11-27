"""Streamlit-free application service layer for document QA.

Encapsulates document ingestion, Chroma vector store lifecycle,
semantic caching, multi-turn conversational retrieval, and token streaming
into a decoupled, UI-agnostic engine.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from documate_ai.cache_manager import CacheLookupResult, CacheManager
from documate_ai.chat_history import ChatHistory
from documate_ai.config import Settings, get_settings
from documate_ai.conversation import ConversationalRAG, process_multi_question_query
from documate_ai.document_processor import process_pdf_bytes
from documate_ai.errors import DocumentProcessingError, EngineNotReadyError
from documate_ai.schemas import ChatResponse, Citation, IngestReport, UploadedPdf
from documate_ai.stream_handler import TokenSink
from documate_ai.vector_store import (
    build_retriever,
    delete_vector_store,
    get_or_create_vector_store,
)

if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from langchain_core.messages import BaseMessage
    from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger(__name__)


class DocuMateEngine:
    """Streamlit-free application service layer for document QA.

    Coordinates document ingestion, vector stores, caching, and conversational
    RAG without any Streamlit dependencies.

    Attributes:
        settings: Application settings.
        embeddings: Embeddings model for vectorisation.
        llm: Chat language model for QA and condensation.
        cache_manager: Semantic cache manager instance.
        history: Multi-turn chat history tracker.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        llm: BaseChatModel,
        cache_manager: CacheManager | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialise DocuMateEngine.

        Args:
            embeddings: Embedding model for document and query vectorisation.
            llm: Chat language model for condensing and QA generation.
            cache_manager: Optional CacheManager instance. If None, initialized
                using application settings.
            settings: Application settings. If None, defaults are loaded.
        """
        self.settings = settings if settings is not None else get_settings()
        self.embeddings = embeddings
        self.llm = llm
        self.cache_manager = (
            cache_manager
            if cache_manager is not None
            else CacheManager(settings=self.settings)
        )

        self.history = ChatHistory(settings=self.settings)
        self._lock = threading.RLock()

        self._document_fingerprint: str | None = None
        self._loaded_documents: list[str] = []
        self._vector_store: Chroma | None = None
        self._retriever: VectorStoreRetriever | None = None
        self._rag: ConversationalRAG | None = None

    @property
    def is_ready(self) -> bool:
        """Return True if documents are ingested and the engine is ready for queries."""
        with self._lock:
            return self._rag is not None and self._document_fingerprint is not None

    @property
    def loaded_documents(self) -> list[str]:
        """Return a copy of the currently loaded document names."""
        with self._lock:
            return list(self._loaded_documents)

    @property
    def document_fingerprint(self) -> str | None:
        """Return the SHA-256 fingerprint of the currently loaded document set."""
        with self._lock:
            return self._document_fingerprint

    @property
    def history_messages(self) -> list[BaseMessage]:
        """Return a copy of the conversation history messages."""
        return self.history.get_messages()

    def set_rag(
        self,
        rag: ConversationalRAG,
        fingerprint: str = "test_fingerprint",
        loaded_documents: list[str] | None = None,
    ) -> None:
        """Explicitly set the ConversationalRAG pipeline (testing hook).

        Args:
            rag: ConversationalRAG engine instance.
            fingerprint: Document fingerprint to associate with the pipeline.
            loaded_documents: Optional list of document names.
        """
        with self._lock:
            self._rag = rag
            self._document_fingerprint = fingerprint
            self._loaded_documents = list(loaded_documents or ["document.pdf"])

    def ingest(
        self,
        files: list[UploadedPdf],
        *,
        reset_history: bool = True,
    ) -> IngestReport:
        """Ingest, validate, chunk, and embed uploaded PDF files into the vector store.

        Builds the retriever and ConversationalRAG chain, updates the loaded
        document fingerprint, and resets chat history by default.

        Args:
            files: List of UploadedPdf instances containing filename and PDF bytes.
            reset_history: Whether to clear chat history upon loading a new document set.

        Returns:
            IngestReport detailing ingested files, pages, chunks, and timing.

        Raises:
            DocumentProcessingError: If files is empty, unreadable, or contains no
                extractable text.
        """
        if not files:
            raise DocumentProcessingError("No PDF files provided for ingestion.")

        start_time = time.perf_counter()

        with self._lock:
            chunks, summary = process_pdf_bytes(files, self.settings)

            # Build or reuse vector store keyed by fingerprint
            vector_store = get_or_create_vector_store(
                chunks=chunks,
                embeddings=self.embeddings,
                settings=self.settings,
                fingerprint=summary.document_fingerprint,
            )

            retriever = build_retriever(vector_store, self.settings)
            rag = ConversationalRAG(
                retriever=retriever,
                llm=self.llm,
                settings=self.settings,
            )

            self._vector_store = vector_store
            self._retriever = retriever
            self._rag = rag
            self._document_fingerprint = summary.document_fingerprint
            self._loaded_documents = list(summary.processed_files)

            if reset_history:
                self.history.clear()

            elapsed = time.perf_counter() - start_time
            logger.info(
                "Ingestion completed: %d files (%d pages, %d chunks) in %.2fs",
                len(summary.processed_files),
                summary.total_pages,
                summary.total_chunks,
                elapsed,
            )

            return IngestReport(
                files=summary.processed_files,
                pages=summary.total_pages,
                chunks=summary.total_chunks,
                skipped=summary.skipped_files,
                fingerprint=summary.document_fingerprint,
                elapsed_seconds=round(elapsed, 4),
            )

    def ask(
        self,
        question: str,
        *,
        sink: TokenSink | None = None,
        use_cache: bool = True,
        threshold: float | None = None,
    ) -> ChatResponse:
        """Answer a user question against the ingested document set.

        Workflow:
            1. Validate engine readiness (raises EngineNotReadyError if not ready).
            2. Check semantic cache (if use_cache is True). On cache hit:
               stream answer to sink, update history, and return ChatResponse.
            3. On cache miss: execute conversational RAG pipeline (supporting
               single- or multi-question queries via LCEL and TokenSink streaming).
            4. Save new answer and deduplicated citations to cache.
            5. Finalize sink with formatted answer.
            6. Return structured ChatResponse with execution timings and metadata.

        Args:
            question: The user prompt (single or compound multi-question query).
            sink: Optional TokenSink receiving streamed answer tokens.
            use_cache: Whether to check and write to the semantic cache.
            threshold: Optional similarity threshold override for this query.

        Returns:
            ChatResponse containing answer, citations, cache status, and timings.

        Raises:
            EngineNotReadyError: If called before documents are ingested.
        """
        with self._lock:
            if not self.is_ready or self._rag is None or self._document_fingerprint is None:
                raise EngineNotReadyError(
                    "Engine is not ready for queries. Ingest PDF documents first via ingest()."
                )
            rag = self._rag
            fingerprint = self._document_fingerprint
            doc_names = list(self._loaded_documents)

        total_start = time.perf_counter()
        timings: dict[str, float] = {
            "cache_ms": 0.0,
            "retrieval_ms": 0.0,
            "generation_ms": 0.0,
            "total_ms": 0.0,
        }

        lookup_result: CacheLookupResult | None = None

        # ------------------------------------------------------------------
        # 1. Cache lookup
        # ------------------------------------------------------------------
        if use_cache and self.cache_manager is not None:
            cache_start = time.perf_counter()
            lookup_result = self.cache_manager.lookup(
                question=question,
                document_fingerprint=fingerprint,
                embeddings_model=self.embeddings,
                threshold=threshold,
            )
            timings["cache_ms"] = round(
                (time.perf_counter() - cache_start) * 1000.0, 2
            )

            if lookup_result.entry is not None:
                cached_entry = lookup_result.entry
                answer = cached_entry["answer"]

                # Reconstruct Citation objects from cached dicts
                citations = [
                    Citation(
                        source=str(c.get("source", "Unknown Document")),
                        page=int(c.get("page", 1)),
                        snippet=str(c.get("snippet", c.get("content", ""))),
                    )
                    for c in cached_entry.get("citations", [])
                ]

                # If a sink is provided, stream the cached answer and finalize
                if sink is not None:
                    sink.on_token(answer)
                    sink.finalize(answer)

                # Add turn to chat history so follow-ups work seamlessly
                self.history.add_turn(question, answer)

                timings["total_ms"] = round(
                    (time.perf_counter() - total_start) * 1000.0, 2
                )
                logger.info(
                    "Cache hit (similarity=%.3f) for question: %s",
                    lookup_result.similarity,
                    question[:60],
                )
                return ChatResponse(
                    answer=answer,
                    citations=citations,
                    from_cache=True,
                    similarity=lookup_result.similarity,
                    matched_question=cached_entry.get("question"),
                    standalone_question=None,
                    sub_questions=[],
                    timings=timings,
                )

        # ------------------------------------------------------------------
        # 2. RAG execution (Single or Multi-Question)
        # ------------------------------------------------------------------
        gen_start = time.perf_counter()

        result = process_multi_question_query(
            chain=rag,
            prompt=question,
            history=self.history,
            sink=sink,
            settings=self.settings,
        )

        # Finalize sink with complete post-processed answer (idempotent)
        if sink is not None:
            sink.finalize(result["answer"])

        timings["generation_ms"] = round(
            (time.perf_counter() - gen_start) * 1000.0, 2
        )

        # ------------------------------------------------------------------
        # 3. Cache persistence
        # ------------------------------------------------------------------
        if use_cache and self.cache_manager is not None:
            citations_dicts = [c.to_dict() for c in result["citations"]]
            try:
                llm_name = str(
                    getattr(
                        self.llm,
                        "model",
                        getattr(self.llm, "model_name", self.settings.llm_model_name),
                    )
                )
                emb_name = str(
                    getattr(
                        self.embeddings,
                        "model",
                        getattr(
                            self.embeddings,
                            "model_name",
                            self.settings.embedding_model_name,
                        ),
                    )
                )
                self.cache_manager.save(
                    question=question,
                    answer=result["answer"],
                    citations=citations_dicts,
                    document_fingerprint=fingerprint,
                    document_names=doc_names,
                    embeddings_model=self.embeddings,
                    llm_model=llm_name,
                    embedding_model=emb_name,
                    query_embedding=(
                        lookup_result.query_embedding if lookup_result else None
                    ),
                )
            except Exception as exc:
                logger.warning("Failed to save answer to cache: %s", exc)

        timings["total_ms"] = round(
            (time.perf_counter() - total_start) * 1000.0, 2
        )

        return ChatResponse(
            answer=result["answer"],
            citations=result["citations"],
            from_cache=False,
            similarity=lookup_result.similarity if lookup_result else None,
            matched_question=None,
            standalone_question=result.get("standalone_question"),
            sub_questions=result.get("sub_questions", []),
            timings=timings,
        )

    def clear_history(self) -> None:
        """Clear conversation chat history."""
        self.history.clear()
        logger.info("Chat history cleared.")

    def clear_cache(self, *, all_documents: bool = False) -> int:
        """Clear cached answers.

        Args:
            all_documents: If True, clears all cached answers. If False,
                clears only answers associated with the current document set.

        Returns:
            Number of cleared cache entries.
        """
        if self.cache_manager is None:
            return 0

        target_fp = None if all_documents else self._document_fingerprint
        cleared = self.cache_manager.clear(document_fingerprint=target_fp)
        logger.info(
            "Cleared %d cache entries (all_documents=%s).", cleared, all_documents
        )
        return cleared

    def reset(self, *, delete_store: bool = False) -> None:
        """Reset the engine state.

        Args:
            delete_store: Whether to also delete the underlying Chroma
                collection from disk.
        """
        with self._lock:
            self.clear_history()
            if delete_store and self._document_fingerprint:
                try:
                    delete_vector_store(self._document_fingerprint, self.settings)
                except Exception as exc:
                    logger.warning("Failed to delete vector store on reset: %s", exc)

            self._vector_store = None
            self._retriever = None
            self._rag = None
            self._document_fingerprint = None
            self._loaded_documents = []
            logger.info("DocuMateEngine reset (delete_store=%s).", delete_store)
