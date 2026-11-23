"""Integration tests for the vector store lifecycle module.

These tests use a temporary Chroma directory and fake embeddings so they
run offline without an Ollama server.
"""

from __future__ import annotations

import hashlib
import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from documate_ai.config import Settings
from documate_ai.vector_store import (
    _collection_name,
    build_retriever,
    delete_vector_store,
    get_or_create_vector_store,
    load_vector_store,
    reset_all_vector_stores,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_FINGERPRINT = "a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890"
ALT_FINGERPRINT = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"


def _make_settings(chroma_dir: str) -> Settings:
    """Create a Settings instance pointing at the given chroma directory."""
    from pathlib import Path

    return Settings(chroma_dir=Path(chroma_dir))


def _make_chunks(
    n: int = 3,
    file_hash: str = "abc123",
    source: str = "test.pdf",
) -> list[Document]:
    """Create *n* dummy chunks with deterministic chunk IDs."""
    chunks: list[Document] = []
    for i in range(n):
        chunk_id = hashlib.sha256(f"{file_hash}:1:{i}".encode()).hexdigest()
        chunks.append(
            Document(
                page_content=f"Chunk number {i} with some text for embedding.",
                metadata={
                    "source": source,
                    "page": 1,
                    "file_hash": file_hash,
                    "chunk_index": i,
                    "chunk_id": chunk_id,
                },
            )
        )
    return chunks


@pytest.fixture
def embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=384)


@pytest.fixture
def settings(tmp_chroma_dir) -> Settings:
    return _make_settings(str(tmp_chroma_dir))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCollectionName:
    def test_prefix_and_length(self):
        name = _collection_name(FAKE_FINGERPRINT)
        assert name.startswith("docs_")
        assert name == f"docs_{FAKE_FINGERPRINT[:16]}"

    def test_different_fingerprints_differ(self):
        assert _collection_name(FAKE_FINGERPRINT) != _collection_name(ALT_FINGERPRINT)


class TestGetOrCreateVectorStore:
    def test_creates_and_populates(self, embeddings, settings):
        chunks = _make_chunks(3)
        vs = get_or_create_vector_store(
            chunks, embeddings, settings=settings, fingerprint=FAKE_FINGERPRINT
        )
        col = vs._collection
        assert col.count() == 3

    def test_idempotent_add(self, embeddings, settings):
        """Adding the same chunks twice must not duplicate documents."""
        chunks = _make_chunks(3)
        vs1 = get_or_create_vector_store(
            chunks, embeddings, settings=settings, fingerprint=FAKE_FINGERPRINT
        )
        assert vs1._collection.count() == 3

        vs2 = get_or_create_vector_store(
            chunks, embeddings, settings=settings, fingerprint=FAKE_FINGERPRINT
        )
        assert vs2._collection.count() == 3

    def test_different_fingerprints_separate_collections(self, embeddings, settings):
        chunks_a = _make_chunks(2, file_hash="aaa", source="a.pdf")
        chunks_b = _make_chunks(4, file_hash="bbb", source="b.pdf")

        vs_a = get_or_create_vector_store(
            chunks_a, embeddings, settings=settings, fingerprint=FAKE_FINGERPRINT
        )
        vs_b = get_or_create_vector_store(
            chunks_b, embeddings, settings=settings, fingerprint=ALT_FINGERPRINT
        )

        assert vs_a._collection.count() == 2
        assert vs_b._collection.count() == 4


class TestLoadVectorStore:
    def test_load_existing(self, embeddings, settings):
        chunks = _make_chunks(3)
        get_or_create_vector_store(
            chunks, embeddings, settings=settings, fingerprint=FAKE_FINGERPRINT
        )

        loaded = load_vector_store(FAKE_FINGERPRINT, embeddings, settings=settings)
        assert loaded._collection.count() == 3

    def test_load_missing_raises(self, embeddings, settings):
        with pytest.raises(ValueError, match="does not exist"):
            load_vector_store("nonexistent_fp", embeddings, settings=settings)


class TestDeleteVectorStore:
    def test_delete_existing(self, embeddings, settings):
        chunks = _make_chunks(2)
        get_or_create_vector_store(
            chunks, embeddings, settings=settings, fingerprint=FAKE_FINGERPRINT
        )

        delete_vector_store(FAKE_FINGERPRINT, settings=settings)

        with pytest.raises(ValueError, match="does not exist"):
            load_vector_store(FAKE_FINGERPRINT, embeddings, settings=settings)

    def test_delete_nonexistent_no_error(self, settings):
        # Should not raise
        delete_vector_store("does_not_exist_fp", settings=settings)


class TestResetAllVectorStores:
    def test_reset_clears_all(self, embeddings, settings):
        chunks_a = _make_chunks(2, file_hash="aaa")
        chunks_b = _make_chunks(2, file_hash="bbb")
        get_or_create_vector_store(
            chunks_a, embeddings, settings=settings, fingerprint=FAKE_FINGERPRINT
        )
        get_or_create_vector_store(
            chunks_b, embeddings, settings=settings, fingerprint=ALT_FINGERPRINT
        )

        deleted = reset_all_vector_stores(settings=settings)
        assert deleted == 2

        with pytest.raises(ValueError, match="does not exist"):
            load_vector_store(FAKE_FINGERPRINT, embeddings, settings=settings)


class TestBuildRetriever:
    def test_retriever_k(self, embeddings, settings):
        chunks = _make_chunks(5)
        vs = get_or_create_vector_store(
            chunks, embeddings, settings=settings, fingerprint=FAKE_FINGERPRINT
        )

        retriever = build_retriever(vs, settings=settings)
        assert retriever.search_kwargs["k"] == settings.retriever_result_count

    def test_retriever_mmr(self, embeddings, settings):
        chunks = _make_chunks(5)
        vs = get_or_create_vector_store(
            chunks, embeddings, settings=settings, fingerprint=FAKE_FINGERPRINT
        )

        retriever = build_retriever(vs, settings=settings, search_type="mmr")
        assert retriever.search_type == "mmr"
