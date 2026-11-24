"""Unit tests for the semantic answer cache (cache_manager.py).

Tests SHA-256 keying, determinism, schema versioning, legacy purging,
Tier 1 exact match, Tier 2 semantic match, cross-document isolation,
runtime threshold changes, corrupt-file tolerance, embedding reuse,
thread safety, and oldest-first eviction.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from documate_ai.cache_manager import (
    CACHE_SCHEMA_VERSION,
    CacheLookupResult,
    CacheManager,
    CachedAnswer,
    _normalise_question,
)
from test.conftest import ControllableEmbeddings

FINGERPRINT_A = hashlib.sha256(b"doc_a_content").hexdigest()
FINGERPRINT_B = hashlib.sha256(b"doc_b_content").hexdigest()


class CountingEmbeddings(Embeddings):
    """Wraps an embedding model and tracks the number of embed_query calls."""

    def __init__(self, inner: Embeddings) -> None:
        self.inner = inner
        self.call_count = 0

    def embed_query(self, text: str) -> list[float]:
        self.call_count += 1
        return self.inner.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


@pytest.fixture
def fake_embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=384)


@pytest.fixture
def cm(tmp_cache_dir: Path) -> CacheManager:
    return CacheManager(cache_dir=tmp_cache_dir, threshold=0.85)


# ---------------------------------------------------------------------------
# _normalise_question
# ---------------------------------------------------------------------------


class TestNormaliseQuestion:
    def test_lowercase_and_strip(self):
        assert _normalise_question("  What Is Revenue?  ") == "what is revenue"

    def test_collapse_whitespace(self):
        assert _normalise_question("what  is   the   revenue") == "what is the revenue"

    def test_strip_trailing_punctuation(self):
        assert _normalise_question("what is revenue?!") == "what is revenue"
        assert _normalise_question("tell me more...") == "tell me more"


# ---------------------------------------------------------------------------
# Cache key generation
# ---------------------------------------------------------------------------


class TestCacheKey:
    def test_key_is_sha256_length(self):
        key = CacheManager._make_cache_key("what is revenue", FINGERPRINT_A)
        assert len(key) == 64

    def test_deterministic(self):
        key1 = CacheManager._make_cache_key("what is revenue", FINGERPRINT_A)
        key2 = CacheManager._make_cache_key("what is revenue", FINGERPRINT_A)
        assert key1 == key2

    def test_different_fingerprints_differ(self):
        key1 = CacheManager._make_cache_key("what is revenue", FINGERPRINT_A)
        key2 = CacheManager._make_cache_key("what is revenue", FINGERPRINT_B)
        assert key1 != key2

    def test_same_filename_different_content_differs(self):
        fp_v1 = hashlib.sha256(b"report_v1").hexdigest()
        fp_v2 = hashlib.sha256(b"report_v2").hexdigest()
        key1 = CacheManager._make_cache_key("summary?", fp_v1)
        key2 = CacheManager._make_cache_key("summary?", fp_v2)
        assert key1 != key2

    def test_normalisation_in_key(self):
        key1 = CacheManager._make_cache_key("What is revenue?", FINGERPRINT_A)
        key2 = CacheManager._make_cache_key("  what  is  revenue?!  ", FINGERPRINT_A)
        assert key1 == key2


# ---------------------------------------------------------------------------
# CachedAnswer schema
# ---------------------------------------------------------------------------


class TestCachedAnswer:
    def test_to_dict_includes_schema_version(self):
        entry = CachedAnswer(
            question="What is revenue?",
            normalised_question="what is revenue",
            answer="Revenue is 5M.",
            citations=[{"source": "report.pdf", "page": 1, "snippet": "Rev..."}],
            document_fingerprint=FINGERPRINT_A,
            document_names=["report.pdf"],
            llm_model="phi4-mini",
            embedding_model="all-minilm:l6-v2",
        )
        d = entry.to_dict()
        assert d["schema_version"] == CACHE_SCHEMA_VERSION
        assert d["normalised_question"] == "what is revenue"
        assert d["document_fingerprint"] == FINGERPRINT_A
        assert d["llm_model"] == "phi4-mini"
        assert "timestamp" in d


# ---------------------------------------------------------------------------
# Save and lookup (Exact & Disk hit)
# ---------------------------------------------------------------------------


class TestSaveAndLookup:
    def test_save_and_tier1_hit(self, cm, fake_embeddings):
        cm.save(
            question="What is revenue?",
            answer="Revenue is 5M.",
            citations=[],
            document_fingerprint=FINGERPRINT_A,
            document_names=["report.pdf"],
            embeddings_model=fake_embeddings,
            llm_model="phi4-mini",
            embedding_model="all-minilm",
        )
        result = cm.lookup("What is revenue?", FINGERPRINT_A)
        assert result.entry is not None
        assert result.similarity == 1.0
        assert result.entry["answer"] == "Revenue is 5M."
        assert result.entry["schema_version"] == CACHE_SCHEMA_VERSION
        assert result.query_embedding is None

    def test_tier2_disk_hit(self, cm, fake_embeddings, tmp_cache_dir):
        """Saving to disk then creating a new CacheManager loads on first lookup."""
        cm.save(
            question="What is revenue?",
            answer="Revenue is 5M.",
            citations=[],
            document_fingerprint=FINGERPRINT_A,
            document_names=["report.pdf"],
            embeddings_model=fake_embeddings,
        )

        cm2 = CacheManager(cache_dir=tmp_cache_dir, threshold=0.85)
        result = cm2.lookup("What is revenue?", FINGERPRINT_A)
        assert result.entry is not None
        assert result.similarity == 1.0

    def test_miss_different_fingerprint_exact(self, cm, fake_embeddings):
        cm.save(
            question="What is revenue?",
            answer="Revenue is 5M.",
            citations=[],
            document_fingerprint=FINGERPRINT_A,
            document_names=["a.pdf"],
            embeddings_model=fake_embeddings,
        )
        result = cm.lookup("What is revenue?", FINGERPRINT_B)
        assert result.entry is None
        assert result.similarity == 0.0

    def test_atomic_write(self, cm, fake_embeddings, tmp_cache_dir):
        cm.save(
            question="test",
            answer="answer",
            citations=[],
            document_fingerprint=FINGERPRINT_A,
            document_names=["a.pdf"],
            embeddings_model=fake_embeddings,
        )
        tmp_files = list(tmp_cache_dir.glob("*.tmp"))
        assert tmp_files == []


# ---------------------------------------------------------------------------
# Semantic lookup & Cross-document isolation (T11)
# ---------------------------------------------------------------------------


class TestSemanticLookupAndIsolation:
    def test_semantic_hit_above_threshold(self, cm, controllable_embeddings):
        # Set up 2 vectors with high similarity (~0.95)
        vec_q1 = [1.0] + [0.0] * 383
        vec_q2 = [0.95, 0.3122499] + [0.0] * 382  # unit vector dot product = 0.95

        controllable_embeddings.register("what is total revenue", vec_q1)
        controllable_embeddings.register("tell me about total revenue", vec_q2)

        cm.save(
            question="what is total revenue",
            answer="Revenue was $10M.",
            citations=[],
            document_fingerprint=FINGERPRINT_A,
            document_names=["report.pdf"],
            embeddings_model=controllable_embeddings,
        )

        result = cm.lookup(
            question="tell me about total revenue",
            document_fingerprint=FINGERPRINT_A,
            embeddings_model=controllable_embeddings,
        )
        assert result.entry is not None
        assert result.similarity >= 0.90
        assert result.entry["answer"] == "Revenue was $10M."
        assert result.query_embedding is not None

    def test_semantic_miss_below_threshold(self, cm, controllable_embeddings):
        # Orthogonal vectors: dot product = 0.0
        vec_q1 = [1.0] + [0.0] * 383
        vec_q2 = [0.0, 1.0] + [0.0] * 382

        controllable_embeddings.register("what is total revenue", vec_q1)
        controllable_embeddings.register("how is the weather", vec_q2)

        cm.save(
            question="what is total revenue",
            answer="Revenue was $10M.",
            citations=[],
            document_fingerprint=FINGERPRINT_A,
            document_names=["report.pdf"],
            embeddings_model=controllable_embeddings,
        )

        result = cm.lookup(
            question="how is the weather",
            document_fingerprint=FINGERPRINT_A,
            embeddings_model=controllable_embeddings,
        )
        assert result.entry is None
        assert result.similarity < 0.1
        assert result.query_embedding is not None

    def test_no_cross_document_set_hits(self, cm, controllable_embeddings):
        """Answers generated for doc A must NEVER match queries against doc B."""
        # Identical questions and identical embedding vectors
        vec = [1.0] + [0.0] * 383
        controllable_embeddings.register("summary of findings", vec)

        cm.save(
            question="summary of findings",
            answer="Findings for Document A.",
            citations=[],
            document_fingerprint=FINGERPRINT_A,
            document_names=["doc_a.pdf"],
            embeddings_model=controllable_embeddings,
        )

        # Lookup with Document B's fingerprint
        result = cm.lookup(
            question="summary of findings",
            document_fingerprint=FINGERPRINT_B,
            embeddings_model=controllable_embeddings,
        )
        # Even with identical query and vector, document B must miss!
        assert result.entry is None
        assert result.similarity == 0.0


# ---------------------------------------------------------------------------
# Threshold changes & Embedding reuse
# ---------------------------------------------------------------------------


class TestThresholdAndEmbeddingReuse:
    def test_runtime_threshold_change(self, cm, controllable_embeddings):
        vec_q1 = [1.0] + [0.0] * 383
        vec_q2 = [0.90, 0.43588989] + [0.0] * 382  # dot product = 0.90

        controllable_embeddings.register("what is revenue", vec_q1)
        controllable_embeddings.register("revenue amount", vec_q2)

        cm.save(
            question="what is revenue",
            answer="$5M",
            citations=[],
            document_fingerprint=FINGERPRINT_A,
            document_names=["a.pdf"],
            embeddings_model=controllable_embeddings,
        )

        # Default threshold is 0.85 -> 0.90 hits
        assert cm.lookup("revenue amount", FINGERPRINT_A, controllable_embeddings).entry is not None

        # Change threshold via set_threshold to 0.95 -> 0.90 misses
        cm.set_threshold(0.95)
        assert cm.lookup("revenue amount", FINGERPRINT_A, controllable_embeddings).entry is None

        # Per-call threshold override: 0.80 -> hits
        res = cm.lookup("revenue amount", FINGERPRINT_A, controllable_embeddings, threshold=0.80)
        assert res.entry is not None

        # Invalid threshold raises ValueError
        with pytest.raises(ValueError):
            cm.set_threshold(0.0)
        with pytest.raises(ValueError):
            cm.set_threshold(1.5)

    def test_embedding_reuse_halves_calls(self, cm, controllable_embeddings):
        counting = CountingEmbeddings(controllable_embeddings)

        # Miss during lookup should embed the query
        res = cm.lookup(
            question="New question?",
            document_fingerprint=FINGERPRINT_A,
            embeddings_model=counting,
        )
        assert res.entry is None
        assert res.query_embedding is not None
        assert counting.call_count == 1

        # Save reusing the precomputed query_embedding
        cm.save(
            question="New question?",
            answer="Generated answer",
            citations=[],
            document_fingerprint=FINGERPRINT_A,
            document_names=["a.pdf"],
            query_embedding=res.query_embedding,
        )
        # Call count must still be 1 (embed_query was NOT called during save)
        assert counting.call_count == 1

    def test_dimension_mismatch_guard(self, cm):
        """Cached vectors with different dimension than query are safely skipped."""
        dim384 = ControllableEmbeddings(size=384)
        dim128 = ControllableEmbeddings(size=128)

        cm.save(
            question="Question with dim 384",
            answer="Answer 384",
            citations=[],
            document_fingerprint=FINGERPRINT_A,
            document_names=["a.pdf"],
            embeddings_model=dim384,
        )

        # Query with dim 128 model must not raise ValueError / crash
        res = cm.lookup(
            question="Another query",
            document_fingerprint=FINGERPRINT_A,
            embeddings_model=dim128,
        )
        assert res.entry is None


# ---------------------------------------------------------------------------
# Oldest-first Eviction & Max Entries
# ---------------------------------------------------------------------------


class TestMaxEntriesEviction:
    def test_oldest_entry_evicted_when_max_exceeded(self, tmp_cache_dir, fake_embeddings):
        cm = CacheManager(cache_dir=tmp_cache_dir, max_entries=2)

        cm.save(
            question="Q1",
            answer="A1",
            citations=[],
            document_fingerprint=FINGERPRINT_A,
            document_names=["a.pdf"],
            embeddings_model=fake_embeddings,
        )
        cm.save(
            question="Q2",
            answer="A2",
            citations=[],
            document_fingerprint=FINGERPRINT_A,
            document_names=["a.pdf"],
            embeddings_model=fake_embeddings,
        )
        assert cm.entry_count == 2

        # Adding a 3rd entry must evict Q1
        cm.save(
            question="Q3",
            answer="A3",
            citations=[],
            document_fingerprint=FINGERPRINT_A,
            document_names=["a.pdf"],
            embeddings_model=fake_embeddings,
        )
        assert cm.entry_count == 2
        assert cm.lookup("Q1", FINGERPRINT_A).entry is None
        assert cm.lookup("Q2", FINGERPRINT_A).entry is not None
        assert cm.lookup("Q3", FINGERPRINT_A).entry is not None


# ---------------------------------------------------------------------------
# Legacy Purging & Corrupt File Tolerance
# ---------------------------------------------------------------------------


class TestLegacyPurgingAndCorruptFiles:
    def test_md5_key_purged_on_load(self, cm, tmp_cache_dir):
        md5_key = "a" * 32
        answer_path = tmp_cache_dir / f"{md5_key}.json"
        embedding_path = tmp_cache_dir / f"{md5_key}.embedding.json"
        answer_path.write_text(json.dumps({"question": "old", "answer": "old"}))
        embedding_path.write_text(json.dumps({"embedding": [0.0] * 10}))

        cm.load_index_from_disk()

        assert md5_key not in cm._memory_cache
        assert not answer_path.exists()
        assert not embedding_path.exists()

    def test_old_schema_version_purged(self, cm, tmp_cache_dir):
        fake_key = "b" * 64
        answer_path = tmp_cache_dir / f"{fake_key}.json"
        embedding_path = tmp_cache_dir / f"{fake_key}.embedding.json"
        answer_path.write_text(
            json.dumps({"schema_version": 1, "question": "old", "answer": "old"})
        )
        embedding_path.write_text(json.dumps({"embedding": [0.0] * 10}))

        cm.load_index_from_disk()

        assert fake_key not in cm._memory_cache
        assert not answer_path.exists()
        assert not embedding_path.exists()

    def test_corrupt_json_file_tolerated_and_purged(self, cm, tmp_cache_dir):
        fake_key = "c" * 64
        answer_path = tmp_cache_dir / f"{fake_key}.json"
        embedding_path = tmp_cache_dir / f"{fake_key}.embedding.json"
        answer_path.write_text("{this is corrupt json...")
        embedding_path.write_text(json.dumps({"embedding": [0.1] * 384}))

        # Must not raise JSONDecodeError
        cm.load_index_from_disk()
        assert fake_key not in cm._memory_cache
        assert not answer_path.exists()


# ---------------------------------------------------------------------------
# Selective Clear
# ---------------------------------------------------------------------------


class TestSelectiveClear:
    def test_clear_all_preserves_non_cache_files(self, cm, fake_embeddings, tmp_cache_dir):
        # Create a non-cache file
        keep_file = tmp_cache_dir / "notes.txt"
        keep_file.write_text("Do not delete me")

        cm.save(
            question="Q1",
            answer="A1",
            citations=[],
            document_fingerprint=FINGERPRINT_A,
            document_names=["a.pdf"],
            embeddings_model=fake_embeddings,
        )
        removed = cm.clear()
        assert removed == 1
        assert cm.entry_count == 0
        assert keep_file.exists()

    def test_clear_by_fingerprint(self, cm, fake_embeddings):
        cm.save(
            question="Q1",
            answer="A1",
            citations=[],
            document_fingerprint=FINGERPRINT_A,
            document_names=["a.pdf"],
            embeddings_model=fake_embeddings,
        )
        cm.save(
            question="Q2",
            answer="A2",
            citations=[],
            document_fingerprint=FINGERPRINT_B,
            document_names=["b.pdf"],
            embeddings_model=fake_embeddings,
        )
        removed = cm.clear(document_fingerprint=FINGERPRINT_A)
        assert removed == 1
        assert cm.entry_count == 1
        assert cm.lookup("Q2", FINGERPRINT_B).entry is not None


# ---------------------------------------------------------------------------
# Thread Safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_saves(self, tmp_cache_dir, fake_embeddings):
        cm = CacheManager(cache_dir=tmp_cache_dir)
        errors: list[Exception] = []

        def worker(idx: int):
            try:
                cm.save(
                    question=f"Question {idx}",
                    answer=f"Answer {idx}",
                    citations=[],
                    document_fingerprint=FINGERPRINT_A,
                    document_names=["doc.pdf"],
                    embeddings_model=fake_embeddings,
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert cm.entry_count == 10
