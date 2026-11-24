"""Unit tests for the semantic answer cache (cache_manager.py).

Tests SHA-256 keying, determinism, schema versioning, legacy purging,
3-tier lookup, and document fingerprint isolation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from documate_ai.cache_manager import (
    CACHE_SCHEMA_VERSION,
    CacheManager,
    CachedAnswer,
    _normalise_question,
)

FINGERPRINT_A = hashlib.sha256(b"doc_a_content").hexdigest()
FINGERPRINT_B = hashlib.sha256(b"doc_b_content").hexdigest()


@pytest.fixture
def embeddings() -> DeterministicFakeEmbedding:
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
        assert len(key) == 64  # SHA-256 hex digest

    def test_deterministic(self):
        key1 = CacheManager._make_cache_key("what is revenue", FINGERPRINT_A)
        key2 = CacheManager._make_cache_key("what is revenue", FINGERPRINT_A)
        assert key1 == key2

    def test_different_fingerprints_differ(self):
        key1 = CacheManager._make_cache_key("what is revenue", FINGERPRINT_A)
        key2 = CacheManager._make_cache_key("what is revenue", FINGERPRINT_B)
        assert key1 != key2

    def test_same_filename_different_content_differs(self):
        """Same question but different document content (fingerprint) must differ."""
        fp_v1 = hashlib.sha256(b"report_v1").hexdigest()
        fp_v2 = hashlib.sha256(b"report_v2").hexdigest()
        key1 = CacheManager._make_cache_key("summary?", fp_v1)
        key2 = CacheManager._make_cache_key("summary?", fp_v2)
        assert key1 != key2

    def test_normalisation_in_key(self):
        """Whitespace and trailing punctuation should not change the key."""
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
# Save and lookup (Tier 1 + Tier 2)
# ---------------------------------------------------------------------------


class TestSaveAndLookup:
    def test_save_and_tier1_hit(self, cm, embeddings):
        cm.save(
            question="What is revenue?",
            answer="Revenue is 5M.",
            citations=[],
            document_fingerprint=FINGERPRINT_A,
            document_names=["report.pdf"],
            embeddings_model=embeddings,
            llm_model="phi4-mini",
            embedding_model="all-minilm",
        )
        entry, score = cm.lookup("What is revenue?", FINGERPRINT_A)
        assert entry is not None
        assert score == 1.0
        assert entry["answer"] == "Revenue is 5M."
        assert entry["schema_version"] == CACHE_SCHEMA_VERSION

    def test_tier2_disk_hit(self, cm, embeddings, tmp_cache_dir):
        """Saving to disk then creating a new CacheManager should produce a Tier 2 hit."""
        cm.save(
            question="What is revenue?",
            answer="Revenue is 5M.",
            citations=[],
            document_fingerprint=FINGERPRINT_A,
            document_names=["report.pdf"],
            embeddings_model=embeddings,
        )

        # Fresh CacheManager with empty memory
        cm2 = CacheManager(cache_dir=tmp_cache_dir, threshold=0.85)
        entry, score = cm2.lookup("What is revenue?", FINGERPRINT_A)
        assert entry is not None
        assert score == 1.0

    def test_miss_different_fingerprint(self, cm, embeddings):
        cm.save(
            question="What is revenue?",
            answer="Revenue is 5M.",
            citations=[],
            document_fingerprint=FINGERPRINT_A,
            document_names=["a.pdf"],
            embeddings_model=embeddings,
        )
        entry, score = cm.lookup("What is revenue?", FINGERPRINT_B)
        assert entry is None
        assert score == 0.0

    def test_atomic_write(self, cm, embeddings, tmp_cache_dir):
        """After save, no .tmp files should remain."""
        cm.save(
            question="test",
            answer="answer",
            citations=[],
            document_fingerprint=FINGERPRINT_A,
            document_names=["a.pdf"],
            embeddings_model=embeddings,
        )
        tmp_files = list(tmp_cache_dir.glob("*.tmp"))
        assert tmp_files == []


# ---------------------------------------------------------------------------
# Legacy purging
# ---------------------------------------------------------------------------


class TestLegacyPurging:
    def test_md5_key_purged_on_load(self, cm, tmp_cache_dir):
        """32-char MD5-style cache keys should be silently purged on load."""
        md5_key = "a" * 32  # simulates an old MD5 key
        answer_path = tmp_cache_dir / f"{md5_key}.json"
        embedding_path = tmp_cache_dir / f"{md5_key}.embedding.json"
        answer_path.write_text(json.dumps({"question": "old", "answer": "old"}))
        embedding_path.write_text(json.dumps({"embedding": [0.0] * 10}))

        cm.load_index_from_disk()

        assert md5_key not in cm._memory_cache
        assert not answer_path.exists()
        assert not embedding_path.exists()

    def test_old_schema_version_purged(self, cm, embeddings, tmp_cache_dir):
        """Entries with schema_version < CACHE_SCHEMA_VERSION should be purged."""
        fake_key = "b" * 64  # 64-char key (looks like SHA-256)
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

    def test_current_schema_loaded(self, cm, embeddings, tmp_cache_dir):
        """Entries with the current schema version should be loaded successfully."""
        entry = CachedAnswer(
            question="What is X?",
            normalised_question="what is x",
            answer="X is ...",
            citations=[],
            document_fingerprint=FINGERPRINT_A,
            document_names=["x.pdf"],
            llm_model="phi4-mini",
            embedding_model="all-minilm",
        )
        # Build a valid cache key
        key = CacheManager._make_cache_key("What is X?", FINGERPRINT_A)
        answer_path = tmp_cache_dir / f"{key}.json"
        embedding_path = tmp_cache_dir / f"{key}.embedding.json"
        answer_path.write_text(json.dumps(entry.to_dict()))
        embedding_path.write_text(json.dumps({"embedding": [0.1] * 384}))

        cm.load_index_from_disk()

        assert key in cm._memory_cache
        assert cm._memory_cache[key]["schema_version"] == CACHE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_all(self, cm, embeddings):
        cm.save(
            question="Q1",
            answer="A1",
            citations=[],
            document_fingerprint=FINGERPRINT_A,
            document_names=["a.pdf"],
            embeddings_model=embeddings,
        )
        cm.save(
            question="Q2",
            answer="A2",
            citations=[],
            document_fingerprint=FINGERPRINT_B,
            document_names=["b.pdf"],
            embeddings_model=embeddings,
        )
        removed = cm.clear()
        assert removed == 2
        assert cm.entry_count == 0

    def test_clear_by_fingerprint(self, cm, embeddings):
        cm.save(
            question="Q1",
            answer="A1",
            citations=[],
            document_fingerprint=FINGERPRINT_A,
            document_names=["a.pdf"],
            embeddings_model=embeddings,
        )
        cm.save(
            question="Q2",
            answer="A2",
            citations=[],
            document_fingerprint=FINGERPRINT_B,
            document_names=["b.pdf"],
            embeddings_model=embeddings,
        )
        removed = cm.clear(document_fingerprint=FINGERPRINT_A)
        assert removed == 1
        assert cm.entry_count == 1

        # The remaining entry should be from fingerprint B
        entry, _ = cm.lookup("Q2", FINGERPRINT_B)
        assert entry is not None


# ---------------------------------------------------------------------------
# No MD5 usage
# ---------------------------------------------------------------------------


class TestNoMD5:
    def test_no_md5_in_module(self):
        import inspect
        import documate_ai.cache_manager as mod

        source = inspect.getsource(mod)
        assert "hashlib.md5" not in source
        assert ".md5(" not in source
