"""Semantic answer cache with 3-tier lookup, versioned schema, and robust isolation.

Implements a hardened caching strategy to avoid redundant LLM calls:
    Tier 1 — In-memory exact match via SHA-256 hash (instant, O(1))
    Tier 2 — In-memory semantic match via cosine similarity of embeddings (fast)

All persistent entries are loaded lazily into memory on first access, making
a separate on-disk exact-match tier redundant.

**Hardening and Robustness features (T11):**
    - **Cross-document isolation**: Semantic lookup filters candidates strictly
      by ``document_fingerprint``, completely preventing cross-document cache hits.
    - **L2-normalised vectors**: All vectors are unit-normalised upon insertion so
      cosine similarity reduces to a fast matrix-vector dot product.
    - **Dimension mismatch guards**: Prevents crashes if an embedding model changes
      dimensions between runs.
    - **Embedding reuse**: ``lookup()`` returns the computed query embedding so
      subsequent ``save()`` calls can reuse it without re-embedding the question.
    - **Thread safety**: Internal state and file writes are guarded by an ``RLock``.
    - **Selective clear**: ``clear()`` deletes only cache-owned files (matching a
      strict 64-char hex filename pattern) and supports clearing a single document set.
    - **Oldest-first eviction**: Optional ``max_entries`` setting to cap cache size.
    - **Runtime threshold**: Threshold can be updated via ``set_threshold()`` or
      overridden per ``lookup()`` call.

Storage layout: **two JSON files per entry**:
    - ``<sha256>.json`` — the answer payload.
    - ``<sha256>.embedding.json`` — the query embedding vector.

This module is **fully decoupled from Streamlit** — no ``st.session_state``
references. All state is held inside the ``CacheManager`` instance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from documate_ai.config import Settings, get_settings

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION: int = 2
"""Current on-disk schema version. Entries with an older version are ignored."""

_CACHE_FILE_RE = re.compile(r"^[0-9a-f]{64}(\.embedding)?\.json(\.tmp)?$")
"""Strict regex matching only files created by the cache manager."""


class CacheLookupResult(NamedTuple):
    """Result of a cache lookup operation."""

    entry: dict[str, Any] | None
    similarity: float
    query_embedding: list[float] | None = None


@dataclass
class CachedAnswer:
    """A single cached question-answer pair with full metadata."""

    question: str
    normalised_question: str
    answer: str
    citations: list[dict[str, Any]]
    document_fingerprint: str
    document_names: list[str]
    llm_model: str
    embedding_model: str
    schema_version: int = CACHE_SCHEMA_VERSION
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary (excludes embedding)."""
        return {
            "schema_version": self.schema_version,
            "question": self.question,
            "normalised_question": self.normalised_question,
            "answer": self.answer,
            "citations": self.citations,
            "document_fingerprint": self.document_fingerprint,
            "document_names": self.document_names,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "timestamp": self.timestamp,
        }


def _normalise_question(question: str) -> str:
    """Normalise a question for deterministic cache key generation.

    Lowercases, strips leading/trailing whitespace, collapses runs of
    internal whitespace to a single space, and removes trailing
    punctuation.

    Args:
        question: Raw question string.

    Returns:
        Normalised question string.
    """
    q = question.lower().strip()
    q = re.sub(r"\s+", " ", q)
    q = q.rstrip("?!.,;:")
    return q


def _l2_normalise(vec: np.ndarray) -> np.ndarray:
    """Normalise a vector to unit length (L2 norm)."""
    norm = float(np.linalg.norm(vec))
    if norm > 1e-12:
        return vec / norm
    return vec


class CacheManager:
    """Thread-safe semantic answer cache with per-document isolation.

    Attributes:
        cache_dir: Path to the on-disk cache directory.
        threshold: Minimum cosine similarity for a semantic cache hit.
        max_entries: Optional maximum number of entries to retain.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        threshold: float | None = None,
        settings: Settings | None = None,
        max_entries: int | None = None,
    ) -> None:
        """Initialise CacheManager with configuration and thread locks.

        Args:
            cache_dir: Directory where cached answers are persisted.
            threshold: Minimum cosine similarity to accept a semantic match.
            settings: Application settings for defaults.
            max_entries: Maximum number of entries to keep before oldest-first
                eviction takes place.
        """
        if settings is None:
            settings = get_settings()

        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else (settings.cache_dir or Path("./data/cache"))
        )
        self.threshold = (
            threshold if threshold is not None else settings.similarity_threshold
        )
        self.max_entries = max_entries

        self._lock = threading.RLock()

        # In-memory cache: cache_key -> entry_dict
        self._memory_cache: dict[str, dict[str, Any]] = {}

        # Cached L2-normalised embeddings: cache_key -> unit vector
        self._embeddings: dict[str, np.ndarray] = {}

        # Per-fingerprint index cache: fingerprint -> (list_of_keys, 2D_matrix)
        self._fingerprint_matrices: dict[str, tuple[list[str], np.ndarray]] = {}

        self._index_loaded: bool = False

    # ------------------------------------------------------------------
    # Properties & Configuration
    # ------------------------------------------------------------------

    @property
    def entry_count(self) -> int:
        """Return the number of entries currently held in memory."""
        with self._lock:
            return len(self._memory_cache)

    def set_threshold(self, threshold: float) -> None:
        """Update the default similarity threshold.

        Args:
            threshold: Value in range (0.0, 1.0].

        Raises:
            ValueError: If threshold is outside (0.0, 1.0].
        """
        if not (0.0 < threshold <= 1.0):
            raise ValueError(
                f"similarity_threshold must be between 0 and 1, got {threshold}"
            )
        with self._lock:
            self.threshold = threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(
        self,
        question: str,
        document_fingerprint: str,
        embeddings_model: Embeddings | None = None,
        threshold: float | None = None,
    ) -> CacheLookupResult:
        """Look up a cached answer using exact and isolated semantic matching.

        Checks:
            1. In-memory exact match on SHA-256 key (O(1)).
            2. Semantic match restricted strictly to entries matching
               ``document_fingerprint`` (dot-product of unit vectors).

        Args:
            question: The user's question.
            document_fingerprint: SHA-256 fingerprint of the loaded document set.
            embeddings_model: Embedding model needed for semantic matching.
            threshold: Per-call similarity threshold override.

        Returns:
            A ``CacheLookupResult`` containing ``(entry, similarity, query_embedding)``.
        """
        self.load_index_from_disk()

        effective_threshold = (
            threshold if threshold is not None else self.threshold
        )
        cache_key = self._make_cache_key(question, document_fingerprint)

        with self._lock:
            # Tier 1: exact match
            if cache_key in self._memory_cache:
                logger.debug(
                    "Cache HIT (Tier 1 — exact) for key %s", cache_key[:8]
                )
                return CacheLookupResult(
                    entry=self._memory_cache[cache_key],
                    similarity=1.0,
                    query_embedding=None,
                )

        # Tier 2: semantic match (filtered strictly by fingerprint)
        if embeddings_model is not None:
            return self._semantic_lookup(
                question=question,
                document_fingerprint=document_fingerprint,
                embeddings_model=embeddings_model,
                threshold=effective_threshold,
            )

        return CacheLookupResult(entry=None, similarity=0.0, query_embedding=None)

    def save(
        self,
        question: str,
        answer: str,
        citations: list[dict[str, Any]],
        document_fingerprint: str,
        document_names: list[str],
        embeddings_model: Embeddings | None = None,
        llm_model: str = "",
        embedding_model: str = "",
        query_embedding: list[float] | None = None,
    ) -> None:
        """Persist a new answer to both memory and disk caches.

        Reuses ``query_embedding`` if supplied, avoiding a redundant embedding
        computation.

        Args:
            question: The question that was asked.
            answer: The generated answer.
            citations: Structured Citation dicts.
            document_fingerprint: SHA-256 fingerprint of the document set.
            document_names: List of loaded PDF filenames (for display).
            embeddings_model: Embedding model (required if ``query_embedding``
                is None).
            llm_model: Name of the LLM model used.
            embedding_model: Name of the embedding model used.
            query_embedding: Precomputed embedding vector from ``lookup()``.

        Raises:
            ValueError: If neither ``query_embedding`` nor ``embeddings_model``
                is provided.
        """
        normalised = _normalise_question(question)
        cache_key = self._make_cache_key(question, document_fingerprint)

        if query_embedding is not None:
            raw_embedding = list(query_embedding)
        elif embeddings_model is not None:
            raw_embedding = embeddings_model.embed_query(normalised)
        else:
            raise ValueError(
                "Either query_embedding or embeddings_model must be provided to save()."
            )

        unit_embedding = _l2_normalise(
            np.asarray(raw_embedding, dtype=np.float32)
        )

        cache_entry = CachedAnswer(
            question=question,
            normalised_question=normalised,
            answer=answer,
            citations=citations,
            document_fingerprint=document_fingerprint,
            document_names=sorted(document_names),
            llm_model=llm_model,
            embedding_model=embedding_model,
        )
        entry_dict = cache_entry.to_dict()

        with self._lock:
            # In-memory update
            self._memory_cache[cache_key] = entry_dict
            self._embeddings[cache_key] = unit_embedding
            self._invalidate_fingerprint_matrix(document_fingerprint)

            # Evict oldest if max_entries is exceeded
            if (
                self.max_entries is not None
                and len(self._memory_cache) > self.max_entries
            ):
                self._evict_oldest()

            # Disk persistence (atomic write)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            answer_path = self.cache_dir / f"{cache_key}.json"
            embedding_path = self.cache_dir / f"{cache_key}.embedding.json"

            tmp_answer = answer_path.with_suffix(".json.tmp")
            tmp_embedding = embedding_path.with_suffix(".json.tmp")

            with open(tmp_answer, "w", encoding="utf-8") as f:
                json.dump(entry_dict, f)
            os.replace(str(tmp_answer), str(answer_path))

            with open(tmp_embedding, "w", encoding="utf-8") as f:
                json.dump({"embedding": raw_embedding}, f)
            os.replace(str(tmp_embedding), str(embedding_path))

            logger.info("Saved answer to cache (key=%s)", cache_key[:8])

    def clear(self, *, document_fingerprint: str | None = None) -> int:
        """Remove cached answers from memory and disk.

        Only files matching the cache naming pattern (64-char hex) are removed;
        unrelated files in ``cache_dir`` are preserved.

        Args:
            document_fingerprint: If given, only entries matching this
                fingerprint are cleared. Otherwise all cache entries are removed.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            if document_fingerprint is not None:
                return self._clear_by_fingerprint(document_fingerprint)

            count = len(self._memory_cache)
            self._memory_cache.clear()
            self._embeddings.clear()
            self._fingerprint_matrices.clear()
            self._index_loaded = False

            if self.cache_dir.exists():
                for file in self.cache_dir.iterdir():
                    if file.is_file() and _CACHE_FILE_RE.match(file.name):
                        file.unlink(missing_ok=True)

            logger.info("Cache cleared (%d entries)", count)
            return count

    def load_index_from_disk(self) -> None:
        """Load cached questions and embeddings from disk into memory.

        Scans the cache directory once, loading valid entries into memory and
        silently purging legacy MD5 keys (32-char) or obsolete schema entries.
        Guarded to run at most once.
        """
        with self._lock:
            if self._index_loaded:
                return

            if not self.cache_dir.is_dir():
                self._index_loaded = True
                return

            loaded_count = 0
            purged_count = 0

            for entry in os.scandir(self.cache_dir):
                if not entry.is_file() or not entry.name.endswith(".embedding.json"):
                    continue

                cache_key = entry.name.replace(".embedding.json", "")
                answer_path = self.cache_dir / f"{cache_key}.json"

                # Purge legacy 32-char MD5 keys
                if len(cache_key) == 32:
                    self._purge_entry(cache_key)
                    purged_count += 1
                    continue

                if not answer_path.exists():
                    continue

                try:
                    with open(answer_path, encoding="utf-8") as f:
                        answer_data = json.load(f)

                    # Purge old schema versions
                    if (
                        answer_data.get("schema_version", 0)
                        < CACHE_SCHEMA_VERSION
                    ):
                        self._purge_entry(cache_key)
                        purged_count += 1
                        continue

                    with open(entry.path, encoding="utf-8") as f:
                        embedding_data = json.load(f)

                    raw_emb = embedding_data.get("embedding", [])
                    if not raw_emb:
                        self._purge_entry(cache_key)
                        continue

                    unit_emb = _l2_normalise(
                        np.asarray(raw_emb, dtype=np.float32)
                    )

                    self._memory_cache[cache_key] = answer_data
                    self._embeddings[cache_key] = unit_emb
                    loaded_count += 1
                except (json.JSONDecodeError, KeyError, OSError) as exc:
                    logger.warning(
                        "Corrupt cache entry %s: %s; purging.",
                        cache_key[:8],
                        exc,
                    )
                    self._purge_entry(cache_key)
                    purged_count += 1
                    continue

            self._fingerprint_matrices.clear()
            self._index_loaded = True

            if purged_count:
                logger.info("Purged %d legacy cache entries", purged_count)
            logger.info("Loaded %d cached entries from disk", loaded_count)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_cache_key(question: str, document_fingerprint: str) -> str:
        """Generate a SHA-256 hash key from normalised question and document fingerprint."""
        normalised = _normalise_question(question)
        raw = normalised + "|" + document_fingerprint
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_fingerprint_matrix(
        self, document_fingerprint: str, expected_dim: int
    ) -> tuple[list[str], np.ndarray] | None:
        """Retrieve or build the candidate embedding matrix for a document fingerprint."""
        if document_fingerprint in self._fingerprint_matrices:
            keys, matrix = self._fingerprint_matrices[document_fingerprint]
            if matrix.shape[1] == expected_dim:
                return keys, matrix
            logger.warning(
                "Cached matrix dimension %d does not match expected %d; rebuilding.",
                matrix.shape[1],
                expected_dim,
            )

        # Collect keys belonging strictly to this fingerprint with matching dimension
        matching_keys: list[str] = []
        matching_vectors: list[np.ndarray] = []

        for key, entry in self._memory_cache.items():
            if entry.get("document_fingerprint") != document_fingerprint:
                continue
            emb = self._embeddings.get(key)
            if emb is None:
                continue
            if emb.shape[0] != expected_dim:
                logger.warning(
                    "Skipping key %s: vector dimension %d != query dimension %d",
                    key[:8],
                    emb.shape[0],
                    expected_dim,
                )
                continue
            matching_keys.append(key)
            matching_vectors.append(emb)

        if not matching_keys:
            return None

        matrix = np.vstack(matching_vectors)
        self._fingerprint_matrices[document_fingerprint] = (matching_keys, matrix)
        return matching_keys, matrix

    def _invalidate_fingerprint_matrix(self, document_fingerprint: str) -> None:
        """Invalidate the cached matrix for a given fingerprint."""
        self._fingerprint_matrices.pop(document_fingerprint, None)

    def _semantic_lookup(
        self,
        question: str,
        document_fingerprint: str,
        embeddings_model: Embeddings,
        threshold: float,
    ) -> CacheLookupResult:
        """Search for a semantically similar cached question within the same document set."""
        normalised = _normalise_question(question)
        raw_query_embedding = embeddings_model.embed_query(normalised)
        query_vec = _l2_normalise(
            np.asarray(raw_query_embedding, dtype=np.float32)
        )
        query_dim = query_vec.shape[0]

        with self._lock:
            index_data = self._get_fingerprint_matrix(
                document_fingerprint, expected_dim=query_dim
            )
            if index_data is None:
                return CacheLookupResult(
                    entry=None,
                    similarity=0.0,
                    query_embedding=raw_query_embedding,
                )

            keys, matrix = index_data
            # Fast dot product on pre-normalised unit vectors
            similarities = np.clip(matrix @ query_vec, -1.0, 1.0)
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])
            best_key = keys[best_idx]

            logger.debug(
                "Semantic search [%s]: best match score=%.4f (threshold=%.2f)",
                document_fingerprint[:8],
                best_score,
                threshold,
            )

            if best_score >= threshold:
                logger.debug(
                    "Cache HIT (Tier 2 — semantic) for key %s", best_key[:8]
                )
                return CacheLookupResult(
                    entry=self._memory_cache[best_key],
                    similarity=best_score,
                    query_embedding=raw_query_embedding,
                )

            return CacheLookupResult(
                entry=None,
                similarity=best_score,
                query_embedding=raw_query_embedding,
            )

    def _purge_entry(self, cache_key: str) -> None:
        """Remove a single cache entry from disk."""
        answer_path = self.cache_dir / f"{cache_key}.json"
        embedding_path = self.cache_dir / f"{cache_key}.embedding.json"
        answer_path.unlink(missing_ok=True)
        embedding_path.unlink(missing_ok=True)

    def _clear_by_fingerprint(self, fingerprint: str) -> int:
        """Clear only entries that belong to a specific document fingerprint."""
        to_remove = [
            k
            for k, v in self._memory_cache.items()
            if v.get("document_fingerprint") == fingerprint
        ]
        for k in to_remove:
            self._memory_cache.pop(k, None)
            self._embeddings.pop(k, None)
            self._purge_entry(k)

        self._invalidate_fingerprint_matrix(fingerprint)

        if to_remove:
            logger.info(
                "Cleared %d cache entries for fingerprint %s",
                len(to_remove),
                fingerprint[:8],
            )

        return len(to_remove)

    def _evict_oldest(self) -> None:
        """Evict the single oldest cache entry by timestamp."""
        if not self._memory_cache:
            return

        oldest_key = min(
            self._memory_cache.keys(),
            key=lambda k: self._memory_cache[k].get("timestamp", ""),
        )
        fp = self._memory_cache[oldest_key].get("document_fingerprint", "")
        self._memory_cache.pop(oldest_key, None)
        self._embeddings.pop(oldest_key, None)
        self._purge_entry(oldest_key)
        if fp:
            self._invalidate_fingerprint_matrix(fp)

        logger.info("Evicted oldest cache entry %s", oldest_key[:8])
