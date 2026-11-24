"""Semantic answer cache with 3-tier lookup and versioned schema.

Implements a three-tier caching strategy to avoid redundant LLM calls:
    Tier 1 — In-memory exact match via SHA-256 hash (instant)
    Tier 2 — On-disk exact match via JSON files (very fast)
    Tier 3 — Semantic match via cosine similarity of embeddings (fast)

If all tiers miss, the caller should invoke the LLM and then call
``save()`` to persist the new answer.

**Schema v2 changes (T10):**
    - Cache keys are SHA-256 hex digests (64 chars) instead of MD5 (32 chars).
    - Keys incorporate the document-set **fingerprint** (content hash) rather
      than filenames, so the same filename with different content invalidates
      the cache.
    - Each persisted entry includes ``schema_version``, ``normalised_question``,
      ``document_fingerprint``, ``document_names``, ``llm_model``,
      ``embedding_model``, ``citations`` (structured Citation dicts from T09),
      and a UTC ISO-8601 ``timestamp``.
    - On load, entries with ``schema_version < CACHE_SCHEMA_VERSION`` or
      32-character (legacy MD5) keys are silently skipped and purged.

Storage layout: **two JSON files per entry** (unchanged from v1):
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from documate_ai.config import Settings, get_settings

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION: int = 2
"""Current on-disk schema version.  Entries with an older version are ignored."""


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


class CacheManager:
    """Three-tier semantic answer cache with SHA-256 keys and versioned schema.

    Attributes:
        cache_dir: Path to the on-disk cache directory.
        threshold: Minimum cosine similarity for a semantic cache hit.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        threshold: float | None = None,
        settings: Settings | None = None,
    ) -> None:
        if settings is None:
            settings = get_settings()

        self.cache_dir = Path(cache_dir) if cache_dir is not None else (settings.cache_dir or Path("./data/cache"))
        self.threshold = threshold if threshold is not None else settings.similarity_threshold

        # Tier 1: in-memory cache keyed by SHA-256 hash
        self._memory_cache: dict[str, dict[str, Any]] = {}

        # Tier 3: dense embedding matrix for fast vectorised cosine similarity
        self._embeddings: dict[str, list[float]] = {}
        self._embedding_matrix: np.ndarray | None = None
        self._embedding_keys: list[str] = []

        self._index_loaded: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def entry_count(self) -> int:
        """Return the number of entries currently held in memory."""
        return len(self._memory_cache)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(
        self,
        question: str,
        document_fingerprint: str,
        embeddings_model: Embeddings | None = None,
        threshold: float | None = None,
    ) -> tuple[dict[str, Any] | None, float]:
        """Look up a cached answer using the 3-tier strategy.

        Args:
            question: The user's question.
            document_fingerprint: SHA-256 fingerprint of the loaded
                document set (from ``IngestSummary``).
            embeddings_model: Embedding model needed for Tier 3 semantic
                lookup.  If ``None``, only Tier 1 and 2 are checked.
            threshold: Per-call similarity threshold override.  If
                ``None``, the instance default is used.

        Returns:
            A tuple ``(cached_entry, similarity_score)``. An exact match
            returns a score of ``1.0``. Returns ``(None, 0.0)`` on miss.
        """
        effective_threshold = threshold if threshold is not None else self.threshold
        cache_key = self._make_cache_key(question, document_fingerprint)

        # Tier 1: in-memory exact match
        if cache_key in self._memory_cache:
            logger.debug("Cache HIT (Tier 1 — memory) for key %s", cache_key[:8])
            return self._memory_cache[cache_key], 1.0

        # Tier 2: on-disk exact match
        answer_path = self.cache_dir / f"{cache_key}.json"
        embedding_path = self.cache_dir / f"{cache_key}.embedding.json"

        if answer_path.exists() and embedding_path.exists():
            try:
                with open(answer_path, encoding="utf-8") as f:
                    answer_data = json.load(f)

                # Reject legacy / old-schema entries
                if answer_data.get("schema_version", 0) < CACHE_SCHEMA_VERSION:
                    logger.debug("Skipping stale schema entry %s", cache_key[:8])
                else:
                    with open(embedding_path, encoding="utf-8") as f:
                        embedding_data = json.load(f)

                    # Promote to memory for future lookups
                    self._memory_cache[cache_key] = answer_data
                    self._embeddings[cache_key] = embedding_data.get("embedding", [])
                    self._rebuild_index()

                    logger.debug("Cache HIT (Tier 2 — disk) for key %s", cache_key[:8])
                    return answer_data, 1.0
            except (json.JSONDecodeError, KeyError, OSError) as exc:
                logger.warning("Corrupt cache file for key %s: %s", cache_key[:8], exc)

        # Tier 3: semantic match
        if embeddings_model is not None:
            return self._semantic_lookup(question, embeddings_model, effective_threshold)

        return None, 0.0

    def save(
        self,
        question: str,
        answer: str,
        citations: list[dict[str, Any]],
        document_fingerprint: str,
        document_names: list[str],
        embeddings_model: Embeddings,
        llm_model: str = "",
        embedding_model: str = "",
    ) -> None:
        """Persist a new answer to both memory and disk caches.

        Args:
            question: The question that was asked.
            answer: The generated answer.
            citations: Structured Citation dicts.
            document_fingerprint: SHA-256 fingerprint of the document set.
            document_names: List of loaded PDF filenames (for display).
            embeddings_model: The embedding model (to embed the question
                for future semantic matches).
            llm_model: Name of the LLM model used to generate the answer.
            embedding_model: Name of the embedding model used.
        """
        normalised = _normalise_question(question)
        cache_key = self._make_cache_key(question, document_fingerprint)
        question_embedding = embeddings_model.embed_query(normalised)

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

        # --- In-memory update ------------------------------------------------
        self._memory_cache[cache_key] = entry_dict
        self._embeddings[cache_key] = question_embedding

        # Incremental matrix update (faster than full rebuild)
        embedding_vector = np.asarray(question_embedding, dtype=np.float32)
        if self._embedding_matrix is None:
            self._embedding_keys = [cache_key]
            self._embedding_matrix = embedding_vector.reshape(1, -1)
        else:
            self._embedding_keys.append(cache_key)
            self._embedding_matrix = np.vstack(
                [self._embedding_matrix, embedding_vector],
            )

        # --- Disk persistence ------------------------------------------------
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        answer_path = self.cache_dir / f"{cache_key}.json"
        embedding_path = self.cache_dir / f"{cache_key}.embedding.json"

        # Atomic write: write to temp file then rename
        tmp_answer = answer_path.with_suffix(".json.tmp")
        tmp_embedding = embedding_path.with_suffix(".json.tmp")

        with open(tmp_answer, "w", encoding="utf-8") as f:
            json.dump(entry_dict, f)
        os.replace(str(tmp_answer), str(answer_path))

        with open(tmp_embedding, "w", encoding="utf-8") as f:
            json.dump({"embedding": question_embedding}, f)
        os.replace(str(tmp_embedding), str(embedding_path))

        logger.info("Saved answer to cache (key=%s)", cache_key[:8])

    def clear(self, *, document_fingerprint: str | None = None) -> int:
        """Remove cached answers from memory and disk.

        Args:
            document_fingerprint: If given, only entries matching this
                fingerprint are cleared.  Otherwise all entries are removed.

        Returns:
            Number of entries removed.
        """
        if document_fingerprint is not None:
            return self._clear_by_fingerprint(document_fingerprint)

        count = len(self._memory_cache)
        self._memory_cache.clear()
        self._embeddings.clear()
        self._embedding_matrix = None
        self._embedding_keys = []
        self._index_loaded = False

        if self.cache_dir.exists():
            for file in self.cache_dir.iterdir():
                if file.suffix == ".json":
                    file.unlink(missing_ok=True)

        logger.info("Cache cleared (%d entries)", count)
        return count

    def load_index_from_disk(self) -> None:
        """Load cached questions and embeddings from disk into memory.

        Scans the cache directory for ``*.embedding.json`` files, loads
        each one together with its companion answer file, and populates
        the in-memory structures. Guarded to run at most once.

        Legacy entries (schema_version < current, or 32-char MD5 keys) are
        silently skipped and purged.
        """
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
                if answer_data.get("schema_version", 0) < CACHE_SCHEMA_VERSION:
                    self._purge_entry(cache_key)
                    purged_count += 1
                    continue

                with open(entry.path, encoding="utf-8") as f:
                    embedding_data = json.load(f)

                self._memory_cache[cache_key] = answer_data
                self._embeddings[cache_key] = embedding_data.get("embedding", [])
                loaded_count += 1
            except (json.JSONDecodeError, KeyError, OSError):
                continue

        self._rebuild_index()
        self._index_loaded = True

        if purged_count:
            logger.info("Purged %d legacy cache entries", purged_count)
        logger.info("Loaded %d cached entries from disk", loaded_count)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_cache_key(question: str, document_fingerprint: str) -> str:
        """Generate a SHA-256 hash key from normalised question and document fingerprint.

        Args:
            question: The user's question.
            document_fingerprint: SHA-256 hex digest of the document set.

        Returns:
            A 64-character hexadecimal SHA-256 digest.
        """
        normalised = _normalise_question(question)
        raw = normalised + "|" + document_fingerprint
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _rebuild_index(self) -> None:
        """Rebuild the dense NumPy matrix from all cached embeddings."""
        if not self._embeddings:
            self._embedding_keys = []
            self._embedding_matrix = None
            return

        keys = list(self._embeddings.keys())
        matrix = np.array(
            [self._embeddings[k] for k in keys],
            dtype=np.float32,
        )
        self._embedding_keys = keys
        self._embedding_matrix = matrix

    def _semantic_lookup(
        self,
        question: str,
        embeddings_model: Embeddings,
        threshold: float | None = None,
    ) -> tuple[dict[str, Any] | None, float]:
        """Tier 3: find the most semantically similar cached question."""
        self.load_index_from_disk()

        effective_threshold = threshold if threshold is not None else self.threshold

        if self._embedding_matrix is None or len(self._embedding_keys) == 0:
            return None, 0.0

        normalised = _normalise_question(question)
        query_embedding = np.asarray(
            embeddings_model.embed_query(normalised),
            dtype=np.float32,
        )

        # Vectorised cosine similarity: (M · q) / (‖rows‖ · ‖q‖)
        query_norm = float(np.linalg.norm(query_embedding)) + 1e-12
        row_norms = np.linalg.norm(self._embedding_matrix, axis=1) + 1e-12
        similarities = (self._embedding_matrix @ query_embedding) / (row_norms * query_norm)

        best_index = int(np.argmax(similarities))
        best_score = float(similarities[best_index])
        best_key = self._embedding_keys[best_index]

        logger.debug(
            "Semantic search: best match score=%.4f (threshold=%.2f)",
            best_score,
            effective_threshold,
        )

        if best_score >= effective_threshold:
            logger.debug("Cache HIT (Tier 3 — semantic) for key %s", best_key[:8])
            return self._memory_cache[best_key], best_score

        return None, 0.0

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

        if to_remove:
            self._rebuild_index()
            logger.info(
                "Cleared %d cache entries for fingerprint %s",
                len(to_remove),
                fingerprint[:8],
            )

        return len(to_remove)
