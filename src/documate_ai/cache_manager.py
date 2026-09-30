"""Semantic answer cache with 3-tier lookup.

Implements a three-tier caching strategy to avoid redundant LLM calls:
    Tier 1 — In-memory exact match via MD5 hash (instant)
    Tier 2 — On-disk exact match via JSON files (very fast)
    Tier 3 — Semantic match via cosine similarity of embeddings (fast)

If all tiers miss, the caller should invoke the LLM and then call
``save()`` to persist the new answer.

Unlike the old code, this module is **fully decoupled from Streamlit** —
no ``st.session_state`` references. All state is held inside the
``CacheManager`` instance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from documate_ai.config import (
    CACHE_DIRECTORY,
    DEFAULT_SIMILARITY_THRESHOLD,
)

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


@dataclass
class CachedAnswer:
    """A single cached question-answer pair with metadata."""

    question: str
    answer: str
    sources: list[dict]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    embedding: list[float] | None = None

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dictionary (excludes embedding)."""
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "timestamp": self.timestamp,
        }


class CacheManager:
    """Three-tier semantic answer cache.

    Attributes:
        cache_dir: Path to the on-disk cache directory.
        threshold: Minimum cosine similarity for a semantic cache hit.
    """

    def __init__(
        self,
        cache_dir: Path = CACHE_DIRECTORY,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.threshold = threshold

        # Tier 1: in-memory cache keyed by MD5 hash
        self._memory_cache: dict[str, dict] = {}

        # Tier 3: dense embedding matrix for fast vectorised cosine similarity
        self._embeddings: dict[str, list[float]] = {}
        self._embedding_matrix: np.ndarray | None = None
        self._embedding_keys: list[str] = []

        self._index_loaded: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(
        self,
        question: str,
        document_names: list[str],
        embeddings_model: Embeddings | None = None,
    ) -> tuple[dict | None, float]:
        """Look up a cached answer using the 3-tier strategy.

        Args:
            question: The user's question.
            document_names: Sorted list of loaded PDF filenames (part of
                the cache key so different document sets get separate caches).
            embeddings_model: Embedding model needed for Tier 3 semantic
                lookup. If ``None``, only Tier 1 and 2 are checked.

        Returns:
            A tuple ``(cached_entry, similarity_score)``. An exact match
            returns a score of ``1.0``. Returns ``(None, 0.0)`` on miss.
        """
        cache_key = self._make_cache_key(question, document_names)

        # Tier 1: in-memory exact match
        if cache_key in self._memory_cache:
            logger.debug("Cache HIT (Tier 1 — memory) for key %s", cache_key[:8])
            return self._memory_cache[cache_key], 1.0

        # Tier 2: on-disk exact match
        answer_path = self.cache_dir / f"{cache_key}.json"
        embedding_path = self.cache_dir / f"{cache_key}.embedding.json"

        if answer_path.exists() and embedding_path.exists():
            try:
                with open(answer_path, "r", encoding="utf-8") as f:
                    answer_data = json.load(f)
                with open(embedding_path, "r", encoding="utf-8") as f:
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
            return self._semantic_lookup(question, embeddings_model)

        return None, 0.0

    def save(
        self,
        question: str,
        answer: str,
        sources: list[dict],
        document_names: list[str],
        embeddings_model: Embeddings,
    ) -> None:
        """Persist a new answer to both memory and disk caches.

        Args:
            question: The question that was asked.
            answer: The generated answer.
            sources: Source citation dicts with ``source`` and ``page`` keys.
            document_names: Sorted list of loaded PDF filenames.
            embeddings_model: The embedding model (to embed the question
                for future semantic matches).
        """
        cache_key = self._make_cache_key(question, document_names)
        question_embedding = embeddings_model.embed_query(question.lower().strip())

        cache_entry = CachedAnswer(
            question=question,
            answer=answer,
            sources=sources,
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

        with open(answer_path, "w", encoding="utf-8") as f:
            json.dump(entry_dict, f)

        with open(embedding_path, "w", encoding="utf-8") as f:
            json.dump({"embedding": question_embedding}, f)

        logger.info("Saved answer to cache (key=%s)", cache_key[:8])

    def clear(self) -> None:
        """Remove all cached answers from memory and disk."""
        self._memory_cache.clear()
        self._embeddings.clear()
        self._embedding_matrix = None
        self._embedding_keys = []
        self._index_loaded = False

        if self.cache_dir.exists():
            for file in self.cache_dir.iterdir():
                if file.suffix == ".json":
                    file.unlink(missing_ok=True)

        logger.info("Cache cleared")

    def load_index_from_disk(self) -> None:
        """Load cached questions and embeddings from disk into memory.

        Scans the cache directory for ``*.embedding.json`` files, loads
        each one together with its companion answer file, and populates
        the in-memory structures. Guarded to run at most once.
        """
        if self._index_loaded:
            return

        if not self.cache_dir.is_dir():
            self._index_loaded = True
            return

        loaded_count = 0
        for entry in os.scandir(self.cache_dir):
            if not entry.is_file() or not entry.name.endswith(".embedding.json"):
                continue

            cache_key = entry.name.replace(".embedding.json", "")
            answer_path = self.cache_dir / f"{cache_key}.json"

            if not answer_path.exists():
                continue

            try:
                with open(answer_path, "r", encoding="utf-8") as f:
                    answer_data = json.load(f)
                with open(entry.path, "r", encoding="utf-8") as f:
                    embedding_data = json.load(f)

                self._memory_cache[cache_key] = answer_data
                self._embeddings[cache_key] = embedding_data.get("embedding", [])
                loaded_count += 1
            except (json.JSONDecodeError, KeyError, OSError):
                continue

        self._rebuild_index()
        self._index_loaded = True
        logger.info("Loaded %d cached entries from disk", loaded_count)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_cache_key(question: str, document_names: list[str]) -> str:
        """Generate an MD5 hash key from the question and document names."""
        raw = question.lower().strip() + "|" + ",".join(sorted(document_names))
        return hashlib.md5(raw.encode()).hexdigest()

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
    ) -> tuple[dict | None, float]:
        """Tier 3: find the most semantically similar cached question."""
        self.load_index_from_disk()

        if self._embedding_matrix is None or len(self._embedding_keys) == 0:
            return None, 0.0

        query_embedding = np.asarray(
            embeddings_model.embed_query(question.lower().strip()),
            dtype=np.float32,
        )

        # Vectorised cosine similarity: (M · q) / (‖rows‖ · ‖q‖)
        query_norm = np.linalg.norm(query_embedding) + 1e-12
        row_norms = np.linalg.norm(self._embedding_matrix, axis=1) + 1e-12
        similarities = (self._embedding_matrix @ query_embedding) / (row_norms * query_norm)

        best_index = int(np.argmax(similarities))
        best_score = float(similarities[best_index])
        best_key = self._embedding_keys[best_index]

        logger.debug(
            "Semantic search: best match score=%.4f (threshold=%.2f)",
            best_score,
            self.threshold,
        )

        if best_score >= self.threshold:
            logger.debug("Cache HIT (Tier 3 — semantic) for key %s", best_key[:8])
            return self._memory_cache[best_key], best_score

        return None, 0.0
