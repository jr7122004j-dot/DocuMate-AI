"""Basic smoke tests for the DocuMate-AI package.

These tests verify that modules can be imported and core classes
can be instantiated without errors.
"""

from documate_ai import __version__
from documate_ai.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
)


def test_version() -> None:
    """Package version should be set to 1.0.0."""
    assert __version__ == "1.0.0"


def test_config_values_are_sane() -> None:
    """Config constants should have reasonable values."""
    assert CHUNK_SIZE > 0
    assert CHUNK_OVERLAP > 0
    assert CHUNK_OVERLAP < CHUNK_SIZE
    assert isinstance(EMBEDDING_MODEL_NAME, str)
    assert isinstance(LLM_MODEL_NAME, str)


def test_cache_manager_creation() -> None:
    """CacheManager should instantiate with defaults."""
    from documate_ai.cache_manager import CacheManager

    cm = CacheManager()
    assert cm.threshold == 0.85
    assert cm._memory_cache == {}


def test_cache_key_deterministic() -> None:
    """Same inputs should produce the same SHA-256 cache key."""
    from documate_ai.cache_manager import CacheManager

    fingerprint = "a" * 64
    key1 = CacheManager._make_cache_key("what is revenue", fingerprint)
    key2 = CacheManager._make_cache_key("what is revenue", fingerprint)
    assert key1 == key2
    assert len(key1) == 64  # SHA-256 hex digest is always 64 chars


def test_cache_key_differs_for_different_docs() -> None:
    """Different document fingerprints should produce different cache keys."""
    from documate_ai.cache_manager import CacheManager

    fp1 = "a" * 64
    fp2 = "b" * 64
    key1 = CacheManager._make_cache_key("what is revenue", fp1)
    key2 = CacheManager._make_cache_key("what is revenue", fp2)
    assert key1 != key2
