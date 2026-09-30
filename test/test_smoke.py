"""Basic smoke tests for the DocuMate-AI package.

These tests verify that modules can be imported and core classes
can be instantiated without errors.
"""

from documate_ai import __version__
from documate_ai.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
)


def test_version():
    """Package version should be set."""
    assert __version__ == "0.1.0"


def test_config_values_are_sane():
    """Config constants should have reasonable values."""
    assert CHUNK_SIZE > 0
    assert CHUNK_OVERLAP > 0
    assert CHUNK_OVERLAP < CHUNK_SIZE
    assert isinstance(EMBEDDING_MODEL_NAME, str)
    assert isinstance(LLM_MODEL_NAME, str)


def test_cache_manager_creation():
    """CacheManager should instantiate with defaults."""
    from documate_ai.cache_manager import CacheManager

    cm = CacheManager()
    assert cm.threshold == 0.85
    assert cm._memory_cache == {}


def test_cache_key_deterministic():
    """Same inputs should produce the same MD5 cache key."""
    from documate_ai.cache_manager import CacheManager

    key1 = CacheManager._make_cache_key("what is revenue", ["report.pdf"])
    key2 = CacheManager._make_cache_key("what is revenue", ["report.pdf"])
    assert key1 == key2
    assert len(key1) == 32  # MD5 hex digest is always 32 chars


def test_cache_key_differs_for_different_docs():
    """Different document names should produce different cache keys."""
    from documate_ai.cache_manager import CacheManager

    key1 = CacheManager._make_cache_key("what is revenue", ["report.pdf"])
    key2 = CacheManager._make_cache_key("what is revenue", ["other.pdf"])
    assert key1 != key2
