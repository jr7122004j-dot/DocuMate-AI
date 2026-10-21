"""Shared pytest fixtures and test infrastructure for DocuMate-AI.

Provides lightweight, offline test fakes for embeddings, language models,
file bytes, and UI containers so unit tests can execute deterministically
without requiring an active Ollama instance or Streamlit server.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

FIXTURES_DIR: Path = Path(__file__).parent / "fixtures"

_ollama_online: bool | None = None


def is_ollama_online(url: str = "http://localhost:11434") -> bool:
    """Check if a local Ollama server is reachable, caching the result.

    Args:
        url: Base URL of the Ollama server.

    Returns:
        True if the server responded, False otherwise.
    """
    global _ollama_online
    if _ollama_online is None:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "DocuMate-AI-Test"},
            )
            with urllib.request.urlopen(req, timeout=0.5):
                _ollama_online = True
        except (urllib.error.URLError, TimeoutError, OSError):
            _ollama_online = False
    return _ollama_online


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Auto-skip tests marked with @pytest.mark.ollama when Ollama is unreachable.

    Args:
        item: The pytest test item being executed.
    """
    if item.get_closest_marker("ollama") and not is_ollama_online():
        pytest.skip("Ollama server is not reachable at http://localhost:11434")


# ---------------------------------------------------------------------------
# Embedding Fakes
# ---------------------------------------------------------------------------


class ControllableEmbeddings(Embeddings):
    """Custom Embeddings class mapping known strings to predetermined vectors.

    Allows tests to control exact cosine similarities (e.g. orthogonal, identical,
    or arbitrary target scores) for semantic cache and retrieval tests.
    """

    def __init__(
        self,
        mapping: dict[str, list[float]] | None = None,
        default_vector: list[float] | None = None,
        size: int = 384,
    ) -> None:
        """Initialise controllable embeddings.

        Args:
            mapping: Optional dictionary mapping string text to float vectors.
            default_vector: Vector returned when text is not in mapping.
            size: Dimension of the vectors.
        """
        self.mapping: dict[str, list[float]] = dict(mapping or {})
        self.size: int = size
        self.default_vector: list[float] = default_vector or [0.0] * size

    def register(self, text: str, vector: list[float]) -> None:
        """Register a chosen vector for a specific input text.

        Args:
            text: Input string.
            vector: Target float vector of dimension self.size.
        """
        self.mapping[text] = vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts.

        Args:
            texts: List of input strings.

        Returns:
            List of float vectors.
        """
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        Args:
            text: Input query string.

        Returns:
            Float vector corresponding to the query text.
        """
        return self.mapping.get(text, self.default_vector)


@pytest.fixture
def fake_embeddings() -> DeterministicFakeEmbedding:
    """Fixture providing a deterministic 384-dimensional fake embedding model."""
    return DeterministicFakeEmbedding(size=384)


@pytest.fixture
def controllable_embeddings() -> ControllableEmbeddings:
    """Fixture providing a controllable embedding model with mutable text-to-vector mappings."""
    return ControllableEmbeddings(size=384)


# ---------------------------------------------------------------------------
# Chat Model Fakes
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_chat_model() -> GenericFakeChatModel:
    """Fixture providing a fake chat model that yields streaming tokens."""
    return GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(content="This is a test response from the fake chat model."),
                AIMessage(content="Follow-up response answering the second question."),
                AIMessage(content="Third response for conversational continuity."),
            ]
        )
    )


# ---------------------------------------------------------------------------
# Temporary Directory Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory for semantic cache storage."""
    cache_path = tmp_path / "cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


@pytest.fixture
def tmp_chroma_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory for ChromaDB storage."""
    chroma_path = tmp_path / "chroma_db"
    chroma_path.mkdir(parents=True, exist_ok=True)
    return chroma_path


# ---------------------------------------------------------------------------
# PDF Bytes Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Fixture providing raw bytes of a valid 3-page PDF with extractable text."""
    sample_file = FIXTURES_DIR / "sample.pdf"
    if not sample_file.exists():
        raise FileNotFoundError(f"Fixture PDF not found at {sample_file}")
    return sample_file.read_bytes()


@pytest.fixture
def blank_pdf_bytes() -> bytes:
    """Fixture providing raw bytes of a valid 1-page PDF containing no extractable text."""
    blank_file = FIXTURES_DIR / "blank.pdf"
    if not blank_file.exists():
        raise FileNotFoundError(f"Fixture PDF not found at {blank_file}")
    return blank_file.read_bytes()


@pytest.fixture
def corrupt_pdf_bytes() -> bytes:
    """Fixture providing raw bytes that simulate a corrupt or unreadable PDF file."""
    corrupt_file = FIXTURES_DIR / "corrupt.pdf"
    if corrupt_file.exists():
        return corrupt_file.read_bytes()
    return b"%PDF-1.4\n%Invalid-corrupted-stream-data\x00\xff\xfe"


# ---------------------------------------------------------------------------
# LangChain Document Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_documents() -> list[Document]:
    """Fixture providing a list of mock LangChain Document objects with source/page metadata."""
    return [
        Document(
            page_content="DocuMate-AI is an intelligent document analysis assistant.",
            metadata={"source": "report_a.pdf", "page": 1, "chunk_index": 0},
        ),
        Document(
            page_content="Semantic caching reduces latency by matching query embeddings.",
            metadata={"source": "report_a.pdf", "page": 2, "chunk_index": 1},
        ),
        Document(
            page_content="Pinpoint citations provide reference tracking for each response.",
            metadata={"source": "report_b.pdf", "page": 1, "chunk_index": 0},
        ),
    ]


# ---------------------------------------------------------------------------
# UI Container Fakes
# ---------------------------------------------------------------------------


class FakeContainer:
    """Simulates a Streamlit empty or delta container by recording markdown render calls."""

    def __init__(self) -> None:
        """Initialise FakeContainer with an empty list of markdown calls."""
        self.markdown_calls: list[str] = []

    def markdown(self, body: str, **kwargs: Any) -> None:
        """Record a markdown render call.

        Args:
            body: The rendered markdown text.
            **kwargs: Extra rendering options (ignored in test fake).
        """
        self.markdown_calls.append(body)

    def empty(self) -> FakeContainer:
        """Simulate container clearing."""
        return self

    @property
    def last_content(self) -> str | None:
        """Return the most recently rendered markdown content, or None if empty."""
        return self.markdown_calls[-1] if self.markdown_calls else None


@pytest.fixture
def fake_container() -> FakeContainer:
    """Fixture providing a fresh FakeContainer instance for stream rendering tests."""
    return FakeContainer()
