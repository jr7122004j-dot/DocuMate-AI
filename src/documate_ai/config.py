"""Centralised configuration and settings management for DocuMate-AI.

Provides a typed, validated ``Settings`` object backed by ``pydantic-settings``
with environment variable override support (prefixed with ``DOCUMATE_``) and
``.env`` file loading.

Also retains module-level constants and prompt templates for backwards
compatibility and shared access across modules.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self


class Settings(BaseSettings):
    """Application settings schema and environment variable loader.

    All properties can be overridden via environment variables prefixed with
    ``DOCUMATE_`` (e.g. ``DOCUMATE_LLM_MODEL_NAME=llama3.2``) or through
    a local ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_prefix="DOCUMATE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------------------------------------------------------------------------
    # Model defaults & Ollama
    # ---------------------------------------------------------------------------
    embedding_model_name: str = "all-minilm:l6-v2"
    """Ollama embedding model used to convert text into vector representations."""

    llm_model_name: str = "phi4-mini"
    """Ollama chat model used for generating answers."""

    ollama_base_url: str = "http://localhost:11434"
    """Base URL of the Ollama server."""

    llm_temperature: float = 0.15
    """Sampling temperature — lower values produce more deterministic output."""

    llm_top_k: int = 10
    """Number of highest-probability tokens considered during sampling."""

    llm_top_p: float = 0.9
    """Nucleus-sampling cumulative probability mass."""

    llm_repeat_penalty: float = 1.1
    """Multiplicative penalty applied to tokens that have already appeared."""

    llm_num_ctx: int = 4096
    """Context window size (number of tokens) configured for the chat model."""

    # ---------------------------------------------------------------------------
    # Document processing & ingestion limits
    # ---------------------------------------------------------------------------
    chunk_size: int = 500
    """Maximum number of characters per document chunk."""

    chunk_overlap: int = 100
    """Number of overlapping characters between consecutive chunks."""

    retriever_result_count: int = 3
    """Number of most-similar chunks to retrieve per query."""

    max_pdf_pages: int = 500
    """Maximum allowed number of pages per uploaded PDF."""

    max_upload_size_mb: int = 50
    """Maximum allowed upload file size in megabytes."""

    # ---------------------------------------------------------------------------
    # Persistence paths
    # ---------------------------------------------------------------------------
    data_dir: Path = Path("./data")
    """Base directory for storing persistent application data."""

    cache_dir: Path | None = None
    """Directory where cached answers are stored on disk (resolved relative to data_dir)."""

    chroma_dir: Path | None = None
    """Directory where ChromaDB vector data is stored (resolved relative to data_dir)."""

    # ---------------------------------------------------------------------------
    # Semantic cache & UI threshold
    # ---------------------------------------------------------------------------
    similarity_threshold: float = 0.85
    """Default minimum cosine similarity score to consider a cached answer a match."""

    similarity_slider_min: float = 0.50
    """Minimum value the user can set the similarity threshold slider to."""

    similarity_slider_max: float = 0.99
    """Maximum value the user can set the similarity threshold slider to."""

    similarity_slider_step: float = 0.01
    """Step size of the similarity threshold slider."""

    # ---------------------------------------------------------------------------
    # Chat history, streaming & UI
    # ---------------------------------------------------------------------------
    chat_history_window: int = 5
    """Number of previous QA turns to include in conversational retrieval."""

    stream_render_interval_seconds: float = 0.03
    """Minimum time between Streamlit UI updates during token streaming."""

    multi_question_stream_chunk_size: int = 200
    """Character chunk size when streaming multi-question responses."""

    source_content_preview_length: int = 300
    """Maximum characters shown in source document preview expanders."""

    # ---------------------------------------------------------------------------
    # Question-detection heuristics
    # ---------------------------------------------------------------------------
    question_overlap_threshold: float = 0.5
    """Fraction of shared meaningful words to detect question repetition."""

    min_meaningful_word_length: int = 3
    """Minimum word length to be considered meaningful in overlap detection."""

    # ---------------------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------------------
    log_level: str = "INFO"
    """Default logging level (e.g. DEBUG, INFO, WARNING, ERROR)."""

    @model_validator(mode="after")
    def _validate_and_resolve_paths(self) -> Self:
        """Validate configuration invariants and resolve data directory paths."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than "
                f"chunk_size ({self.chunk_size})"
            )

        if not (0.0 < self.similarity_threshold <= 1.0):
            raise ValueError(
                f"similarity_threshold ({self.similarity_threshold}) must be between 0 and 1"
            )

        if self.similarity_slider_min >= self.similarity_slider_max:
            raise ValueError(
                f"similarity_slider_min ({self.similarity_slider_min}) must be less than "
                f"similarity_slider_max ({self.similarity_slider_max})"
            )

        min_threshold = self.similarity_slider_min
        max_threshold = self.similarity_slider_max
        if not (min_threshold <= self.similarity_threshold <= max_threshold):
            raise ValueError(
                f"similarity_threshold ({self.similarity_threshold}) must be between "
                f"similarity_slider_min ({self.similarity_slider_min}) and "
                f"similarity_slider_max ({self.similarity_slider_max})"
            )

        # Resolve cache and chroma paths relative to data_dir
        if self.cache_dir is None:
            self.cache_dir = self.data_dir / "cache"
        elif not self.cache_dir.is_absolute():
            self.cache_dir = self.data_dir / self.cache_dir

        if self.chroma_dir is None:
            self.chroma_dir = self.data_dir / "chroma_db"
        elif not self.chroma_dir.is_absolute():
            self.chroma_dir = self.data_dir / self.chroma_dir

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application ``Settings`` singleton."""
    return Settings()


# ---------------------------------------------------------------------------
# Sentinel string & Prompt templates
# ---------------------------------------------------------------------------

NO_INFO_RESPONSE: str = "I don't have enough information to answer this question."
"""Deterministic sentinel string emitted when context lacks sufficient information."""

QA_SYSTEM_PROMPT: str = (
    "You are a helpful assistant that answers questions based strictly on the provided context.\n"
    f"If the answer is not in the context, reply exactly: {NO_INFO_RESPONSE}\n"
    "Do not extrapolate or speculate beyond what is directly stated."
)
"""System prompt for ChatPromptTemplate grounding."""

QA_HUMAN_PROMPT: str = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
"""Human message template for document QA queries."""

QA_PROMPT_TEMPLATE: str = f"""Answer questions about documents based ONLY on the following context:
{{context}}
If the answer is not in the context, reply exactly: {NO_INFO_RESPONSE}
Question: {{question}}
Answer:"""
"""Combined prompt grounding the LLM to answer only from retrieved context."""

CONDENSE_QUESTION_SYSTEM_PROMPT: str = (
    "Given the conversation history and a follow-up question, "
    "rephrase the follow-up question to be a standalone question."
)
"""System prompt for question condensation."""

CONDENSE_QUESTION_HUMAN_PROMPT: str = (
    "Chat History:\n{chat_history}\nFollow-Up Question: {question}\nStandalone Question:"
)
"""Human message template for question condensation."""

CONDENSE_QUESTION_TEMPLATE: str = """
Given the following conversation and a follow-up question,
rephrase the follow-up question to be a standalone question.
Chat History: {chat_history}
Follow-Up Question: {question}
Standalone Question:"""
"""Prompt used to convert follow-up questions into standalone queries."""

# ---------------------------------------------------------------------------
# Question-detection heuristics constants
# ---------------------------------------------------------------------------

QUESTION_INDICATOR_WORDS: frozenset[str] = frozenset(
    {
        "what",
        "why",
        "how",
        "when",
        "where",
        "who",
        "which",
        "whose",
        "whom",
        "is",
        "are",
        "was",
        "were",
        "will",
        "can",
        "could",
        "should",
        "would",
        "do",
        "does",
        "did",
    }
)
"""Words that indicate a fragment is likely a question."""

COMMON_STOP_WORDS: frozenset[str] = frozenset(
    {
        "what",
        "when",
        "where",
        "who",
        "how",
        "does",
        "is",
        "are",
        "the",
        "and",
        "that",
        "this",
        "with",
        "for",
        "from",
        "have",
        "had",
    }
)
"""Stop words filtered out when computing question overlap scores."""

# ---------------------------------------------------------------------------
# Backward-compatibility module-level aliases
# ---------------------------------------------------------------------------

_settings = get_settings()

EMBEDDING_MODEL_NAME: str = _settings.embedding_model_name
LLM_MODEL_NAME: str = _settings.llm_model_name
LLM_TEMPERATURE: float = _settings.llm_temperature
LLM_TOP_K: int = _settings.llm_top_k
LLM_TOP_P: float = _settings.llm_top_p
LLM_REPEAT_PENALTY: float = _settings.llm_repeat_penalty
CHUNK_SIZE: int = _settings.chunk_size
CHUNK_OVERLAP: int = _settings.chunk_overlap
RETRIEVER_RESULT_COUNT: int = _settings.retriever_result_count
CACHE_DIRECTORY: Path = _settings.cache_dir or Path("./data/cache")
CHROMA_PERSIST_DIRECTORY: Path = _settings.chroma_dir or Path("./data/chroma_db")
DEFAULT_SIMILARITY_THRESHOLD: float = _settings.similarity_threshold
SIMILARITY_SLIDER_DEFAULT: float = _settings.similarity_threshold
SIMILARITY_SLIDER_MIN: float = _settings.similarity_slider_min
SIMILARITY_SLIDER_MAX: float = _settings.similarity_slider_max
SIMILARITY_SLIDER_STEP: float = _settings.similarity_slider_step
STREAM_RENDER_INTERVAL_SECONDS: float = _settings.stream_render_interval_seconds
MULTI_QUESTION_STREAM_CHUNK_SIZE: int = _settings.multi_question_stream_chunk_size
SOURCE_CONTENT_PREVIEW_LENGTH: int = _settings.source_content_preview_length
MIN_MEANINGFUL_WORD_LENGTH: int = _settings.min_meaningful_word_length
QUESTION_OVERLAP_THRESHOLD: float = _settings.question_overlap_threshold
