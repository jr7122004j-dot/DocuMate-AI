"""Centralised configuration constants for the DocuMate-AI application.

All tunable parameters — model names, chunking settings, cache behaviour,
prompt templates, and heuristic thresholds — are collected here so that
they can be adjusted in a single place without touching business logic.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME: str = "all-minilm:l6-v2"
"""Ollama embedding model used to convert text into vector representations."""

LLM_MODEL_NAME: str = "phi4-mini"
"""Ollama chat model used for generating answers."""

LLM_TEMPERATURE: float = 0.15
"""Sampling temperature — lower values produce more deterministic output."""

LLM_TOP_K: int = 10
"""Number of highest-probability tokens considered during sampling."""

LLM_TOP_P: float = 0.9
"""Nucleus-sampling cumulative probability mass."""

LLM_REPEAT_PENALTY: float = 1.1
"""Multiplicative penalty applied to tokens that have already appeared."""

# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------

CHUNK_SIZE: int = 500
"""Maximum number of characters per document chunk."""

CHUNK_OVERLAP: int = 100
"""Number of overlapping characters between consecutive chunks."""

RETRIEVER_RESULT_COUNT: int = 3
"""Number of most-similar chunks to retrieve per query."""

# ---------------------------------------------------------------------------
# Persistence paths
# ---------------------------------------------------------------------------

CACHE_DIRECTORY: Path = Path("./cache")
"""Directory where cached answers and embeddings are stored on disk."""

CHROMA_PERSIST_DIRECTORY: Path = Path("./chroma_db")
"""Directory where ChromaDB stores its persistent vector data."""

# ---------------------------------------------------------------------------
# Semantic cache
# ---------------------------------------------------------------------------

DEFAULT_SIMILARITY_THRESHOLD: float = 0.85
"""Minimum cosine similarity score to consider a cached answer a match."""

SIMILARITY_SLIDER_DEFAULT: float = 0.95
"""Default position of the similarity threshold slider in the UI."""

SIMILARITY_SLIDER_MIN: float = 0.50
"""Minimum value the user can set the similarity threshold to."""

SIMILARITY_SLIDER_MAX: float = 0.99
"""Maximum value the user can set the similarity threshold to."""

SIMILARITY_SLIDER_STEP: float = 0.01
"""Step size of the similarity threshold slider."""

# ---------------------------------------------------------------------------
# Streaming / UI
# ---------------------------------------------------------------------------

STREAM_RENDER_INTERVAL_SECONDS: float = 0.03
"""Minimum time between Streamlit UI updates during token streaming."""

MULTI_QUESTION_STREAM_CHUNK_SIZE: int = 200
"""Character chunk size when streaming multi-question responses."""

SOURCE_CONTENT_PREVIEW_LENGTH: int = 300
"""Maximum characters shown in source document preview expanders."""

# ---------------------------------------------------------------------------
# Question-detection heuristics
# ---------------------------------------------------------------------------

QUESTION_INDICATOR_WORDS: frozenset[str] = frozenset({
    "what", "why", "how", "when", "where", "who", "which",
    "whose", "whom", "is", "are", "was", "were", "will",
    "can", "could", "should", "would", "do", "does", "did",
})
"""Words that indicate a fragment is likely a question."""

COMMON_STOP_WORDS: frozenset[str] = frozenset({
    "what", "when", "where", "who", "how", "does", "is", "are",
    "the", "and", "that", "this", "with", "for", "from", "have", "had",
})
"""Stop words filtered out when computing question overlap scores."""

MIN_MEANINGFUL_WORD_LENGTH: int = 3
"""Minimum word length to be considered meaningful in overlap detection."""

QUESTION_OVERLAP_THRESHOLD: float = 0.5
"""Fraction of shared meaningful words to detect question repetition."""

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

CONDENSE_QUESTION_TEMPLATE: str = """
Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
Chat History: {chat_history}
Follow-Up Question: {question}
Standalone Question:"""
"""Prompt used to convert follow-up questions into standalone queries."""

QA_PROMPT_TEMPLATE: str = """
Answer questions about documents based ONLY on the following context:
{context}
Question: {question}
Answer:"""
"""Prompt that grounds the LLM to answer only from retrieved context."""
