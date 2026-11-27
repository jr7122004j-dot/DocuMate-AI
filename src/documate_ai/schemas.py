"""Data transfer objects and schemas for DocuMate-AI.

Defines structured representations for citations, engine chat responses,
and execution metadata used across the engine, cache, UI, and future API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class UploadedPdf:
    """Represents an uploaded PDF file with its filename and raw byte content."""

    name: str
    data: bytes


@dataclass
class IngestReport:
    """Summary report detailing PDF ingestion, extraction, and chunking results."""

    files: list[str] = field(default_factory=list)
    pages: int = 0
    chunks: int = 0
    skipped: dict[str, str] = field(default_factory=dict)
    fingerprint: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a plain dictionary."""
        return asdict(self)


@dataclass
class Citation:
    """Structured citation referencing a specific page and snippet in a source document."""

    source: str
    page: int
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        """Convert the citation to a plain dictionary."""
        return asdict(self)


@dataclass
class ChatResponse:
    """Structured response from the QA engine with citations and execution metadata."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    from_cache: bool = False
    similarity: float | None = None
    matched_question: str | None = None
    standalone_question: str | None = None
    sub_questions: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the response to a plain dictionary."""
        return asdict(self)
