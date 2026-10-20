"""Application-level exception hierarchy for DocuMate-AI.

All custom exceptions derive from ``DocuMateError``, allowing callers
to catch generic application errors or target specific failure modes.
"""

from __future__ import annotations


class DocuMateError(Exception):
    """Base exception for all DocuMate-AI errors."""

    def __init__(self, message: str = "An error occurred in DocuMate-AI.") -> None:
        """Initialize the error with a message.

        Args:
            message: Explanation of the error.
        """
        super().__init__(message)
        self.message = message


class OllamaUnavailableError(DocuMateError):
    """Raised when the Ollama service cannot be reached."""

    def __init__(
        self,
        message: str = "Ollama server is unavailable.",
        base_url: str | None = None,
    ) -> None:
        """Initialize OllamaUnavailableError.

        Args:
            message: Explanation of the error.
            base_url: Optional base URL of the unreachable Ollama server.
        """
        super().__init__(message)
        self.base_url = base_url


class ModelNotFoundError(DocuMateError):
    """Raised when a requested Ollama model is not installed or available."""

    def __init__(
        self,
        message: str = "The specified model was not found.",
        model_name: str | None = None,
    ) -> None:
        """Initialize ModelNotFoundError.

        Args:
            message: Explanation of the error.
            model_name: Name of the missing model.
        """
        super().__init__(message)
        self.model_name = model_name


class DocumentProcessingError(DocuMateError):
    """Raised when document loading, extraction, or chunking fails."""

    def __init__(
        self,
        message: str = "Failed to process document.",
        filename: str | None = None,
    ) -> None:
        """Initialize DocumentProcessingError.

        Args:
            message: Explanation of the error.
            filename: Name of the file that triggered the failure.
        """
        super().__init__(message)
        self.filename = filename


class EngineNotReadyError(DocuMateError):
    """Raised when attempting to query an engine before documents are ingested."""

    def __init__(
        self,
        message: str = "Engine is not ready. Ingest documents before querying.",
    ) -> None:
        """Initialize EngineNotReadyError.

        Args:
            message: Explanation of the error.
        """
        super().__init__(message)
