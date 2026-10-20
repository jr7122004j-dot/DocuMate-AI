"""Logging configuration for DocuMate-AI.

Provides a unified log format and log level management across all modules.
"""

from __future__ import annotations

import logging
import sys

DEFAULT_LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: str | int = "INFO",
    log_format: str = DEFAULT_LOG_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> None:
    """Configure standard logging for the DocuMate-AI application.

    Idempotent: Reconfigures existing handlers instead of stacking duplicate handlers.

    Args:
        level: Log level name (e.g. ``"DEBUG"``, ``"INFO"``) or integer constant.
        log_format: Format string for log messages.
        date_format: Format string for timestamp display.
    """
    numeric_level: int
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), logging.INFO)
    else:
        numeric_level = level

    root_logger = logging.getLogger()
    app_logger = logging.getLogger("documate_ai")

    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # If root logger has no handlers, create one pointing to stdout
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    else:
        for existing_handler in root_logger.handlers:
            existing_handler.setFormatter(formatter)

    root_logger.setLevel(numeric_level)
    app_logger.setLevel(numeric_level)


def set_debug_mode(enabled: bool) -> None:
    """Switch the application logging level dynamically.

    Args:
        enabled: If True, set level to ``DEBUG``; otherwise, set to ``INFO``.
    """
    new_level = logging.DEBUG if enabled else logging.INFO
    configure_logging(level=new_level)
