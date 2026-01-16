"""Logging configuration for HeartSafe RAG.

This module provides a centralized logger with consistent formatting and log levels.
"""

import logging
import sys
from pathlib import Path

# Import settings to get the environment and log level
from heartsafe_rag.config import settings


class ContextFilter(logging.Filter):
    """Custom filter to add contextual information to log records.

    This ensures the 'environment' field is available in every log record,
    even for third-party loggers that don't know about this field.
    """

    def __init__(self, environment: str = "unknown"):
        super().__init__()
        self.environment = environment

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "environment"):
            record.environment = self.environment
        return True


def setup_logger(
    name: str,
    log_level: str = settings.LOG_LEVEL,
    log_file: Path | None = settings.LOG_FILE,
) -> logging.Logger:
    """Configure and return a logger with specified settings.

    Args:
        name: Name of the logger (usually __name__).
        log_level: Logging level (defaults to settings.LOG_LEVEL).
        log_file: Optional path to a log file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Convert string log level to logging constant
    log_level_numeric = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(log_level_numeric)

    # Prevent adding multiple handlers if logger is already configured
    if logger.handlers:
        return logger

    # --- CRITICAL FIX: Initialize and Add the Filter ---
    env_filter = ContextFilter(environment=settings.ENVIRONMENT)
    logger.addFilter(env_filter)
    # ---------------------------------------------------

    # Formatter including the environment field
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(environment)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(env_filter)  # Add filter to handler as well for safety
    logger.addHandler(console_handler)

    # File handler if log file is specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(env_filter)  # Add filter to handler
        logger.addHandler(file_handler)

    return logger


# Default logger instance
logger = setup_logger("heartsafe_rag")
