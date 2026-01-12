"""Logging configuration for HeartSafe RAG.

This module provides a centralized logger with consistent formatting and log levels.
"""
import logging
import sys
from pathlib import Path
from typing import Optional


class ContextFilter(logging.Filter):
    """Custom filter to add contextual information to log records.

    This allows adding fields like 'environment' to every log message.
    """
    def __init__(self, environment: str = "unknown"):
        super().__init__()
        self.environment = environment

    def filter(self, record: logging.LogRecord) -> bool:
        record.environment = self.environment
        return True


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_file: Path | None = None,
) -> logging.Logger:
    """Configure and return a logger with specified settings.

    Args:
        name: Name of the logger (usually __name__).
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to a log file. If not provided, logs to stderr.

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

    # Updated formatter to include environment (if filter is added later)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(environment)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if log file is specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Default logger instance
logger = setup_logger(__name__)
