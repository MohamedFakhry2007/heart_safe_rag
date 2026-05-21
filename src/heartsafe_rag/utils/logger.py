"""Logging configuration for HeartSafe RAG.

Provides JSON structured logging with consistent fields across all modules.
"""
import json
import logging
import sys
from pathlib import Path

from heartsafe_rag.config import settings


class JsonFormatter(logging.Formatter):
    """JSON formatter that includes all standard fields plus optional extras."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, object] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "name": record.name,
            "level": record.levelname,
            "environment": getattr(record, "environment", "unknown"),
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


class ContextFilter(logging.Filter):
    def __init__(self, environment: str = "unknown"):
        super().__init__()
        self.environment = environment

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, 'environment'):
            record.environment = self.environment
        return True


def setup_logger(
    name: str,
    log_level: str = settings.LOG_LEVEL,
    log_file: Path | None = settings.LOG_FILE,
) -> logging.Logger:
    logger = logging.getLogger(name)
    log_level_numeric = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(log_level_numeric)

    if logger.handlers:
        return logger

    env_filter = ContextFilter(environment=settings.ENVIRONMENT)
    logger.addFilter(env_filter)

    formatter = JsonFormatter(datefmt='%Y-%m-%d %H:%M:%S')

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(env_filter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(env_filter)
        logger.addHandler(file_handler)

    return logger


# Default logger instance
logger = setup_logger("heartsafe_rag")
