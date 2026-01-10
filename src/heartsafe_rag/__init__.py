"""HeartSafe RAG: Zero-Hallucination Heart Failure Decision Support System.

This package provides a RAG (Retrieval-Augmented Generation) system specifically
designed for heart failure guidelines, with a focus on accuracy and reliability.
"""

__version__ = "0.1.0"

# Import key components to make them available at package level
from heartsafe_rag.config import settings  # noqa: F401
from heartsafe_rag.exceptions import (  # noqa: F401
    HeartSafeError,
    ConfigurationError,
    DocumentProcessingError,
    ImageProcessingError,
    LLMError,
    ValidationError,
)

# Set up logging configuration
import logging
from heartsafe_rag.utils.logger import setup_logger

# Configure root logger
logger = setup_logger(__name__)

# Clean up namespace
del logging
