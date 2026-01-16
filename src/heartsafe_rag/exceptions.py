"""Custom exceptions for the HeartSafe RAG project.

This module defines custom exceptions for different error scenarios in the application.
"""


class HeartSafeError(Exception):
    """Base exception for all HeartSafe RAG specific exceptions."""

    pass


class ConfigurationError(HeartSafeError):
    """Raised when there is an error in the application configuration."""

    pass


class DocumentProcessingError(HeartSafeError):
    """Raised when there is an error processing a document."""

    pass


class ImageProcessingError(DocumentProcessingError):
    """Raised when there is an error processing an image within a document."""

    pass


class LLMError(HeartSafeError):
    """Raised when there is an error communicating with the LLM service."""

    pass


class ValidationError(HeartSafeError):
    """Raised when input validation fails."""

    pass
