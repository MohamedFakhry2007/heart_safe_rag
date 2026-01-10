"""Configuration management for HeartSafe RAG.

This module handles loading and validating configuration from environment variables.
"""
import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from pydantic.types import DirectoryPath, FilePath

from heartsafe_rag.exceptions import ConfigurationError
from heartsafe_rag.utils.logger import logger, ContextFilter


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application settings
    ENVIRONMENT: str = Field("development", env="ENVIRONMENT")
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    LOG_FILE: Optional[Path] = Field(None, env="LOG_FILE")
    
    # Data directories
    DATA_DIR: DirectoryPath = Field(
        Path(__file__).parent.parent.parent / "data",
        env="DATA_DIR"
    )
    
    # LLM settings
    GROQ_API_KEY: str = Field(..., env="GROQ_API_KEY")
    LLM_MODEL: str = Field("meta-llama/llama-4-scout-17b-16e-instruct", env="LLM_MODEL")
    LLM_TEMPERATURE: float = Field(0.0, env="LLM_TEMPERATURE")
    
    # RAG settings - Chunking Configuration
    CHUNK_SIZE: int = Field(500, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(100, env="CHUNK_OVERLAP")
    
    # Embedding model settings
    EMBEDDING_MODEL: str = Field("sentence-transformers/all-MiniLM-L6-v2", 
                              env="EMBEDDING_MODEL")
    
    # Text Splitter Separators (Priority Order)
    # 1. Double newline (Paragraph break)
    # 2. Single newline (List item or line break)
    # 3. Period + Space (Sentence end)
    # 4. Space (Word break)
    # 5. Empty string (Character break - last resort)
    CHUNK_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v: str) -> str:
        """Validate LOG_LEVEL is a valid logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {', '.join(valid_levels)}")
        return v.upper()
    
    @validator("DATA_DIR", always=True)
    def validate_data_dir(cls, v: Path) -> Path:
        """Ensure DATA_DIR exists and is writable."""
        try:
            v.mkdir(parents=True, exist_ok=True)
            # Test write access
            test_file = v / ".write_test"
            test_file.touch()
            test_file.unlink()
            return v
        except (OSError, PermissionError) as e:
            raise ConfigurationError(f"Cannot access or create data directory {v}: {e}")


# Global settings instance
settings: Settings

try:
    settings = Settings()
    # Add the filter to inject environment into all future logs
    logger.addFilter(ContextFilter(environment=settings.ENVIRONMENT))
    logger.info(f"Loaded settings for environment: {settings.ENVIRONMENT}")
except Exception as e:
    logger.critical(f"Failed to load configuration: {e}")
    raise
