import os
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic.types import DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # Application settings
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path | None = None

    # Data directories
    DATA_DIR: DirectoryPath = Path("data")

    # LLM settings
    # MyPy complains if required fields don't have defaults during instantiation.
    # We use '...' to mark it required, but handle the mypy error at the bottom.
    GROQ_API_KEY: SecretStr = Field(..., description="Groq API Key")
    LLM_MODEL: str = "meta-llama/llama-3.2-11b-vision-preview"
    LLM_TEMPERATURE: float = 0.0

    # RAG settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100

    # Embedding model
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Text Splitter Separators (Renamed to match chunking.py expectation)
    CHUNK_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {', '.join(valid_levels)}")
        return v.upper()

    @field_validator("DATA_DIR")
    @classmethod
    def validate_data_dir(cls, v: Path) -> Path:
        if not v.exists():
            os.makedirs(v, exist_ok=True)
        return v

# Initialize settings with GROQ_API_KEY from environment as SecretStr
settings = Settings(GROQ_API_KEY=SecretStr(os.getenv("GROQ_API_KEY", "")))
