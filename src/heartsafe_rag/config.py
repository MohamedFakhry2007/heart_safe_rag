import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import Field, SecretStr, field_validator
from pydantic.types import DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore")

    # Application settings
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path | None = None

    # Data directories
    # Validates that 'data' exists, creates it if not (via validator below)
    DATA_DIR: DirectoryPath = Path("data")

    # Paths for indices
    # We use 'vector_store' for the FAISS folder
    VECTOR_DB_PATH: Path = Path("data/vector_store")
    BM25_PATH: Path = Path("data/bm25_index.pkl")

    # Langfuse Tracing
    LANGFUSE_PUBLIC_KEY: str = Field(..., description="Langfuse Public Key")
    LANGFUSE_SECRET_KEY: SecretStr = Field(..., description="Langfuse Secret Key")
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # LLM settings
    # Pydantic will automatically read GROQ_API_KEY from .env
    GROQ_API_KEY: SecretStr = Field(..., description="Groq API Key")
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.0

    # RAG Ingestion settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    CHUNK_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]

    # Retrieval settings
    RETRIEVAL_K: int = 7
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

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


# Initialize settings.
# BaseSettings automatically reads from os.environ and .env file.
# We don't need to manually pass the key here unless we want to override it.
settings = Settings()
