"""Chunking strategy module for HeartSafe RAG.

This module splits raw documents into semantic chunks optimized for embedding models.
It preserves critical metadata (Year, Source) and ensures no context is lost
due to arbitrary cuts.
"""

import os
import re
from typing import Any

from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from heartsafe_rag.config import settings
from heartsafe_rag.exceptions import DocumentProcessingError
from heartsafe_rag.utils.logger import logger


class GuidelineChunker:
    """
    Handles the splitting of documents into fixed-size chunks with metadata preservation.
    """

    def __init__(
        self, chunk_size: int | None = None, chunk_overlap: int | None = None, separators: list[str] | None = None
    ):
        """Initialize the chunker with specific size and overlap settings."""
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        # Fixed attribute name to match Config
        self.separators = separators or settings.CHUNK_SEPARATORS

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            strip_whitespace=True,
            length_function=len,
        )
        logger.debug(f"Initialized Chunker: Size={self.chunk_size}, Overlap={self.chunk_overlap}")

    def _extract_metadata_from_source(self, source_path: str) -> dict[str, Any]:
        """Extracts Guideline Year and Normalized Source Name from the filename."""
        filename = os.path.basename(source_path)

        # Regex to find a year (e.g., 2022, 2024, 1999)
        year_match = re.search(r"(199\d|20[0-2]\d)", filename)
        year = int(year_match.group(0)) if year_match else None

        return {"source": filename, "guideline_year": year if year else "Unknown", "processed_type": "guideline_text"}

    def _clean_content(self, text: str) -> str:
        """Sanitizes text to remove PDF artifacts."""
        # Collapse excessive vertical whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Splits a list of Documents into smaller chunks."""
        if not documents:
            logger.warning("No documents provided to chunker.")
            return []

        logger.info(f"Chunking {len(documents)} raw pages...")

        try:
            for doc in documents:
                source = doc.metadata.get("source", "unknown_file")
                meta_update = self._extract_metadata_from_source(source)
                doc.metadata.update(meta_update)
                doc.page_content = self._clean_content(doc.page_content)

            chunks = self.splitter.split_documents(documents)

            final_chunks = []
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = i
                final_chunks.append(chunk)

            logger.info(f"Chunking Complete. Input Pages: {len(documents)} -> Output Chunks: {len(final_chunks)}")
            return final_chunks

        except Exception as e:
            error_msg = f"Failed to chunk documents: {e!s}"
            logger.error(error_msg, exc_info=True)
            raise DocumentProcessingError(error_msg) from e
