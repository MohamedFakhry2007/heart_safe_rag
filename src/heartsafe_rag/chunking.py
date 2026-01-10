"""Chunking strategy module for HeartSafe RAG.

This module splits raw documents into semantic chunks optimized for embedding models.
It preserves critical metadata (Year, Source) and ensures no context is lost
due to arbitrary cuts.
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from heartsafe_rag.config import settings
from heartsafe_rag.utils.logger import logger
from heartsafe_rag.exceptions import DocumentProcessingError


class GuidelineChunker:
    """
    Handles the splitting of documents into fixed-size chunks with metadata preservation.
    """

    def __init__(
        self, 
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None
    ):
        """Initialize the chunker with specific size and overlap settings.

        Args:
            chunk_size: Max characters per chunk. Defaults to settings.CHUNK_SIZE.
            chunk_overlap: Overlap characters between chunks. Defaults to settings.CHUNK_OVERLAP.
            separators: List of separators to use for splitting. Defaults to settings.CHUNK_SEPARATORS.
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.separators = separators or settings.CHUNK_SEPARATORS
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            strip_whitespace=True,
            length_function=len,
        )
        logger.debug(f"Initialized Chunker: Size={self.chunk_size}, Overlap={self.chunk_overlap}")

    def _extract_metadata_from_source(self, source_path: str) -> Dict[str, Any]:
        """Extracts Guideline Year and Normalized Source Name from the filename.
        
        Logic:
            - Looks for 4-digit years (19xx or 20xx) in the filename.
            - Uses the filename as the source identifier.
            
        Args:
            source_path: The file path or name string.

        Returns:
            Dictionary containing 'guideline_year' and 'source'.
        """
        filename = os.path.basename(source_path)
        
        # Regex to find a year (e.g., 2022, 2024, 1999)
        # We assume guidelines are somewhat modern (1990-2029)
        year_match = re.search(r'(199\d|20[0-2]\d)', filename)
        year = int(year_match.group(0)) if year_match else None
        
        return {
            "source": filename,
            "guideline_year": year if year else "Unknown",
            "processed_type": "guideline_text"
        }

    @staticmethod
    def _clean_content(text: str) -> str:
        """Sanitizes text to remove PDF artifacts.

        - Replaces 3+ newlines with double newlines.
        - Strips leading/trailing whitespace.
        """
        # Collapse excessive vertical whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits a list of Documents into smaller chunks.

        Args:
            documents: The raw documents output from ingestion.

        Returns:
            The list of chunked documents ready for embedding.
        
        Raises:
            DocumentProcessingError: If splitting fails completely.
        """
        if not documents:
            logger.warning("No documents provided to chunker.")
            return []

        logger.info(f"Chunking {len(documents)} raw pages...")

        try:
            # 1. Pre-processing: Enrich Metadata & Clean Text
            for doc in documents:
                # Extract year/source from the file path in metadata
                source = doc.metadata.get("source", "unknown_file")
                meta_update = self._extract_metadata_from_source(source)
                doc.metadata.update(meta_update)
                
                # Sanitize content
                doc.page_content = self._clean_content(doc.page_content)

            # 2. Perform Splitting
            chunks = self.splitter.split_documents(documents)

            # 3. Post-processing: Add Unique IDs
            final_chunks = []
            for i, chunk in enumerate(chunks):
                # Add a relative chunk ID (useful for debugging retrieval order)
                chunk.metadata["chunk_id"] = i
                final_chunks.append(chunk)

            logger.info(
                f"Chunking Complete. "
                f"Input Pages: {len(documents)} -> Output Chunks: {len(final_chunks)}"
            )
            return final_chunks

        except Exception as e:
            error_msg = f"Failed to chunk documents: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise DocumentProcessingError(error_msg) from e
