"""Tests for the chunking module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain.docstore.document import Document

from heartsafe_rag.chunking import GuidelineChunker


def test_chunking_logic():
    """Test the core chunking logic with a mock document."""
    # Mock Document mimicking a PDF page with an image description
    mock_text = (
        "1.2 DIAGNOSIS OF HEART FAILURE\n\n"
        "Heart failure is a complex clinical syndrome resulting from any structural or "
        "functional impairment of ventricular filling or ejection of blood. " * 5 + "\n\n[IMAGE DESCRIPTION START]\n"
        "Flowchart showing diagnostic algorithm: Step 1 assess clinical probability. "
        "Step 2 measure peptides. Step 3 echocardiography.\n"
        "[IMAGE DESCRIPTION END]"
    )

    doc = Document(page_content=mock_text, metadata={"source": "data/guidelines/2022-AHA-HF-Management.pdf"})

    print(f"--- Input Document Length: {len(mock_text)} chars ---")

    # Initialize chunker with test settings
    chunker = GuidelineChunker(chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""])

    chunks = chunker.split_documents([doc])

    print(f"--- Generated {len(chunks)} Chunks ---")

    for i, chunk in enumerate(chunks):
        print(f"\n[Chunk {i}] (Len: {len(chunk.page_content)})")
        print(f"Metadata: {chunk.metadata}")
        print(f"Content Preview: {chunk.page_content[:100]}...")

        # Verification Logic
        assert "2022" in str(chunk.metadata["guideline_year"]), "Year not extracted correctly!"
        assert len(chunk.page_content) <= 550, f"Chunk too large: {len(chunk.page_content)} chars"
        assert chunk.metadata["chunk_id"] == i, "Chunk ID not set correctly"
        assert chunk.metadata["processed_type"] == "guideline_text", "Processed type not set correctly"
        assert chunk.metadata["source"] == "2022-AHA-HF-Management.pdf", "Source not set correctly"


def test_empty_documents():
    """Test that an empty list of documents returns an empty list."""
    chunker = GuidelineChunker()
    result = chunker.split_documents([])
    assert result == []


def test_metadata_extraction():
    """Test the metadata extraction from filenames."""
    chunker = GuidelineChunker()

    # Test with year in filename
    meta = chunker._extract_metadata_from_source("data/guidelines/2022-AHA-HF-Management.pdf")
    assert meta["guideline_year"] == 2022
    assert meta["source"] == "2022-AHA-HF-Management.pdf"

    # Test without year in filename
    meta = chunker._extract_metadata_from_source("data/guidelines/AHA-HF-Management.pdf")
    assert meta["guideline_year"] == "Unknown"
    assert meta["source"] == "AHA-HF-Management.pdf"


def test_content_cleaning():
    """Test the content cleaning logic."""
    chunker = GuidelineChunker()

    # Test excessive newlines
    cleaned = chunker._clean_content("First line\n\n\n\nSecond line")
    assert "\n\n" in cleaned
    assert "\n\n\n" not in cleaned

    # Test leading/trailing whitespace
    cleaned = chunker._clean_content("  test  \n")
    assert cleaned == "test"


if __name__ == "__main__":
    test_chunking_logic()
