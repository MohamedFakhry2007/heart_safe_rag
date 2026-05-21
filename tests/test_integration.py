"""Integration tests for the HeartSafe RAG pipeline."""


from langchain_core.documents import Document

from heartsafe_rag.chunking import GuidelineChunker


def test_chunking_integration() -> None:
    """Test the full chunking pipeline with realistic guideline text."""
    text = (
        "2. DIAGNOSIS OF HEART FAILURE\n\n"
        "Heart failure is a clinical syndrome. "
        "Class of Recommendation I. Level of Evidence A. "
        "BNP testing is recommended."
    )

    doc = Document(page_content=text, metadata={"source": "2022-AHA-HF-Guideline.pdf"})
    chunker = GuidelineChunker(chunk_size=200, chunk_overlap=20)

    chunks = chunker.split_documents([doc])

    assert len(chunks) >= 1
    for chunk in chunks:
        assert "source" in chunk.metadata
        assert "guideline_year" in chunk.metadata
        assert "section" in chunk.metadata
        assert "section_path" in chunk.metadata
        assert "chunk_id" in chunk.metadata


def test_chunking_preserves_metadata() -> None:
    """Test that metadata is preserved through the chunking pipeline."""
    doc = Document(
        page_content="This is a test section about heart failure management.",
        metadata={"source": "2022-AHA-HF-Guideline.pdf"},
    )

    chunker = GuidelineChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.split_documents([doc])

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata["guideline_year"] == 2022
        assert chunk.metadata["source"] == "2022-AHA-HF-Guideline.pdf"
        assert chunk.metadata["processed_type"] == "guideline_text"


def test_chunking_empty_input() -> None:
    """Test that empty input returns empty list."""
    chunker = GuidelineChunker()
    result = chunker.split_documents([])
    assert result == []


def test_section_detection() -> None:
    """Test that section headers are detected and preserved in metadata."""
    text = (
        "3. PHARMACOLOGICAL MANAGEMENT\n\n"
        "3.1 Diuretics\n\n"
        "Loop diuretics are recommended for patients with fluid retention. "
        "3.2 ACE Inhibitors\n\n"
        "ACE inhibitors are recommended for all patients with HFrEF."
    )

    doc = Document(page_content=text, metadata={"source": "2022-AHA-HF-Guideline.pdf"})
    chunker = GuidelineChunker(chunk_size=500, chunk_overlap=50)

    chunks = chunker.split_documents([doc])

    assert len(chunks) > 0
    sections = {chunk.metadata["section"] for chunk in chunks}
    assert len(sections) > 0, "Expected at least one section to be detected"
