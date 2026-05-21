"""Integration tests for the HeartSafe RAG pipeline."""

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from heartsafe_rag.chunking import GuidelineChunker
from heartsafe_rag.ingestion import process_batch, process_single_pdf

from .conftest import MULTI_PAGE_EXPECTED, TABLE_EXPECTED_PHRASES


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


# =========================================================================
# Full PDF → Documents → Chunks integration
# =========================================================================


class TestPdfToChunksPipeline:
    @patch("heartsafe_rag.ingestion.HuggingFaceEmbeddings")
    @patch("heartsafe_rag.ingestion.FAISS")
    def test_text_pdf_to_chunks(
        self,
        mock_faiss: MagicMock,
        mock_embeddings: MagicMock,
        text_only_pdf: Path,
        tmp_path: Path,
    ) -> None:
        process_single_pdf(text_only_pdf, tmp_path)
        chunks_path = tmp_path / "chunks.pkl"
        assert chunks_path.exists()
        with chunks_path.open("rb") as f:
            chunks: list[Document] = pickle.load(f)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata["source"] == "text_only.pdf"
            assert "guideline_year" in chunk.metadata
            assert "chunk_id" in chunk.metadata
            assert "section" in chunk.metadata
            assert "section_path" in chunk.metadata
            assert "processed_type" in chunk.metadata
            assert len(chunk.page_content) > 0

    @patch("heartsafe_rag.ingestion.HuggingFaceEmbeddings")
    @patch("heartsafe_rag.ingestion.FAISS")
    def test_multi_page_pdf_all_pages_in_chunks(
        self,
        mock_faiss: MagicMock,
        mock_embeddings: MagicMock,
        multi_page_pdf: Path,
        tmp_path: Path,
    ) -> None:
        process_single_pdf(multi_page_pdf, tmp_path)
        chunks_path = tmp_path / "chunks.pkl"
        with chunks_path.open("rb") as f:
            chunks: list[Document] = pickle.load(f)
        all_content = " ".join(c.page_content for c in chunks)
        for phrase in MULTI_PAGE_EXPECTED:
            assert phrase in all_content

    @patch("heartsafe_rag.ingestion.HuggingFaceEmbeddings")
    @patch("heartsafe_rag.ingestion.FAISS")
    def test_table_content_survives_chunking(
        self,
        mock_faiss: MagicMock,
        mock_embeddings: MagicMock,
        table_pdf: Path,
        tmp_path: Path,
    ) -> None:
        process_single_pdf(table_pdf, tmp_path)
        chunks_path = tmp_path / "chunks.pkl"
        with chunks_path.open("rb") as f:
            chunks: list[Document] = pickle.load(f)
        all_content = " ".join(c.page_content for c in chunks)
        for phrase in TABLE_EXPECTED_PHRASES:
            assert phrase in all_content

    @patch("heartsafe_rag.ingestion.HuggingFaceEmbeddings")
    @patch("heartsafe_rag.ingestion.FAISS")
    def test_batch_pipeline_aggregates_all_pdfs(
        self,
        mock_faiss: MagicMock,
        mock_embeddings: MagicMock,
        text_only_pdf: Path,
        multi_page_pdf: Path,
        tmp_path: Path,
    ) -> None:
        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        (batch_dir / "a.pdf").write_bytes(text_only_pdf.read_bytes())
        (batch_dir / "b.pdf").write_bytes(multi_page_pdf.read_bytes())
        process_batch(batch_dir, tmp_path)
        chunks_path = tmp_path / "chunks.pkl"
        assert chunks_path.exists()
        with chunks_path.open("rb") as f:
            chunks: list[Document] = pickle.load(f)
        assert len(chunks) > 0
