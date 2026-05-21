"""Unit tests for heart_safe_rag.ingestion — verifies PDF content extraction
for text, tables, images, and scanned pages."""
from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest
from langchain_core.documents import Document
from PIL import Image

from heartsafe_rag.exceptions import DocumentProcessingError, ImageProcessingError
from heartsafe_rag.ingestion import (
    _build_and_save_indices,
    _extract_images_from_doc,
    _extract_images_from_page,
    _extract_tables_from_page,
    _ocr_image,
    process_batch,
    process_pdf_page,
    process_single_pdf,
)

from .conftest import (
    MULTI_PAGE_EXPECTED,
    TABLE_EXPECTED_PHRASES,
    TEXT_ONLY_EXPECTED_PHRASES,
    UNICODE_EXPECTED_PHRASES,
)

# =========================================================================
# _extract_tables_from_page
# =========================================================================


class TestExtractTablesFromPage:
    def test_with_table(self, table_pdf: Path) -> None:
        result = _extract_tables_from_page(str(table_pdf), 0)
        assert len(result) == 1
        assert "[TABLE 1]" in result[0]
        for phrase in TABLE_EXPECTED_PHRASES:
            assert phrase in result[0]

    def test_no_table(self, text_only_pdf: Path) -> None:
        result = _extract_tables_from_page(str(text_only_pdf), 0)
        assert result == []

    def test_invalid_page(self, text_only_pdf: Path) -> None:
        result = _extract_tables_from_page(str(text_only_pdf), 999)
        assert result == []

    def test_logs_on_error(self, corrupt_pdf: Path) -> None:
        with patch("heartsafe_rag.ingestion.logger.warning") as mock_log:
            result = _extract_tables_from_page(str(corrupt_pdf), 0)
        assert result == []
        mock_log.assert_called_once()


# =========================================================================
# _extract_images_from_page / _extract_images_from_doc
# =========================================================================


class TestExtractImagesFromPage:
    def test_no_images_on_text_pdf(self, text_only_pdf: Path) -> None:
        result = _extract_images_from_page(str(text_only_pdf), 0)
        assert result == []

    @patch("heartsafe_rag.ingestion.HAS_PYTESSERACT", True)
    @patch("heartsafe_rag.ingestion.settings.OCR_ENABLED", True)
    @patch("heartsafe_rag.ingestion._ocr_image", return_value="CLASS I RECOMMENDATION")
    def test_with_image(self, mock_ocr: MagicMock, image_pdf: Path) -> None:
        result = _extract_images_from_page(str(image_pdf), 0)
        assert len(result) == 1
        assert "[IMAGE 1 OCR]" in result[0]
        assert "CLASS I RECOMMENDATION" in result[0]

    @patch("heartsafe_rag.ingestion.HAS_PYTESSERACT", True)
    @patch("heartsafe_rag.ingestion.settings.OCR_ENABLED", True)
    @patch("heartsafe_rag.ingestion._ocr_image", return_value="CLASS I RECOMMENDATION")
    def test_images_saved_to_output_dir(
        self, mock_ocr: MagicMock, image_pdf: Path, tmp_path: Path
    ) -> None:
        _extract_images_from_page(str(image_pdf), 0, tmp_path)
        img_dir = tmp_path / "images"
        assert img_dir.is_dir()
        images = list(img_dir.iterdir())
        assert len(images) >= 1
        assert images[0].suffix in {".png", ".jpg", ".jpeg"}

    @patch("heartsafe_rag.ingestion.HAS_PYTESSERACT", False)
    def test_skipped_when_pytesseract_unavailable(self, image_pdf: Path) -> None:
        result = _extract_images_from_page(str(image_pdf), 0)
        assert result == []

    @patch("heartsafe_rag.ingestion.HAS_PYTESSERACT", True)
    @patch("heartsafe_rag.ingestion.settings.OCR_ENABLED", False)
    def test_skipped_when_ocr_disabled(self, image_pdf: Path) -> None:
        result = _extract_images_from_page(str(image_pdf), 0)
        assert result == []

    @patch("heartsafe_rag.ingestion.HAS_PYTESSERACT", True)
    @patch("heartsafe_rag.ingestion.settings.OCR_ENABLED", True)
    @patch("heartsafe_rag.ingestion._ocr_image", side_effect=Exception("OCR crash"))
    def test_raises_image_processing_error_on_failure(
        self, mock_ocr: MagicMock, image_pdf: Path
    ) -> None:
        with pytest.raises(ImageProcessingError):
            _extract_images_from_page(str(image_pdf), 0)


class TestExtractImagesFromDoc:
    @patch("heartsafe_rag.ingestion.HAS_PYTESSERACT", True)
    @patch("heartsafe_rag.ingestion.settings.OCR_ENABLED", True)
    @patch("heartsafe_rag.ingestion._ocr_image", return_value="OCR TEXT")
    def test_from_open_doc(self, mock_ocr: MagicMock, image_pdf: Path) -> None:
        doc = fitz.open(image_pdf)
        try:
            result = _extract_images_from_doc(doc, 0)
            assert len(result) == 1
            assert "OCR TEXT" in result[0]
        finally:
            doc.close()


# =========================================================================
# _ocr_image / _ocr_page_image
# =========================================================================


@patch("heartsafe_rag.ingestion.HAS_PYTESSERACT", True)
@patch("heartsafe_rag.ingestion.settings.OCR_ENABLED", True)
@patch("heartsafe_rag.ingestion.pytesseract.image_to_string", return_value="mocked")
def test_ocr_image_with_pytesseract(mock_tess: MagicMock) -> None:
    img = Image.new("RGB", (10, 10))
    result = _ocr_image(img)
    assert result == "mocked"
    mock_tess.assert_called_once()


@patch("heartsafe_rag.ingestion.HAS_PYTESSERACT", False)
def test_ocr_image_returns_empty_without_pytesseract() -> None:
    img = Image.new("RGB", (10, 10))
    assert _ocr_image(img) == ""


@patch("heartsafe_rag.ingestion.HAS_PYTESSERACT", True)
@patch("heartsafe_rag.ingestion.settings.OCR_ENABLED", False)
def test_ocr_image_returns_empty_when_disabled() -> None:
    img = Image.new("RGB", (10, 10))
    assert _ocr_image(img) == ""


# =========================================================================
# process_pdf_page
# =========================================================================


class TestProcessPdfPage:
    def test_text_content(self, text_only_pdf: Path) -> None:
        result = process_pdf_page(str(text_only_pdf), 0)
        for phrase in TEXT_ONLY_EXPECTED_PHRASES:
            assert phrase in result

    def test_invalid_page_returns_empty_string(self, text_only_pdf: Path) -> None:
        assert process_pdf_page(str(text_only_pdf), 999) == ""

    def test_corrupt_pdf_returns_empty_string(self, corrupt_pdf: Path) -> None:
        assert process_pdf_page(str(corrupt_pdf), 0) == ""

    def test_text_and_table(self, table_pdf: Path) -> None:
        result = process_pdf_page(str(table_pdf), 0)
        for phrase in TABLE_EXPECTED_PHRASES:
            assert phrase in result

    def test_unicode_preserved(self, unicode_pdf: Path) -> None:
        result = process_pdf_page(str(unicode_pdf), 0)
        for phrase in UNICODE_EXPECTED_PHRASES:
            assert phrase in result

    @patch("heartsafe_rag.ingestion.HAS_PYTESSERACT", True)
    @patch("heartsafe_rag.ingestion.settings.OCR_ENABLED", True)
    @patch(
        "heartsafe_rag.ingestion._ocr_page_image",
        return_value="SCANNED DOCUMENT PAGE ONE",
    )
    def test_scanned_page_fallback(
        self, mock_ocr: MagicMock, scanned_pdf: Path
    ) -> None:
        result = process_pdf_page(str(scanned_pdf), 0)
        assert "[SCANNED PAGE OCR]" in result
        assert "SCANNED DOCUMENT PAGE ONE" in result

    @patch("heartsafe_rag.ingestion.HAS_PYTESSERACT", False)
    def test_scanned_page_returns_text_without_ocr(self, scanned_pdf: Path) -> None:
        result = process_pdf_page(str(scanned_pdf), 0)
        assert result == ""

    @patch("heartsafe_rag.ingestion.HAS_PYTESSERACT", True)
    @patch("heartsafe_rag.ingestion.settings.OCR_ENABLED", True)
    @patch(
        "heartsafe_rag.ingestion._extract_images_from_doc",
        return_value=["\n[IMAGE 1 OCR]\ntext\n[/IMAGE]"],
    )
    def test_text_table_and_image(
        self, mock_img: MagicMock, text_table_image_pdf: Path
    ) -> None:
        result = process_pdf_page(str(text_table_image_pdf), 0)
        assert "Mixed content page begins" in result
        assert "Furosemide" in result
        assert "[IMAGE 1 OCR]" in result


# =========================================================================
# process_single_pdf
# =========================================================================


class TestProcessSinglePdf:
    @patch("heartsafe_rag.ingestion._build_and_save_indices")
    def test_creates_correct_number_of_documents(
        self, mock_build: MagicMock, multi_page_pdf: Path, tmp_path: Path
    ) -> None:
        process_single_pdf(multi_page_pdf, tmp_path)
        mock_build.assert_called_once()
        docs: list[Document] = mock_build.call_args[0][0]
        assert len(docs) == 3
        for i, doc in enumerate(docs):
            assert doc.metadata["page"] == i + 1
            assert doc.metadata["source"] == "multi_page.pdf"
            assert len(doc.page_content) > 0

    @patch("heartsafe_rag.ingestion._build_and_save_indices")
    def test_content_fidelity(
        self, mock_build: MagicMock, multi_page_pdf: Path, tmp_path: Path
    ) -> None:
        process_single_pdf(multi_page_pdf, tmp_path)
        docs: list[Document] = mock_build.call_args[0][0]
        for i, expected_phrase in enumerate(MULTI_PAGE_EXPECTED):
            assert expected_phrase in docs[i].page_content

    @patch("heartsafe_rag.ingestion._build_and_save_indices")
    def test_empty_pdf(
        self, mock_build: MagicMock, empty_pdf: Path, tmp_path: Path
    ) -> None:
        process_single_pdf(empty_pdf, tmp_path)
        docs: list[Document] = mock_build.call_args[0][0]
        assert len(docs) == 0

    def test_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.pdf"
        with pytest.raises(FileNotFoundError):
            process_single_pdf(missing, tmp_path)

    @patch("heartsafe_rag.ingestion._build_and_save_indices")
    def test_single_page_pdf(
        self, mock_build: MagicMock, text_only_pdf: Path, tmp_path: Path
    ) -> None:
        process_single_pdf(text_only_pdf, tmp_path)
        docs: list[Document] = mock_build.call_args[0][0]
        assert len(docs) == 1

    @patch("heartsafe_rag.ingestion._build_and_save_indices")
    @patch("heartsafe_rag.ingestion.logger.info")
    def test_logs_progress(
        self, mock_log: MagicMock, mock_build: MagicMock, multi_page_pdf: Path, tmp_path: Path
    ) -> None:
        process_single_pdf(multi_page_pdf, tmp_path)
        progress_calls = [c for c in mock_log.call_args_list if "Processing page" in str(c)]
        assert len(progress_calls) == 3

    def test_corrupt_pdf(self, corrupt_pdf: Path, tmp_path: Path) -> None:
        with pytest.raises(DocumentProcessingError):
            process_single_pdf(corrupt_pdf, tmp_path)

    def test_corrupt_pdf_not_raised_as_generic_exception(
        self, corrupt_pdf: Path, tmp_path: Path
    ) -> None:
        with pytest.raises((DocumentProcessingError,)) as exc_info:
            process_single_pdf(corrupt_pdf, tmp_path)
        assert type(exc_info.value) is DocumentProcessingError


# =========================================================================
# process_batch
# =========================================================================


class TestProcessBatch:
    @patch("heartsafe_rag.ingestion._build_and_save_indices")
    def test_multiple_pdfs(
        self,
        mock_build: MagicMock,
        text_only_pdf: Path,
        multi_page_pdf: Path,
        tmp_path: Path,
    ) -> None:
        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        (batch_dir / "a.pdf").write_bytes(text_only_pdf.read_bytes())
        (batch_dir / "b.pdf").write_bytes(multi_page_pdf.read_bytes())

        process_batch(batch_dir, tmp_path)
        docs: list[Document] = mock_build.call_args[0][0]
        assert len(docs) == 4  # 1 page + 3 pages

    @patch("heartsafe_rag.ingestion._build_and_save_indices")
    def test_mixed_valid_and_corrupt(
        self,
        mock_build: MagicMock,
        text_only_pdf: Path,
        corrupt_pdf: Path,
        tmp_path: Path,
    ) -> None:
        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        (batch_dir / "valid.pdf").write_bytes(text_only_pdf.read_bytes())
        (batch_dir / "bad.pdf").write_bytes(corrupt_pdf.read_bytes())

        process_batch(batch_dir, tmp_path)
        docs: list[Document] = mock_build.call_args[0][0]
        assert len(docs) == 1  # only valid PDF processed

    def test_empty_directory(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = process_batch(empty_dir, tmp_path)
        assert result is None

    def test_no_pdf_directory(self, tmp_path: Path) -> None:
        no_pdf_dir = tmp_path / "nopdf"
        no_pdf_dir.mkdir()
        (no_pdf_dir / "notes.txt").write_text("not a pdf")
        result = process_batch(no_pdf_dir, tmp_path)
        assert result is None

    def test_missing_directory(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            process_batch(missing, tmp_path)

    @patch("heartsafe_rag.ingestion._build_and_save_indices")
    @patch("heartsafe_rag.ingestion.logger.warning")
    def test_logs_warning_for_empty_dir(
        self,
        mock_warn: MagicMock,
        mock_build: MagicMock,
        tmp_path: Path,
    ) -> None:
        empty = tmp_path / "empty_dir"
        empty.mkdir()
        process_batch(empty, tmp_path)
        warning_calls = [c for c in mock_warn.call_args_list if "No PDF files" in str(c)]
        assert len(warning_calls) == 1


# =========================================================================
# _build_and_save_indices
# =========================================================================


class TestBuildAndSaveIndices:
    @patch("heartsafe_rag.ingestion.HuggingFaceEmbeddings")
    @patch("heartsafe_rag.ingestion.FAISS")
    def test_creates_output_files(
        self,
        mock_faiss: MagicMock,
        mock_embeddings: MagicMock,
        tmp_path: Path,
    ) -> None:
        docs = [Document(page_content="test text for bm25 indexing", metadata={"source": "test.pdf"})]
        _build_and_save_indices(docs, tmp_path)
        chunks_path = tmp_path / "chunks.pkl"
        assert chunks_path.exists()
        with chunks_path.open("rb") as f:
            chunks = pickle.load(f)
        assert len(chunks) > 0
        assert chunks[0].metadata["source"] == "test.pdf"
        # BM25 index should also be saved
        assert (tmp_path / "bm25_index.pkl").exists()

    @patch("heartsafe_rag.ingestion.HuggingFaceEmbeddings")
    @patch("heartsafe_rag.ingestion.FAISS")
    def test_creates_output_dir_if_missing(
        self,
        mock_faiss: MagicMock,
        mock_embeddings: MagicMock,
        tmp_path: Path,
    ) -> None:
        nested = tmp_path / "a" / "b" / "c"
        docs = [Document(page_content="test text for bm25", metadata={"source": "test.pdf"})]
        _build_and_save_indices(docs, nested)
        assert nested.is_dir()
        assert (nested / "chunks.pkl").exists()
