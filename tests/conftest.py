"""Pytest configuration that auto-generates a minimal test index for CI."""

import io
import pickle
from pathlib import Path

import fitz
import pytest
from langchain_core.documents import Document
from PIL import Image, ImageDraw
from pypdf import PdfWriter

# Paths that tests depend on
INDEX_DIR = Path("data/vector_store")
CHUNKS_PATH = INDEX_DIR / "chunks.pkl"
FAISS_PATH = INDEX_DIR / "index.faiss"


def _create_minimal_test_index() -> None:
    """Create a minimal FAISS + chunks index for CI testing."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if CHUNKS_PATH.exists() and FAISS_PATH.exists():
        return  # Already exists

    docs = [
        Document(
            page_content="Heart failure is a clinical syndrome resulting from structural or functional "
            "impairment of ventricular filling or ejection of blood. The diagnosis requires "
            "clinical assessment, natriuretic peptide testing, and echocardiography.",
            metadata={"source": "2022-AHA-HF-Guideline.pdf", "page": 1, "guideline_year": 2022},
        ),
        Document(
            page_content="GDMT for HFrEF includes four medication classes: ARNi/ACEi/ARB, "
            "beta blockers, MRAs, and SGLT2i. These should be initiated sequentially.",
            metadata={"source": "2022-AHA-HF-Guideline.pdf", "page": 2, "guideline_year": 2022},
        ),
        Document(
            page_content="Lasix (furosemide) is a loop diuretic used for volume management "
            "in heart failure. Typical starting dose is 40 mg once daily.",
            metadata={"source": "2022-AHA-HF-Guideline.pdf", "page": 3, "guideline_year": 2022},
        ),
    ]

    with CHUNKS_PATH.open("wb") as f:
        pickle.dump(docs, f)

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: PLC0415
        from langchain_community.vectorstores import FAISS  # noqa: PLC0415

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(str(INDEX_DIR))
    except Exception:
        pass  # FAISS creation may fail in CI without torch, chunks still useful


def pytest_configure(config) -> None:  # noqa: ARG001
    _create_minimal_test_index()


# ---------------------------------------------------------------------------
# Synthetic PDF fixtures for ingestion tests
# ---------------------------------------------------------------------------


@pytest.fixture
def text_only_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "text_only.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Heart failure is a clinical syndrome resulting from structural or functional "
        "impairment of ventricular filling or ejection of blood.\n\n"
        "The diagnosis requires clinical assessment, natriuretic peptide testing, "
        "and echocardiography.\n\n"
        "GDMT for HFrEF includes four medication classes: ARNi/ACEi/ARB, "
        "beta blockers, MRAs, and SGLT2i.",
        fontsize=11,
    )
    doc.save(str(path))
    doc.close()
    return path


TEXT_ONLY_EXPECTED_PHRASES = [
    "clinical syndrome",
    "ventricular filling",
    "natriuretic peptide testing",
    "ARNi/ACEi/ARB",
    "SGLT2i",
]


@pytest.fixture
def multi_page_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "multi_page.pdf"
    doc = fitz.open()
    for page_num in range(3):
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            f"PAGE {page_num + 1} CONTENT\n\n"
            f"This is page {page_num + 1} of the multi-page test document.",
            fontsize=11,
        )
    doc.save(str(path))
    doc.close()
    return path


MULTI_PAGE_EXPECTED = [
    "PAGE 1 CONTENT",
    "PAGE 2 CONTENT",
    "PAGE 3 CONTENT",
]


@pytest.fixture
def table_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "table.pdf"
    doc = fitz.open()
    page = doc.new_page()

    # Draw a 3x4 table grid
    cols = [72, 200, 350, 500]
    rows_ys = [100, 150, 200, 250, 300]
    for x in cols:
        page.draw_line(fitz.Point(x, rows_ys[0]), fitz.Point(x, rows_ys[-1]))
    for y in rows_ys:
        page.draw_line(fitz.Point(cols[0], y), fitz.Point(cols[-1], y))

    # Fill header row
    headers = ["Drug", "Class", "Dose"]
    for i, h in enumerate(headers):
        page.insert_text((cols[i] + 10, rows_ys[0] + 30), h, fontsize=10)
    # Fill data rows
    data = [
        ["Lisinopril", "ACEi", "5-40 mg"],
        ["Carvedilol", "Beta Blocker", "6.25-50 mg"],
        ["Furosemide", "Loop Diuretic", "20-80 mg"],
    ]
    for row_idx, row_data in enumerate(data):
        y = rows_ys[row_idx + 1] + 30
        for col_idx, cell in enumerate(row_data):
            page.insert_text((cols[col_idx] + 10, y), cell, fontsize=10)

    doc.save(str(path))
    doc.close()
    return path


TABLE_EXPECTED_PHRASES = [
    "Lisinopril", "ACEi", "5-40 mg",
    "Carvedilol", "Beta Blocker",
    "Furosemide", "Loop Diuretic",
]


def _create_image_with_text(text: str = "CLASS I RECOMMENDATION") -> bytes:
    img = Image.new("RGB", (400, 80), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 25), text, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def image_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "image.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Add some text
    page.insert_text((72, 50), "Text content before image.", fontsize=11)
    # Embed image with OCR-able text
    img_bytes = _create_image_with_text("CLASS I RECOMMENDATION")
    page.insert_image(fitz.Rect(72, 100, 472, 180), stream=img_bytes)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def scanned_pdf(tmp_path: Path) -> Path:
    """Image-only PDF (no text layer) — simulates a scanned document."""
    path = tmp_path / "scanned.pdf"
    doc = fitz.open()
    page = doc.new_page()
    img_bytes = _create_image_with_text("SCANNED DOCUMENT PAGE ONE")
    page.insert_image(page.rect, stream=img_bytes)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def text_table_image_pdf(tmp_path: Path) -> Path:
    """Single page with text + table + embedded image."""
    path = tmp_path / "mixed.pdf"
    doc = fitz.open()
    page = doc.new_page()

    page.insert_text((72, 50), "Mixed content page begins here.", fontsize=11)

    # Table
    cols = [72, 200, 350, 500]
    rows_ys = [100, 150, 200]
    for x in cols:
        page.draw_line(fitz.Point(x, rows_ys[0]), fitz.Point(x, rows_ys[-1]))
    for y in rows_ys:
        page.draw_line(fitz.Point(cols[0], y), fitz.Point(cols[-1], y))
    page.insert_text((82, 130), "Drug", fontsize=10)
    page.insert_text((210, 130), "Dose", fontsize=10)
    page.insert_text((360, 130), "Route", fontsize=10)
    page.insert_text((82, 180), "Furosemide", fontsize=10)
    page.insert_text((210, 180), "40mg", fontsize=10)
    page.insert_text((360, 180), "PO", fontsize=10)

    # Image
    img_bytes = _create_image_with_text("CLASS IIa RECOMMENDATION")
    page.insert_image(fitz.Rect(72, 300, 472, 380), stream=img_bytes)

    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def empty_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "empty.pdf"
    writer = PdfWriter()
    with path.open("wb") as f:
        writer.write(f)
    return path


@pytest.fixture
def corrupt_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"\x00\xFF\xFE\xFD\x00\x01\x02GARBAGE\xFF\xFF")
    return path


@pytest.fixture
def unicode_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "unicode.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Insuffisance cardiaque ñoño éàüö", fontsize=11)
    doc.save(str(path))
    doc.close()
    return path


UNICODE_EXPECTED_PHRASES = ["Insuffisance", "cardiaque", "ñoño", "éàüö"]


@pytest.fixture
def long_text_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "long.pdf"
    doc = fitz.open()
    page = doc.new_page()
    paragraph = (
        "Heart failure is a complex clinical syndrome. " * 50
    )
    page.insert_text((72, 72), paragraph, fontsize=11)
    doc.save(str(path))
    doc.close()
    return path
