"""Pytest configuration that auto-generates a minimal test index for CI."""

import pickle
from pathlib import Path

from langchain_core.documents import Document

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
