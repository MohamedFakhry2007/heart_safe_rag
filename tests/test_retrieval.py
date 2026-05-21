from pathlib import Path

import pytest

from heartsafe_rag.config import settings
from heartsafe_rag.retrieval import HybridRetriever


@pytest.mark.skipif(
    not Path("data/vector_store/index.faiss").exists(),
    reason="FAISS index not found. Make sure to run the ingestion process first.",
)
def test_hybrid_retrieval_flow() -> None:
    retriever = HybridRetriever()

    query = "heart failure diagnosis"
    docs = retriever.retrieve(query)

    assert len(docs) > 0, "Retriever returned no results!"
    assert len(docs) <= settings.RERANK_TOP_K + 2, "Retriever returned too many results"

    first_doc = docs[0]
    assert "source" in first_doc.metadata
    assert "page" in first_doc.metadata
