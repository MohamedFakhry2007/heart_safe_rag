"""Tests for the retrieval module."""
import pytest
from heartsafe_rag.retrieval import HybridRetriever
from heartsafe_rag.config import settings
from pathlib import Path

# Skip test if data isn't ingested yet
@pytest.mark.skipif(
    not Path("faiss_index/index.faiss").exists(),
    reason="FAISS index not found. Make sure to run the ingestion process first."
)
def test_hybrid_retrieval_flow():
    """Ensures we can load the index and retrieve results."""
    
    retriever_logic = HybridRetriever()
    ensemble = retriever_logic.get_retriever()
    
    # Test a known medical query (from your PDF)
    query = "heart failure diagnosis"
    docs = ensemble.invoke(query)
    
    # Assertions
    assert len(docs) > 0, "Retriever returned no results!"
    assert len(docs) <= 10, "Retriever returned too many results"
    
    # Check if metadata exists
    first_doc = docs[0]
    assert "source" in first_doc.metadata
    assert "page" in first_doc.metadata