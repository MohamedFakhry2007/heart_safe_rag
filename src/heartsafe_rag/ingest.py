"""
Ingestion script for HeartSafe RAG.
Loads PDFs, chunks text, creates embeddings, and saves FAISS & BM25 indices.
"""
import pickle
import shutil
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from heartsafe_rag.config import settings
from heartsafe_rag.utils.logger import logger

def load_documents() -> List[Document]:
    """Load PDFs from the raw_pdfs directory."""
    pdf_path = settings.DATA_DIR / "raw_pdfs"
    if not pdf_path.exists():
        logger.error(f"PDF Directory not found: {pdf_path}")
        # Create it if it doesn't exist to prevent crash
        pdf_path.mkdir(parents=True, exist_ok=True)
        return []

    logger.info(f"Loading PDFs from {pdf_path}...")
    loader = DirectoryLoader(
        str(pdf_path),
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} documents.")
    return docs

def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks.")
    return chunks

def create_vector_db(chunks: List[Document]):
    """Create and save FAISS vector store."""
    logger.info("Creating Embeddings and Vector Store...")
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save locally
    save_path = settings.VECTOR_DB_PATH
    # FAISS save_local creates a folder, so we ensure parent exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    vectorstore.save_local(save_path)
    logger.info(f"Vector Store saved to {save_path}")

def create_bm25_index(chunks: List[Document]):
    """Create and save BM25 sparse index."""
    logger.info("Creating BM25 Index...")
    retriever = BM25Retriever.from_documents(chunks)
    
    # Save as pickle
    save_path = Path(settings.BM25_PATH)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "wb") as f:
        pickle.dump(retriever, f)
    logger.info(f"BM25 Index saved to {save_path}")

def main():
    try:
        docs = load_documents()
        if not docs:
            logger.warning("No documents found. Skipping ingestion.")
            return

        chunks = chunk_documents(docs)
        
        # Create Indices
        create_vector_db(chunks)
        create_bm25_index(chunks)
        
        logger.info("Ingestion complete successfully.")
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise e

if __name__ == "__main__":
    main()