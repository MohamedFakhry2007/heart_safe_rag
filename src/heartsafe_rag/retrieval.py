"""
Retrieval logic for HeartSafe RAG.

This module handles:
1. Loading the persisted FAISS index (Task 10).
2. Rebuilding the BM25 index from pickled chunks (Task 11).
3. Combining them into a Hybrid EnsembleRetriever (Task 12).
"""
import pickle
import shutil
from pathlib import Path
from typing import List

# Vector Store and Embeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

# Project modules
from heartsafe_rag.config import settings
from heartsafe_rag.utils.logger import logger
from heartsafe_rag.exceptions import ConfigurationError

class HybridRetriever:
    """
    Manages the lifecycle of the Hybrid Search engine (FAISS + BM25).
    """
    
    def __init__(self, data_dir: Path = None):
        # Use the faiss_index directory in the project root if no data_dir is provided
        self.index_path = Path("faiss_index")
        self.chunks_path = self.index_path / "chunks.pkl"
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        self._ensemble_retriever: BaseRetriever | None = None

    def _load_faiss(self) -> BaseRetriever:
        """Loads the FAISS vector store from disk."""
        if not self.index_path.exists():
            raise ConfigurationError(f"FAISS index not found at {self.index_path}. Run ingest.py first.")
        
        logger.debug(f"Loading FAISS index from {self.index_path}...")
        vectorstore = FAISS.load_local(
            folder_path=str(self.index_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True # We created this file ourselves
        )
        return vectorstore.as_retriever(search_kwargs={"k": 5})

    def _build_bm25(self) -> BaseRetriever:
        """Loads chunks and rebuilds BM25 index in memory."""
        if not self.chunks_path.exists():
            raise ConfigurationError(f"Chunks file not found at {self.chunks_path}. Run ingest.py first.")

        logger.debug("Loading chunks for BM25...")
        with open(self.chunks_path, "rb") as f:
            chunks = pickle.load(f)

        logger.debug(f"Building BM25 index for {len(chunks)} chunks...")
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = 5
        return bm25

    def get_retriever(self, weights: list[float] = [0.4, 0.6]) -> BaseRetriever:
        """
        Returns the initialized EnsembleRetriever.
        
        Args:
            weights: [BM25_weight, FAISS_weight]. Default is heavy on semantic (0.6).
        """
        if self._ensemble_retriever:
            return self._ensemble_retriever

        logger.info("Initializing Hybrid Retrieval Engine...")
        
        try:
            faiss_retriever = self._load_faiss()
            bm25_retriever = self._build_bm25()

            self._ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=weights
            )
            logger.info("✅ Hybrid Retriever ready.")
            return self._ensemble_retriever
            
        except Exception as e:
            logger.critical(f"Failed to initialize retriever: {e}")
            raise

# Singleton instance to be used by the app
retriever_factory = HybridRetriever()


class RetrievalService:
    """
    Service to handle loading indices and performing hybrid retrieval.
    """
    def __init__(self):
        self.retriever = self._initialize_retriever()

    def _initialize_retriever(self) -> BaseRetriever:
        """
        Load FAISS and BM25 indices and create the EnsembleRetriever.
        """
        try:
            # 1. Load Embeddings
            embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

            # 2. Load FAISS (Vector Store)
            if not settings.VECTOR_DB_PATH.exists():
                logger.warning("Vector DB not found. Please run ingestion first.")
                raise ConfigurationError("Vector DB not found.")
                
            logger.info(f"Loading Vector Store from {settings.VECTOR_DB_PATH}...")
            
            vectorstore = FAISS.load_local(
                folder_path=str(settings.VECTOR_DB_PATH),
                index_name="index",
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": settings.RETRIEVAL_K})

            # 3. Load BM25 (Keyword)
            bm25_path = Path(settings.BM25_PATH)
            if not bm25_path.exists():
                raise ConfigurationError("BM25 index not found.")
                
            with open(bm25_path, "rb") as f:
                bm25_retriever = pickle.load(f)
            
            # Update BM25 k parameter
            bm25_retriever.k = settings.RETRIEVAL_K

            # 4. Initialize Ensemble Retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[0.4, 0.6]  # BM25 and FAISS weights
            )
            
            logger.info("Hybrid Retriever initialized successfully.")
            return ensemble_retriever

        except Exception as e:
            logger.error(f"Failed to initialize RetrievalService: {str(e)}")
            raise e

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a given query.
        """
        logger.info(f"Retrieving for query: {query}")
        try:
            docs = self.retriever.invoke(query)
            logger.info(f"Retrieved {len(docs)} documents")
            return docs
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise