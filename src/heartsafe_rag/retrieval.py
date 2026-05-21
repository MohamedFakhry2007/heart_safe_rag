import pickle

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from heartsafe_rag.config import settings
from heartsafe_rag.exceptions import ConfigurationError
from heartsafe_rag.utils.logger import logger

HYDE_PROMPT = """You are a cardiology expert. Given a clinical question, generate a short
hypothetical passage from the AHA/ACC Heart Failure Guidelines that would answer it.
Write it as if it were an actual guideline excerpt. Be specific and evidence-based.

Question: {question}

Hypothetical guideline passage:"""

MULTI_QUERY_PROMPT = """You are a cardiology expert. Given a clinical question about heart failure,
generate {count} different versions of the question that capture different aspects or phrasings.
Return each on a new line, numbered.

Original question: {question}

Variant questions:"""


class HybridRetriever:
    def __init__(self) -> None:
        self.index_path = settings.VECTOR_DB_PATH
        self.chunks_path = self.index_path / "chunks.pkl"
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        self._retriever: BaseRetriever | None = None
        self._rewrite_llm: ChatGroq | None = None

    @property
    def rewrite_llm(self) -> ChatGroq:
        if self._rewrite_llm is None:
            self._rewrite_llm = ChatGroq(
                model=settings.LLM_MODEL,
                temperature=0.3,
                api_key=settings.GROQ_API_KEY,
            )
        return self._rewrite_llm

    def _apply_hyde(self, query: str) -> str:
        """Generate a hypothetical guideline passage and use it as the retrieval query."""
        chain = ChatPromptTemplate.from_template(HYDE_PROMPT) | self.rewrite_llm | StrOutputParser()
        hyde_query = chain.invoke({"question": query})
        logger.debug(f"HyDE expanded query: {hyde_query[:100]}...")
        return hyde_query.strip()

    def _generate_multi_queries(self, query: str) -> list[str]:
        """Generate multiple query variants for broader retrieval."""
        chain = (
            ChatPromptTemplate.from_template(MULTI_QUERY_PROMPT)
            | self.rewrite_llm
            | StrOutputParser()
        )
        result = chain.invoke({"question": query, "count": settings.MULTI_QUERY_COUNT})
        variants = [line.strip().split(". ", 1)[-1] for line in result.strip().split("\n") if line.strip()]
        logger.debug(f"Multi-query variants: {variants}")
        return variants[: settings.MULTI_QUERY_COUNT]

    def _merge_results(self, all_docs: list[list[Document]]) -> list[Document]:
        """Merge multiple retrieval results, deduplicating by page_content."""
        seen: set[str] = set()
        merged: list[Document] = []
        for docs in all_docs:
            for doc in docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    merged.append(doc)
        return merged

    def _load_faiss(self) -> BaseRetriever:
        if not self.index_path.exists():
            raise ConfigurationError(f"FAISS index not found at {self.index_path}. Run ingest.py first.")

        logger.debug(f"Loading FAISS index from {self.index_path}...")
        vectorstore = FAISS.load_local(
            folder_path=str(self.index_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return vectorstore.as_retriever(search_kwargs={"k": settings.INITIAL_RETRIEVAL_K})

    def _build_bm25(self) -> BaseRetriever:
        if not self.chunks_path.exists():
            raise ConfigurationError(f"Chunks file not found at {self.chunks_path}. Run ingest.py first.")

        logger.debug("Loading chunks for BM25...")
        with self.chunks_path.open("rb") as f:
            chunks = pickle.load(f)

        logger.debug(f"Building BM25 index for {len(chunks)} chunks...")
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = settings.INITIAL_RETRIEVAL_K
        return bm25

    def get_retriever(self) -> BaseRetriever:
        if self._retriever:
            return self._retriever

        logger.info("Initializing Hybrid Retrieval Engine...")

        try:
            faiss_retriever = self._load_faiss()
            bm25_retriever = self._build_bm25()

            ensemble = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=settings.HYBRID_WEIGHTS,
            )

            logger.info(f"Initializing Re-Ranker: {settings.RERANKER_MODEL}")
            model = HuggingFaceCrossEncoder(model_name=settings.RERANKER_MODEL)
            compressor = CrossEncoderReranker(model=model, top_n=settings.RERANK_TOP_K)

            self._retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=ensemble,
            )

            logger.info("Hybrid Retriever with Cross-Encoder Re-ranking ready.")

        except Exception as e:
            logger.critical(f"Failed to initialize retriever: {e}")
            raise

        return self._retriever

    def retrieve(self, query: str) -> list[Document]:
        logger.info(f"Retrieving for query: {query}")
        try:
            retriever = self.get_retriever()
            queries_to_run: list[str] = []

            if settings.ENABLE_HYDE:
                hyde_query = self._apply_hyde(query)
                queries_to_run.append(hyde_query)

            if settings.ENABLE_MULTI_QUERY:
                variants = self._generate_multi_queries(query)
                queries_to_run.extend(variants)

            queries_to_run.append(query)

            all_results = [retriever.invoke(q) for q in queries_to_run]
            merged = self._merge_results(all_results)
            logger.info(f"Retrieved {len(merged)} documents after query rewriting + re-ranking")
            return merged  # noqa: TRY300  # type: ignore[no-any-return]
        except Exception as e:
            logger.error(f"Error during retrieval: {e!s}")
            raise
