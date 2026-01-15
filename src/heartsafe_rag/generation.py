"""
Generation module for HeartSafe RAG.

Handles LLM interaction with intelligent routing:
- Heart Failure queries -> RAG (Retrieval + Generation)
- General queries -> Direct LLM response
"""

from typing import List, Optional, Any

from langchain_groq import ChatGroq
from langfuse.langchain import CallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from heartsafe_rag.config import settings
from heartsafe_rag.utils.logger import logger


class GenerationService:
    """
    Service for generating responses using an LLM with intent routing.
    """

    def __init__(self) -> None:
        """
        Initialize the GenerationService with LLM, Router, and Chains.
        """
        self.llm = ChatGroq(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            api_key=settings.GROQ_API_KEY,
        )

        # --- 1. Router Chain ---
        router_prompt = ChatPromptTemplate.from_template(
            """You are a query classifier for a Heart Failure Clinical Assistant.
            Classify the following user query into one of two categories:

            1. HF_RELATED
            2. GENERAL

            Return ONLY the category name.

            Query: {question}
            """
        )
        self.router_chain = router_prompt | self.llm | StrOutputParser()

        # --- 2. RAG Chain ---
        rag_prompt = ChatPromptTemplate.from_template(
            """You are an expert cardiologist assistant specializing in Heart Failure.

            Use ONLY the provided clinical context to answer.

            Context:
            {context}

            Question: {question}

            If the answer is not present in the context, explicitly state that the
            guidelines do not contain this information.
            """
        )
        self.rag_chain = rag_prompt | self.llm | StrOutputParser()

        # --- 3. Direct Chain ---
        direct_prompt = ChatPromptTemplate.from_template(
            """You are a helpful AI assistant.

            Question: {question}

            Answer clearly and concisely.
            """
        )
        self.direct_chain = direct_prompt | self.llm | StrOutputParser()

    def route_query(self, query: str) -> str:
        """Determine whether the query is HF-related or general."""
        classification = self.router_chain.invoke({"question": query}).strip()
        logger.info(f"Query classified as: {classification}")
        return classification

    def generate_response(
        self,
        query: str,
        context_docs: Optional[List[Document]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> str:
        """
        Generate a response with Langfuse tracing enabled.
        
        Args:
            query: The user's question.
            context_docs: List of documents for RAG (if applicable).
            callbacks: Optional list of LangChain callbacks. 
                       If None, a default Langfuse CallbackHandler is created.
                       Pass this from evaluate.py to link traces to experiments.
        """

        # Use provided callbacks (for eval) or create default handler (for production)
        if callbacks is None:
            # ✅ Correct Langfuse handler (v2/v3)
            callbacks = [CallbackHandler()]

        category = self.route_query(query)

        if category == "HF_RELATED":
            if not context_docs:
                return (
                    "This appears to be a heart failure–related question, "
                    "but no clinical documents were retrieved."
                )

            context_text = "\n\n".join(doc.page_content for doc in context_docs)

            return self.rag_chain.invoke(
                {"context": context_text, "question": query},
                config={"callbacks": callbacks},
            )

        return self.direct_chain.invoke(
            {"question": query},
            config={"callbacks": callbacks},
        )