"""
Generation module for HeartSafe RAG.

Handles LLM interaction with intelligent routing:
- Heart Failure queries -> RAG (Retrieval + Generation)
- General queries -> Direct LLM response
"""
from typing import List, Dict, Any, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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
            temperature=settings.LLM_TEMPERATURE,  # FIXED: Was settings.TEMPERATURE
            api_key=settings.GROQ_API_KEY,
        )

        # --- 1. Router Chain ---
        # Classifies if the query requires Heart Failure context.
        router_prompt = ChatPromptTemplate.from_template(
            """You are a query classifier for a Heart Failure Clinical Assistant.
            Classify the following user query into one of two categories:
            
            1. 'HF_RELATED': If the query is about heart failure, cardiology guidelines, medications, symptoms, or treatment.
            2. 'GENERAL': If the query is about anything else (greeting, coding, general knowledge, other medical topics).
            
            Return ONLY the category name ('HF_RELATED' or 'GENERAL'). Do not add punctuation.
            
            Query: {question}
            """
        )
        self.router_chain = router_prompt | self.llm | StrOutputParser()

        # --- 2. RAG Chain (For Heart Failure) ---
        rag_prompt = ChatPromptTemplate.from_template(
            """You are an expert cardiologist assistant specializing in Heart Failure.
            Use the following retrieved medical context to answer the question.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer professionally based ONLY on the provided context. 
            If the context doesn't contain the answer, state that you don't have that information in your guidelines.
            """
        )
        self.rag_chain = (
            rag_prompt 
            | self.llm 
            | StrOutputParser()
        )

        # --- 3. Direct Chain (For General Chat) ---
        direct_prompt = ChatPromptTemplate.from_template(
            """You are a helpful AI assistant.
            
            Question: {question}
            
            Answer the question helpfully and concisely.
            """
        )
        self.direct_chain = direct_prompt | self.llm | StrOutputParser()

    def route_query(self, query: str) -> str:
        """Determines if the query is HF_RELATED or GENERAL."""
        classification = self.router_chain.invoke({"question": query}).strip()
        logger.info(f"Query classified as: {classification}")
        return classification

    def generate_response(self, query: str, context_docs: Optional[List[Document]] = None) -> str:
        """
        Generate a response. 
        If context_docs are provided, it assumes RAG is intended (or we can re-route inside).
        
        To strictly follow your requirement: We check routing first.
        """
        category = self.route_query(query)

        if category == "HF_RELATED":
            if not context_docs:
                return "This appears to be a clinical question, but no documents were retrieved."
            
            # Format documents
            context_text = "\n\n".join([doc.page_content for doc in context_docs])
            return self.rag_chain.invoke({
                "context": context_text,
                "question": query
            })
            
        else:
            # General Query - Ignore context, just answer
            return self.direct_chain.invoke({"question": query})