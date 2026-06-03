import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langfuse.langchain import CallbackHandler

from heartsafe_rag.config import settings
from heartsafe_rag.utils.callbacks import LLMResponseLoggingHandler
from heartsafe_rag.utils.logger import logger

_llm_log_handler = LLMResponseLoggingHandler(logger)

SYSTEM_PROMPT_PATH = Path("prompts/system_prompt.txt")


def _load_system_prompt() -> str:
    try:
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning(f"System prompt not found at {SYSTEM_PROMPT_PATH}, using fallback.")
        return (
            "You are an expert cardiologist assistant specializing in Heart Failure. "
            "Use ONLY the provided clinical context to answer. "
            "If the answer is not present in the context, explicitly state that the "
            "guidelines do not contain this information."
        )


class GenerationService:
    def __init__(self) -> None:
        self.llm = ChatGroq(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            api_key=settings.GROQ_API_KEY,
            request_timeout=settings.LLM_TIMEOUT,
        )

        system_prompt = _load_system_prompt()

        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])
        self.rag_chain = rag_prompt | self.llm | StrOutputParser()

    def generate_response(
        self,
        query: str,
        context_docs: list[Document] | None = None,
        callbacks: list[Any] | None = None,
    ) -> str:
        if callbacks is None:
            callbacks = [CallbackHandler(), _llm_log_handler]

        if not context_docs:
            return (
                "I cannot provide a response to this query because:\n"
                "- No relevant guidelines were found for this question.\n"
                "- The AHA/ACC heart failure guidelines do not contain information on this topic.\n"
                "- I am designed to provide information only from the official heart failure guidelines."
            )

        context_text = "\n\n".join(doc.page_content for doc in context_docs)

        t0 = time.perf_counter()
        try:
            answer = self.rag_chain.invoke(
                {"context": context_text, "question": query},
                config={"callbacks": callbacks},
            )
        except Exception as e:
            logger.error(f"RAG generation failed: {e}", exc_info=True)
            return (
                "I encountered an error while generating a response. "
                "Please try rephrasing your question or try again later."
            )
        t1 = time.perf_counter()
        logger.info(f"RAG generation took {t1 - t0:.2f}s")

        return str(answer)
