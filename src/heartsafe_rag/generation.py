from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field

from heartsafe_rag.config import settings
from heartsafe_rag.utils.logger import logger

SYSTEM_PROMPT_PATH = Path("prompts/system_prompt.txt")

GUARD_PROMPT = """You are a clinical safety monitor. Your job is to verify that an AI's answer
is fully grounded in the provided guideline context.

CONTEXT (guideline excerpts):
{context}

AI ANSWER:
{answer}

Does the answer contain any claims, statements, or numerical values that are NOT supported
by the context above? Consider:
- Facts not present in the context
- Numbers/dosages that differ from the context
- Recommendations not found in the context
- Speculation beyond what the context states

Output valid JSON only:
{{
    "is_grounded": true or false,
    "reason": "<if not grounded, explain what claim is unsupported. If grounded, say 'all claims supported'>"
}}
"""


class GuardVerdict(BaseModel):
    is_grounded: bool = Field(..., description="Whether the answer is fully grounded in the context")
    reason: str = Field(..., min_length=1, description="Explanation of grounding check result")


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
        )

        system_prompt = _load_system_prompt()

        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])
        self.rag_chain = rag_prompt | self.llm | StrOutputParser()

        guard_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a clinical safety monitor that verifies factual grounding."),
            ("human", GUARD_PROMPT),
        ])
        self.guard_chain = guard_prompt | self.llm | JsonOutputParser(pydantic_object=GuardVerdict)

    def generate_response(
        self,
        query: str,
        context_docs: list[Document] | None = None,
        callbacks: list[Any] | None = None,
    ) -> str:
        if callbacks is None:
            callbacks = [CallbackHandler()]

        if not context_docs:
            return (
                "I cannot provide a response to this query because:\n"
                "- No relevant guidelines were found for this question.\n"
                "- The AHA/ACC heart failure guidelines do not contain information on this topic.\n"
                "- I am designed to provide information only from the official heart failure guidelines."
            )

        context_text = "\n\n".join(doc.page_content for doc in context_docs)

        answer = self.rag_chain.invoke(
            {"context": context_text, "question": query},
            config={"callbacks": callbacks},
        )

        guard_result = self.guard_chain.invoke({
            "context": context_text,
            "answer": answer,
        })

        if not guard_result.get("is_grounded", False):
            logger.warning(f"Output guard rejected answer. Reason: {guard_result.get('reason', 'unknown')}")
            return (
                "I cannot provide a response to this query because:\n"
                "- The generated response contained information not supported by the retrieved guidelines.\n"
                f"- {guard_result.get('reason', 'Grounding verification failed')}\n"
                "- I am designed to provide information only from the official heart failure guidelines."
            )

        return str(answer)
