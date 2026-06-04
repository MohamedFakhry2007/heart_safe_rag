import json
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langfuse.langchain import CallbackHandler

from heartsafe_rag.config import settings
from heartsafe_rag.schemas import ReasoningStep, ValidationAuditEntry, ValidationResult, SourceDocument
from heartsafe_rag.utils.callbacks import LLMResponseLoggingHandler
from heartsafe_rag.utils.logger import logger

_llm_log_handler = LLMResponseLoggingHandler(logger)

SYSTEM_PROMPT_PATH = Path("prompts/system_prompt_structured.txt")


def _extract_source_indices(text: str) -> list[int]:
    import re as _re
    return sorted({
        int(m.group(1))
        for m in _re.finditer(r"\[Chunk\s+(\d+)\]", text)
    })

_FALLBACK_PROMPT_PATH = Path("prompts/system_prompt.txt")


def _load_system_prompt() -> str:
    try:
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning(f"System prompt not found at {SYSTEM_PROMPT_PATH}, trying fallback.")
        try:
            return _FALLBACK_PROMPT_PATH.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return (
                "You are an expert cardiologist assistant specializing in Heart Failure. "
                "Use ONLY the provided clinical context to answer. "
                "If the answer is not present in the context, explicitly state that the "
                "guidelines do not contain this information."
            )


class GenerationResult:
    """Result of a generation call, before and after validation."""

    def __init__(
        self,
        answer: str,
        reasoning_steps: list[ReasoningStep] | None = None,
        validation: ValidationResult | None = None,
        sources: list[SourceDocument] | None = None,
        raw_llm_output: str | None = None,
    ) -> None:
        self.answer = answer
        self.reasoning_steps = reasoning_steps or []
        self.validation = validation
        self.sources = sources or []
        self.raw_llm_output = raw_llm_output


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
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\n"),
        ])
        self.rag_chain = rag_prompt | self.llm | StrOutputParser()

        self._validation_service: Any = None

    @property
    def validation_service(self) -> Any:
        if self._validation_service is None:
            if settings.ENABLE_VALIDATION:
                from heartsafe_rag.validation import ValidationService
                self._validation_service = ValidationService()
        return self._validation_service

    def generate_response(
        self,
        query: str,
        context_docs: list[Document] | None = None,
        callbacks: list[Any] | None = None,
    ) -> GenerationResult:
        if callbacks is None:
            callbacks = [CallbackHandler(), _llm_log_handler]

        if not context_docs:
            refusal_answer = (
                "I cannot provide a response to this query because the AHA/ACC heart failure "
                "guidelines do not contain information on this topic."
            )
            return GenerationResult(
                answer=refusal_answer,
                reasoning_steps=[ReasoningStep(step="No relevant guidelines found in the provided context.", claim_type="refusal")],
            )

        context_parts = []
        for i, doc in enumerate(context_docs):
            context_parts.append("[Chunk {}]\n{}".format(i, doc.page_content))
        context_text = "\n\n".join(context_parts)

        t0 = time.perf_counter()
        try:
            raw = self.rag_chain.invoke(
                {"context": context_text, "question": query},
                config={"callbacks": callbacks},
            )
        except Exception as e:
            logger.error(f"RAG generation failed: {e}", exc_info=True)
            return GenerationResult(
                answer="I encountered an error while generating a response. Please try again.",
            )
        t1 = time.perf_counter()
        logger.info(f"RAG generation took {t1 - t0:.2f}s")

        raw_text = str(raw).strip()

        parsed = self._parse_structured(raw_text, query)

        if parsed is not None:
            steps, answer_from_llm = parsed
            result = self._validate_generation(steps, answer_from_llm, query, context_docs)
            return result

        return GenerationResult(
            answer=raw_text,
            sources=[
                SourceDocument(
                    content=doc.page_content[:200] + "...",
                    source=doc.metadata.get("source", "unknown"),
                    chunk_index=doc.metadata.get("chunk_id", i),
                )
                for i, doc in enumerate(context_docs)
            ],
        )

    def _parse_structured(
        self,
        raw_text: str,
        query: str,
    ) -> tuple[list[ReasoningStep], str] | None:
        if settings.ENABLE_VLM_GUARD:
            from vlm_guard import parse_reasoning_steps as vlm_parse
            claims, answer, raw_dict = vlm_parse(raw_text, domain="cardiology")
            if claims and answer:
                steps = [
                    ReasoningStep(
                        step=c.claim_text or c.findings,
                        claim_type=c.claim_type,
                        source_indices=_extract_source_indices(c.claim_text or c.findings),
                    )
                    for c in claims
                ]
                return steps, answer

        try:
            import json as _json
            data = _json.loads(raw_text)
            if isinstance(data, dict) and "reasoning_steps" in data and "answer" in data:
                steps_raw = data["reasoning_steps"]
                steps = []
                for s in steps_raw:
                    if isinstance(s, str):
                        steps.append(ReasoningStep(
                            step=s,
                            source_indices=_extract_source_indices(s),
                        ))
                    elif isinstance(s, dict):
                        src_idx = s.get("sources", s.get("source_indices", []))
                        if not src_idx:
                            step_text = s.get("step", s.get("claim", ""))
                            src_idx = _extract_source_indices(step_text)
                        steps.append(ReasoningStep(
                            step=s.get("step", s.get("claim", "")),
                            claim_type=s.get("claim_type", "other"),
                            source_indices=src_idx,
                        ))
                return steps, data.get("answer", "")
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        return None

    def _validate_generation(
        self,
        steps: list[ReasoningStep],
        answer: str,
        query: str,
        context_docs: list[Document],
    ) -> GenerationResult:
        sources = [
            SourceDocument(
                content=doc.page_content[:200] + "...",
                source=doc.metadata.get("source", "unknown"),
                chunk_index=doc.metadata.get("chunk_id", i),
            )
            for i, doc in enumerate(context_docs or [])
        ]

        if not settings.ENABLE_VALIDATION or self.validation_service is None:
            return GenerationResult(
                answer=answer,
                reasoning_steps=steps,
                sources=sources,
            )

        vs = self.validation_service

        try:
            import json as _json
            llm_json = _json.dumps({
                "reasoning_steps": [
                    {"step": s.step, "claim_type": s.claim_type} for s in steps
                ],
                "answer": answer,
            })
        except (TypeError, ValueError):
            llm_json = answer

        context = {"question": query, "domain": "cardiology"}
        pipeline_result = vs.validate(
            llm_raw_output=llm_json,
            domain="cardiology",
            context=context,
        )

        rules_fired = [
            ValidationAuditEntry(
                rule_name=e.rule_name,
                action_type=e.action_type,
                message=e.message,
                severity=e.severity,
                claim_index=e.claim_index,
                modified_fields=e.modified_fields,
            )
            for e in pipeline_result.audit.entries
            if e.action_type != "pass"
        ]

        validation_result = ValidationResult(
            status=pipeline_result.status,
            rules_fired=rules_fired,
        )

        validated_answer = pipeline_result.answer or answer

        if pipeline_result.status == "blocked":
            validated_answer = (
                "I cannot verify this answer against the guidelines. "
                "Some claims could not be validated. "
                "A specialist should review the reasoning below."
            )

        return GenerationResult(
            answer=validated_answer,
            reasoning_steps=steps,
            validation=validation_result,
            sources=sources,
            raw_llm_output=llm_json,
        )
