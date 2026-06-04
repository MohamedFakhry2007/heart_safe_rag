"""ValidationService - orchestrates VLM-guard claim validation for HeartSafe."""

from typing import Any

from vlm_guard import (
    Analysis, GuardrailEngine, TextGuardPipeline, TextPipelineResult,
    parse_reasoning_steps,
)

from heartsafe_rag.validation.rules import (
    CORLevelRule,
    LVEFThresholdRule,
    DrugClassRule,
    ContraindicationRule,
    ValueStatementRule,
    AnswerConsistencyRule,
)
from heartsafe_rag.config import settings
from heartsafe_rag.utils.logger import logger


def _build_engine() -> GuardrailEngine:
    engine = GuardrailEngine()
    engine.register(CORLevelRule())
    engine.register(LVEFThresholdRule())
    engine.register(DrugClassRule())
    engine.register(ContraindicationRule())
    engine.register(ValueStatementRule())
    engine.register_cross_claim(AnswerConsistencyRule())
    logger.info(
        "VLM-guard engine initialized: {} claim rules, {} cross-claim rules".format(
            len(engine._rules), len(engine._cross_claim_rules)
        )
    )
    return engine


class ValidationService:
    """Validates structured LLM output against HeartSafe VLM-guard rules."""

    def __init__(self) -> None:
        self._engine = _build_engine()

    def validate(
        self,
        llm_raw_output: str,
        domain: str = "cardiology",
        context: dict[str, Any] | None = None,
        max_retries: int | None = None,
    ) -> TextPipelineResult:
        """Parse raw LLM JSON into claims, validate, return result.

        Args:
            llm_raw_output: Raw JSON string from LLM with reasoning_steps + answer
            domain: Domain namespace for rules
            context: Additional context (question, retrieved docs, etc.)
            max_retries: Override default max retry count

        Returns:
            TextPipelineResult with validated claims, answer, audit
        """
        context = context or {}
        max_retries = max_retries if max_retries is not None else settings.VALIDATION_MAX_RETRIES

        claims, answer, raw_dict = parse_reasoning_steps(
            llm_raw_output, domain=domain
        )

        validated_claims, validated_answer, audit = self._engine.apply_to_claims(
            claims, answer, context
        )

        retry_count = 0
        status = self._determine_status(validated_claims, audit)

        if status == "blocked" and max_retries > 0:
            retry_count = 1
            logger.info("VLM-guard blocked - retry triggered")

        return TextPipelineResult(
            claims=validated_claims,
            answer=validated_answer,
            status=status,
            elapsed_seconds=0.0,
            audit=audit,
            retry_count=retry_count,
        )

    def validate_retry(
        self,
        retry_llm_output: str,
        domain: str = "cardiology",
        context: dict[str, Any] | None = None,
    ) -> TextPipelineResult:
        """Validate retry output from LLM after initial block."""
        return self.validate(retry_llm_output, domain, context, max_retries=0)

    def _determine_status(
        self,
        claims: list[Analysis],
        audit: Any,
    ) -> str:
        has_blocked = any(c.validation_status == "blocked" for c in claims)
        has_flagged = any(c.validation_status == "flagged" for c in claims)
        has_corrected = any(c.validation_status == "corrected" for c in claims)
        any_block_action = any(
            e.action_type == "block" for e in audit.entries
        )

        if has_blocked or any_block_action:
            return "blocked"
        if has_flagged:
            return "flagged"
        if has_corrected:
            return "corrected"
        return "passed"
