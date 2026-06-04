from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(
        ..., min_length=3, max_length=5000, description="The clinical question to ask."
    )


class SourceDocument(BaseModel):
    content: str = Field(..., description="The content snippet from the document.")
    source: str = Field(..., description="The source filename or identifier.")
    chunk_index: int = Field(default=0, description="Index of this chunk in the retrieval result.")


class ReasoningStep(BaseModel):
    step: str = Field(..., description="The reasoning step text from the LLM.")
    claim_type: str = Field(
        default="other",
        description="Type of claim: diagnosis, recommendation, threshold, contraindication, definition, value_statement, refusal, other",
    )
    source_indices: list[int] = Field(
        default_factory=list,
        description="Indices into the response's sources array that support this step.",
    )


class ValidationAuditEntry(BaseModel):
    rule_name: str = Field(..., description="VLM-guard rule that fired.")
    action_type: str = Field(..., description="Action taken: pass, block, correct, promote, flag.")
    message: str = Field(..., description="Human-readable message from the rule.")
    severity: str = Field(default="info", description="info, warning, or error.")
    claim_index: int | None = Field(default=None, description="Which reasoning step this rule applied to (null = cross-claim).")
    modified_fields: dict[str, str] | None = Field(default=None, description="Fields that were corrected by the rule.")


class ValidationResult(BaseModel):
    status: str = Field(..., description="Overall validation status: passed, blocked, flagged, corrected, unverifiable.")
    rules_fired: list[ValidationAuditEntry] = Field(
        default_factory=list, description="List of rules that took non-pass actions."
    )


class ChatResponse(BaseModel):
    answer: str = Field(..., description="The generated answer from the AI.")
    sources: list[SourceDocument] = Field(
        default_factory=list, description="List of sources used."
    )
    reasoning_steps: list[ReasoningStep] | None = Field(
        default=None,
        description="Structured reasoning steps that led to the answer.",
    )
    validation: ValidationResult | None = Field(
        default=None,
        description="VLM-guard validation result (if enabled).",
    )
