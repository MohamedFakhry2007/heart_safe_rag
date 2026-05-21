from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(
        ..., min_length=3, max_length=5000, description="The clinical question to ask."
    )


class SourceDocument(BaseModel):
    content: str = Field(..., description="The content snippet from the document.")
    source: str = Field(..., description="The source filename or identifier.")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="The generated answer from the AI.")
    sources: list[SourceDocument] = Field(
        default_factory=list, description="List of sources used."
    )
