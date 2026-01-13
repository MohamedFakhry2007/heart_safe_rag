"""
Pydantic schemas for the HeartSafe RAG API.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Request model for the chat endpoint.
    """
    query: str = Field(..., min_length=3, description="The clinical question to ask.")


class SourceDocument(BaseModel):
    """
    Model representing a source document used in the answer.
    """
    content: str = Field(..., description="The content snippet from the document.")
    source: str = Field(..., description="The source filename or identifier.")


class ChatResponse(BaseModel):
    """
    Response model for the chat endpoint.
    """
    answer: str = Field(..., description="The generated answer from the AI.")
    category: str = Field(..., description="The classification of the query (HF_RELATED or GENERAL).")
    sources: List[SourceDocument] = Field(default_factory=list, description="List of sources used if RAG was performed.")