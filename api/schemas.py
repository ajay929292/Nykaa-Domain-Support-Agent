"""FastAPI Request and Response Models for Nykaa Support Agent (Task 11)."""

from typing import Optional, List
from pydantic import BaseModel, Field
from models.response_models import AgentResponse


class AskRequest(BaseModel):
    """Payload for submitting a customer question or order lookup."""
    query: str = Field(
        ...,
        description="The customer inquiry or order inquiry text.",
        min_length=1
    )
    session_id: Optional[str] = Field(
        default="default_session",
        description="Unique conversation session identifier for multi-turn history."
    )


class AddDocumentRequest(BaseModel):
    """Payload for dynamically indexing a new policy document."""
    topic: str = Field(
        ...,
        description="Topic category of the policy document (e.g., 'express shipping policy').",
        min_length=3
    )
    title: str = Field(
        ...,
        description="Title of the policy document.",
        min_length=3
    )
    content: str = Field(
        ...,
        description="Full text of the policy document (at least 2 sentences).",
        min_length=20
    )


class AddDocumentResponse(BaseModel):
    """Response confirming document indexing into ChromaDB."""
    success: bool
    doc_id: str
    topic: str
    chunks_indexed: int
    message: str

