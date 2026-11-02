"""Pydantic Structured Output Models for Nykaa Domain Support Agent (Task 9)."""

from typing import Optional, List
from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Unified structured response model for all Nykaa support agent interactions."""
    query: str = Field(
        ...,
        description="The customer query submitted to the support agent."
    )
    status: str = Field(
        default="answered",
        description="Execution status outcome: 'answered', 'escalated', or 'refused'."
    )
    answer: str = Field(
        ...,
        description="The final grounded answer synthesized for the customer."
    )
    escalation_score: Optional[float] = Field(
        default=None,
        description="Continuous escalation risk score in [0.0, 1.0] if an order was checked."
    )
    escalation_required: bool = Field(
        default=False,
        description="Whether this query has been flagged for human support intervention."
    )
    retrieved_sources: List[str] = Field(
        default_factory=list,
        description="List of document IDs or order record IDs referenced in the response."
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Model confidence score in [0.0, 1.0]."
    )


class VerdictModel(BaseModel):
    """Pydantic structured verdict for the AutoGen secondary review stage (Task 14)."""
    approved: bool = Field(
        ...,
        description="Whether the draft response was approved unchanged by the review team."
    )
    final_answer: str = Field(
        ...,
        description="The finalized answer delivered to the customer (original or revised)."
    )
    reason: str = Field(
        ...,
        description="Policy compliance or grounding justification for the review decision."
    )

