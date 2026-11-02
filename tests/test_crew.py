"""Unit Tests for CrewAI Multi-Agent System (Task 7 & 9).

Tests:
- 3-Agent Crew architecture (Retrieval, Lookup, Response Composer).
- Tool invocation for policy questions (RAG tool).
- Tool invocation for order lookup (check_order_status tool).
- Structured output validation against AgentResponse Pydantic schema.
"""

import pytest
from crew.nykaa_crew import run_nykaa_agent
from models.response_models import AgentResponse


def test_crew_policy_query_invokes_rag():
    """Verify that a policy inquiry triggers RAG retrieval and returns structured AgentResponse."""
    query = "What is the return window for makeup and beauty products?"
    resp = run_nykaa_agent(query, session_id="test_crew_rag_session")

    assert isinstance(resp, AgentResponse)
    assert resp.status in ["answered", "escalated"]
    assert len(resp.answer) > 0
    assert "KB-01" in resp.retrieved_sources
    assert resp.confidence > 0.0


def test_crew_order_query_invokes_lookup():
    """Verify that an order query triggers order lookup and calculates escalation score."""
    query = "Please check the status of my order ORD-1002"
    resp = run_nykaa_agent(query, session_id="test_crew_order_session")

    assert isinstance(resp, AgentResponse)
    assert "ORD-1002" in resp.retrieved_sources
    assert resp.escalation_score is not None
    assert 0.0 <= resp.escalation_score <= 1.0
    # ORD-1002 has delayed_shipment=True and days=13 -> escalation_score = 0.6883 -> escalated
    assert resp.status == "escalated"
    assert resp.escalation_required is True


def test_crew_pydantic_schema_conformity():
    """Verify that every crew response conforms to the declared Pydantic schema."""
    resp = run_nykaa_agent("Can I cancel an order before shipping?", session_id="test_crew_pydantic")
    dumped = resp.model_dump()

    required_keys = {
        "query",
        "status",
        "answer",
        "escalation_score",
        "escalation_required",
        "retrieved_sources",
        "confidence"
    }
    assert required_keys.issubset(dumped.keys())

