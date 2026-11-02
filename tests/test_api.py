"""Integration Tests for FastAPI Backend and Structured Logging (Tasks 11 & 12).

Tests:
1. GET /health
2. POST /ask (Policy query and order lookup)
3. POST /add-document (Dynamic indexing into ChromaDB)
4. WebSocket /ws/chat (Bidirectional exchange + graceful WebSocketDisconnect)
5. Structured JSON-Lines logging and pre-log PII redaction
"""

import json
import pytest
from fastapi.testclient import TestClient

from api.app import app
from observability.logger import read_recent_traces

client = TestClient(app)


def test_health_check_endpoint():
    """Verify health check endpoint returns 200 and active status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ask_endpoint_policy():
    """Verify POST /ask handles policy queries and returns AgentResponse."""
    payload = {
        "query": "What is the return window for lipsticks and beauty products?",
        "session_id": "api_test_policy_session"
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["query"] == payload["query"]
    assert data["status"] in ["answered", "escalated"]
    assert len(data["answer"]) > 0
    assert "KB-01" in data["retrieved_sources"]


def test_ask_endpoint_order():
    """Verify POST /ask handles order lookups and returns escalation score."""
    payload = {
        "query": "Check the status of my order ORD-1002",
        "session_id": "api_test_order_session"
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "ORD-1002" in data["retrieved_sources"]
    assert data["escalation_score"] is not None
    assert 0.0 <= data["escalation_score"] <= 1.0


def test_add_document_endpoint():
    """Verify POST /add-document dynamically indexes a new policy document into ChromaDB."""
    payload = {
        "topic": "Nykaa VIP Concierge Policy",
        "title": "VIP Concierge Personal Shopping SLA",
        "content": (
            "Nykaa VIP members are entitled to 24/7 dedicated personal shopping concierge services. "
            "Concierge requests are acknowledged within 15 minutes by a beauty expert. "
            "Exclusive complimentary gift wrapping and expedited priority shipping apply automatically."
        )
    }
    response = client.post("/add-document", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert data["chunks_indexed"] >= 2
    assert "successfully indexed" in data["message"].lower()


def test_websocket_chat_and_graceful_disconnect():
    """Verify WebSocket /ws/chat handles real-time messages and client disconnect cleanly."""
    with client.websocket_connect("/ws/chat") as websocket:
        # Send message
        websocket.send_text(json.dumps({
            "query": "What are the delivery SLAs for metro cities?",
            "session_id": "ws_test_session"
        }))

        # Receive response
        data_text = websocket.receive_text()
        data = json.loads(data_text)

        assert data["status"] in ["answered", "escalated"]
        assert len(data["answer"]) > 0

    # Exiting context manager triggers WebSocketDisconnect; server must not crash


def test_structured_logging_pii_redaction():
    """Verify structured logger writes valid JSONL and intercepts cleartext PII."""
    raw_query_with_pii = "My phone is 9876543210 and card is 4111-2222-3333-5555 for ORD-1001"
    response = client.post("/ask", json={"query": raw_query_with_pii, "session_id": "pii_log_test"})
    assert response.status_code == 200

    traces = read_recent_traces(limit=10)
    assert len(traces) > 0

    # Find the trace corresponding to our test query
    matched = None
    for t in reversed(traces):
        if t.get("endpoint") == "/ask" and "[PHONE_MASKED]" in t.get("sanitized_query", ""):
            matched = t
            break

    assert matched is not None, "Log entry with masked PII not found in traces.jsonl"
    assert "9876543210" not in matched["sanitized_query"], "Raw phone number leaked into traces.jsonl!"
    assert "4111-2222-3333-5555" not in matched["sanitized_query"], "Raw card leaked into traces.jsonl!"
    assert matched["pii_masked"] is True
    assert "trace_id" in matched
    assert "latency_ms" in matched

