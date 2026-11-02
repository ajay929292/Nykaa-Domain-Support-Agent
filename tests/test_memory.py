"""Unit Tests for Multi-Turn Session Memory (Task 8).

Tests:
1. State correctly carried across multi-turn exchanges within one process run.
2. Separate fresh conversation demonstrating prior state correctly absent/reset.
"""

import pytest
from crew.nykaa_crew import run_nykaa_agent
from crew.memory import format_session_history, clear_session, extract_contextual_order_id


def test_multi_turn_memory_retention():
    """Verify that order reference is correctly carried over across multiple conversation turns."""
    session_id = "test_memory_multi_turn_session"
    clear_session(session_id)

    # Turn 1: User introduces order ORD-1003
    turn1_query = "Hello, I placed an order with ID ORD-1003 recently."
    resp1 = run_nykaa_agent(turn1_query, session_id=session_id)
    assert "ORD-1003" in resp1.retrieved_sources

    # Turn 2: User refers to 'that order' without repeating the ID
    turn2_query = "What is the current delivery status of that order?"
    # Context resolver should extract ORD-1003 from session memory
    resolved = extract_contextual_order_id(session_id, turn2_query)
    assert resolved == "ORD-1003", f"Expected 'ORD-1003' resolved from memory, got {resolved}"

    resp2 = run_nykaa_agent(turn2_query, session_id=session_id)
    assert "ORD-1003" in resp2.retrieved_sources

    # Verify conversation history string contains both turns
    history = format_session_history(session_id)
    assert "ORD-1003" in history
    assert "User: " in history
    assert "Assistant: " in history


def test_fresh_session_state_isolation():
    """Verify that a separate fresh session does NOT inherit state from previous sessions."""
    fresh_session_id = "test_memory_fresh_isolated_session"
    clear_session(fresh_session_id)

    # In a fresh conversation, user asks for 'that order' without any prior context
    turn_query = "What is the current delivery status of that order?"
    resolved = extract_contextual_order_id(fresh_session_id, turn_query)

    # Must be None because fresh session has no prior state
    assert resolved is None, f"Expected None in fresh session, got {resolved}"

    resp = run_nykaa_agent(turn_query, session_id=fresh_session_id)
    # Order lookup should not be triggered; should be treated as general inquiry
    assert "ORD-1003" not in resp.retrieved_sources

