"""Unit Tests for Four-Layer AI Governance Model (Task 15).

Tests:
1. Application Layer:
   - Least Autonomy Enforcement: check_order_status authorized ONLY for Lookup Agent.
   - Blocks unauthorized agents from binding the lookup tool.
   - Formal Risk Classification: Medium Risk with justification.
2. Runtime Layer:
   - Token & Character Budget Cap: Rejects oversized queries exceeding budget.
"""

import pytest
from governance.governance_policy import (
    LeastAutonomyGuard,
    RuntimeBudgetGuard,
    GovernanceViolationError,
    SYSTEM_RISK_CLASSIFICATION,
    get_governance_report,
    MAX_REQUEST_CHARACTERS
)
from crew.nykaa_crew import run_nykaa_agent


def test_least_autonomy_enforcement():
    """Verify that only authorized Lookup Agents may bind check_order_status."""
    # Authorized role succeeds
    authorized_role = "Nykaa Order Fulfillment Specialist"
    assert LeastAutonomyGuard.validate_tool_wiring(authorized_role, "check_order_status") is True

    # Unauthorized role raises GovernanceViolationError
    unauthorized_roles = [
        "Nykaa Policy Specialist",
        "Nykaa Customer Support Lead",
        "Public Chat Agent"
    ]
    for role in unauthorized_roles:
        with pytest.raises(GovernanceViolationError) as exc_info:
            LeastAutonomyGuard.validate_tool_wiring(role, "check_order_status")
        assert "Least Autonomy Violation" in str(exc_info.value)


def test_system_risk_classification():
    """Verify system is classified as Medium Risk with complete justification."""
    report = get_governance_report()
    risk = report["risk_classification"]

    assert risk["tier"] == "Medium Risk"
    assert len(risk["justification"]) > 50
    assert "retail customer service" in risk["justification"].lower()
    assert "high risk" in risk["justification"].lower()
    assert "low risk" in risk["justification"].lower()


def test_runtime_budget_cap_rejection():
    """Verify runtime budget cap rejects deliberately oversized simulated requests."""
    # Normal request within budget
    normal_query = "What is the return window for lipsticks?"
    budget_check = RuntimeBudgetGuard.check_request_budget(normal_query)
    assert budget_check["allowed"] is True

    # Deliberately oversized query (> 800 characters)
    oversized_query = "Please help me with my return request. " * 35
    assert len(oversized_query) > MAX_REQUEST_CHARACTERS

    with pytest.raises(GovernanceViolationError) as exc_info:
        RuntimeBudgetGuard.check_request_budget(oversized_query)
    assert "Runtime Governance Budget Exceeded" in str(exc_info.value)

    # Test rejection via run_nykaa_agent pipeline
    resp = run_nykaa_agent(oversized_query, session_id="test_oversized_session")
    assert resp.status == "refused"
    assert "budget exceeded" in resp.answer.lower()
    assert resp.confidence == 0.0

