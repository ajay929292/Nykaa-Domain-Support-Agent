"""Four-Layer AI Governance Policy for Nykaa Domain Support Agent (Task 15).

Implements:
1. Application Layer:
   - Principle of Least Autonomy: Only the Lookup Agent is permitted to execute
     'check_order_status'. Attempts by any other agent to wire or invoke the tool are blocked.
   - Formal System Risk Classification: Medium Risk with documented justification.
2. Runtime Layer:
   - Per-request token and character budget cap: Oversized requests are rejected
     before LLM dispatch rather than silently exceeding computational resources.
"""

from typing import Dict, Any, Optional

MAX_REQUEST_CHARACTERS = 800
MAX_ESTIMATED_TOKENS = 200


class GovernanceViolationError(Exception):
    """Raised when an AI governance policy rule is violated."""
    pass


class LeastAutonomyGuard:
    """Application-Layer governance enforcing least autonomy across agent roles."""

    AUTHORIZED_ROLES = ["Nykaa Order Fulfillment Specialist", "Lookup Agent"]

    @classmethod
    def validate_tool_wiring(cls, agent_role: str, tool_name: str) -> bool:
        """Enforces that check_order_status is strictly bound only to authorized Lookup agents.

        Explanation:
        In customer support AI architectures, the lookup tool has direct read access
        to order records and fulfillment databases. Granting this tool to retrieval or
        composer agents violates the principle of least privilege, increasing the attack
        surface for prompt injections to exfiltrate private transaction data. Least autonomy
        strictly isolates order query capabilities to the Lookup Agent alone.
        """
        if tool_name == "check_order_status":
            if not any(auth in agent_role for auth in cls.AUTHORIZED_ROLES):
                raise GovernanceViolationError(
                    f"Least Autonomy Violation: Tool '{tool_name}' is restricted exclusively "
                    f"to the Lookup Agent. Role '{agent_role}' is not authorized."
                )
        return True


class RuntimeBudgetGuard:
    """Runtime-Layer governance enforcing per-request compute and cost caps."""

    @classmethod
    def check_request_budget(cls, query_text: str) -> Dict[str, Any]:
        """Validates that incoming requests do not exceed runtime character/token limits.

        Oversized requests (> 800 characters) are rejected before model execution
        to prevent resource exhaustion or denial-of-wallet attacks.
        """
        length = len(query_text)
        estimated_tokens = length // 4

        if length > MAX_REQUEST_CHARACTERS:
            raise GovernanceViolationError(
                f"Runtime Governance Budget Exceeded: Request length ({length} characters, "
                f"~{estimated_tokens} tokens) exceeds the maximum runtime budget cap of "
                f"{MAX_REQUEST_CHARACTERS} characters. Request rejected."
            )

        return {
            "allowed": True,
            "character_count": length,
            "estimated_tokens": estimated_tokens,
            "cap": MAX_REQUEST_CHARACTERS
        }


SYSTEM_RISK_CLASSIFICATION = {
    "tier": "Medium Risk",
    "justification": (
        "The Nykaa Domain Support Agent is formally classified as a Medium Risk AI system. "
        "While it operates in retail customer service handling order tracking, return window policies, "
        "and customer fulfillment disputes, it does not process sensitive health/medical diagnostics, "
        "credit underwriting, or automated hiring decisions (which define High Risk systems). However, "
        "because erroneous policy advice or improper escalation scores directly affect customer gross "
        "merchandise value (GMV), warranty rights, and delivery satisfaction, it exceeds the Low Risk tier "
        "(simple summarization or transcription). Consequently, the system operates under Medium Risk governance "
        "controls: mandatory least-autonomy tool gating, input-side PII masking, secondary multi-agent editorial "
        "review, and runtime budget limits."
    )
}


def get_governance_report() -> Dict[str, Any]:
    """Returns the full 4-layer AI governance profile."""
    return {
        "system_name": "Nykaa Domain Support Agent",
        "risk_classification": SYSTEM_RISK_CLASSIFICATION,
        "application_layer": {
            "least_autonomy_enforced": True,
            "restricted_tools": ["check_order_status"],
            "authorized_agent": "Nykaa Order Fulfillment Specialist"
        },
        "runtime_layer": {
            "budget_cap_characters": MAX_REQUEST_CHARACTERS,
            "budget_cap_tokens": MAX_ESTIMATED_TOKENS,
            "rejection_action": "Reject with GovernanceViolationError"
        }
    }

