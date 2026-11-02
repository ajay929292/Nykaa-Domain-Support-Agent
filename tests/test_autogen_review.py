"""Unit Tests for AutoGen Secondary Review Stage (Task 14).

Tests:
1. 2-Agent AutoGen Round Robin Group Chat (Policy Compliance Reviewer & Final Editor).
2. Pydantic Structured Verdict (VerdictModel).
3. Demonstration Case 1: Grounded draft approved unchanged.
4. Demonstration Case 2: Injected ungrounded claim detected and revised.
"""

import pytest
from review.autogen_team import review_draft_sync
from models.response_models import VerdictModel


def test_autogen_review_approves_grounded_draft():
    """Verify that a compliant, grounded policy draft is approved unchanged."""
    query = "What is the return window for makeup products?"
    context = (
        "Nykaa provides a 15-day return window for unopened beauty, personal care, "
        "and cosmetics items from the delivery date."
    )
    draft = "Nykaa provides a 15-day return window for unopened beauty items."

    verdict = review_draft_sync(draft, context, query)

    assert isinstance(verdict, VerdictModel)
    assert verdict.approved is True
    assert verdict.final_answer == draft
    assert "approved" in verdict.reason.lower()


def test_autogen_review_revises_ungrounded_draft():
    """Verify that an ungrounded claim injected into the draft is caught and revised."""
    query = "What is your replacement warranty on cosmetics?"
    context = (
        "Beauty cosmetics, fragrances, and skincare products do not come with warranties "
        "but are 100% authentic and sourced directly from brands."
    )
    # Deliberately injected ungrounded claim
    injected_draft = (
        "Nykaa offers free lifetime replacements on all opened beauty products, "
        "no questions asked."
    )

    verdict = review_draft_sync(injected_draft, context, query)

    assert isinstance(verdict, VerdictModel)
    assert verdict.approved is False
    assert "lifetime" not in verdict.final_answer.lower()
    assert "excised" in verdict.reason.lower() or "revised" in verdict.reason.lower()

