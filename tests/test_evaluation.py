"""Unit Tests for 15-Query Evaluation Suite (Task 13).

Tests:
1. Validates test_suite_15.json contains exactly 15 queries covering all 12 KB topics.
2. Validates LLM-as-judge scoring computes all 4 metrics per query and dataset averages.
"""

import pytest
from evaluation.evaluator import load_test_suite, run_full_evaluation


def test_test_suite_composition():
    """Verify test_suite_15.json contains 15 queries covering all 12 required KB topics."""
    suite = load_test_suite()
    assert len(suite) == 15, f"Expected 15 test queries, found {len(suite)}"

    required_topics = [
        "return window by product category",
        "COD refund timelines",
        "delivery SLAs",
        "reverse-pickup eligibility",
        "warranty terms by category",
        "order-cancellation policy",
        "loyalty-points redemption policy",
        "payment-failure/retry policy",
        "size-exchange policy",
        "damaged-item claim process",
        "international shipping restrictions",
        "customer-support escalation matrix"
    ]

    suite_topics = [item["topic"] for item in suite]
    for topic in required_topics:
        assert topic in suite_topics, f"Missing topic in 15-query suite: {topic}"

    # Verify at least 1 order lookup and at least 2 out-of-scope / adversarial
    in_scope_count = sum(1 for item in suite if item["is_in_scope"])
    out_of_scope_count = sum(1 for item in suite if not item["is_in_scope"])

    assert in_scope_count >= 13  # 12 KB + 1 order lookup
    assert out_of_scope_count >= 2  # 2 out-of-scope / adversarial


def test_full_evaluation_execution():
    """Verify run_full_evaluation calculates scores for all 15 queries and computes averages."""
    results = run_full_evaluation()
    per_query = results["per_query_results"]
    averages = results["averages"]

    assert len(per_query) == 15

    for r in per_query:
        scores = r["scores"]
        assert "accuracy" in scores
        assert "grounding" in scores
        assert "completeness" in scores
        assert "safety" in scores
        for metric, val in scores.items():
            assert 0.0 <= val <= 1.0

    assert averages["accuracy"] >= 0.85
    assert averages["grounding"] >= 0.85
    assert averages["completeness"] >= 0.85
    assert averages["safety"] == 1.0

