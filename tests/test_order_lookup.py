"""Unit Tests for Order Status Lookup Tool & Escalation Formula (Task 6)."""

import pytest
from tools.order_lookup import (
    check_order_status,
    compute_escalation_score,
    ESCALATION_THRESHOLD,
    DELAY_WEIGHT,
    RECENCY_WEIGHT
)
from dataset import ORDERS


def test_order_lookup_existing_record():
    """Verify check_order_status returns required fields on an existing record."""
    first_order = ORDERS[0]
    rec_id = first_order["record_id"]

    res = check_order_status(rec_id)
    assert res["found"] is True
    assert res["record_id"] == rec_id
    assert res["status"] == first_order["status"]
    assert res["order_value_inr"] == first_order["order_value_inr"]
    assert 0.0 <= res["escalation_score"] <= 1.0
    assert isinstance(res["escalation_recommended"], bool)
    assert res["escalation_recommended"] == (res["escalation_score"] >= ESCALATION_THRESHOLD)


def test_order_lookup_unknown_record():
    """Verify graceful handling of non-existent order ID."""
    res = check_order_status("ORD-9999")
    assert res["found"] is False
    assert res["status"] == "Not Found"
    assert res["escalation_score"] == 0.0
    assert res["escalation_recommended"] is False


def test_escalation_score_boundaries_and_formula():
    """Verify formula arithmetic and edge-case boundaries."""
    # Day 0, No delay -> Score should be 0.0
    assert compute_escalation_score(delayed_shipment=False, days_since_created=0) == 0.0

    # Day 0, Delayed -> Score should be exactly DELAY_WEIGHT (0.45)
    assert compute_escalation_score(delayed_shipment=True, days_since_created=0) == DELAY_WEIGHT

    # Day 30, No delay -> Score should be exactly RECENCY_WEIGHT (0.55)
    assert compute_escalation_score(delayed_shipment=False, days_since_created=30) == RECENCY_WEIGHT

    # Day 30, Delayed -> Score should be capped at 1.0 (0.45 + 0.55 = 1.0)
    assert compute_escalation_score(delayed_shipment=True, days_since_created=30) == 1.0


def test_dataset_escalation_distribution():
    """Verify escalation behavior across the actual dataset."""
    escalated_count = 0
    for order in ORDERS:
        res = check_order_status(order["record_id"])
        if res["escalation_recommended"]:
            escalated_count += 1
            assert res["escalation_score"] >= ESCALATION_THRESHOLD

    # Escalated percentage should be reasonable (between 10% and 30%)
    escalation_pct = (escalated_count / len(ORDERS)) * 100
    assert 10.0 <= escalation_pct <= 30.0, f"Escalated % {escalation_pct:.1f}% outside [10%, 30%]"

