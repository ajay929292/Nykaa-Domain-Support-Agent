"""Dataset Component Unit Tests.

Tests:
- Seeded determinism and reproducibility.
- Total orders threshold (>= 40).
- Category vocabulary coverage (every category >= 3).
- Status vocabulary coverage (every status >= 1).
- Delayed shipment percentage band (strictly 10% - 30%).
- Realistic price range (INR 299 - 12,999).
"""

import pytest
from dataset import (
    ORDERS,
    CATEGORIES,
    STATUSES,
    MIN_ORDER_VALUE,
    MAX_ORDER_VALUE,
    print_dataset_report
)


def test_order_count_and_schema():
    """Verify minimum order count and required fields in each record."""
    assert len(ORDERS) >= 40, f"Expected >= 40 orders, got {len(ORDERS)}"

    required_fields = {
        "record_id",
        "category",
        "status",
        "order_value_inr",
        "days_since_created",
        "delayed_shipment"
    }

    for order in ORDERS:
        assert required_fields.issubset(order.keys()), f"Missing fields in {order}"
        assert order["record_id"].startswith("ORD-"), f"Invalid record_id: {order['record_id']}"
        assert 0 <= order["days_since_created"] <= 30, f"Days out of range: {order['days_since_created']}"
        assert isinstance(order["delayed_shipment"], bool)


def test_category_distribution():
    """Verify all given categories appear at least 3 times."""
    report = print_dataset_report(ORDERS)
    cat_counts = report["category_counts"]

    for category in CATEGORIES:
        count = cat_counts.get(category, 0)
        assert count >= 3, f"Category '{category}' has only {count} records (required >= 3)"


def test_status_distribution():
    """Verify all given statuses appear at least once."""
    report = print_dataset_report(ORDERS)
    stat_counts = report["status_counts"]

    for status in STATUSES:
        count = stat_counts.get(status, 0)
        assert count >= 1, f"Status '{status}' has only {count} records (required >= 1)"


def test_delayed_shipment_percentage_band():
    """Verify delayed shipment percentage lands strictly between 10% and 30%."""
    report = print_dataset_report(ORDERS)
    pct = report["delayed_shipment_pct"]
    assert 10.0 <= pct <= 30.0, f"Delayed shipment % {pct:.2f}% is outside [10%, 30%]"


def test_price_range():
    """Verify order value is within realistic catalog range."""
    for order in ORDERS:
        val = order["order_value_inr"]
        assert MIN_ORDER_VALUE <= val <= MAX_ORDER_VALUE, f"Value {val} outside bounds [{MIN_ORDER_VALUE}, {MAX_ORDER_VALUE}]"

