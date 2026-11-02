"""Dataset Generator for Nykaa Domain Support Agent.

Track: E-commerce & Retail (Nykaa)
Seed: 42
Categories: ['Apparel', 'Electronics', 'Home', 'Footwear', 'Beauty']
Statuses: ['Placed', 'Shipped', 'Delivered', 'Returned', 'Refunded']
Price Range Reasoning: The order value range of INR 299 to INR 12,999 reflects
Nykaa's realistic catalog pricing, spanning affordable daily beauty essentials
(e.g., lipsticks, shampoos) to premium luxury electronics (e.g., hair stylers)
and designer apparel.
"""

import random
from collections import Counter
from typing import Dict, List, Any

SEED = 42
TOTAL_RECORDS = 45

CATEGORIES = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]
STATUSES = ["Placed", "Shipped", "Delivered", "Returned", "Refunded"]
MIN_ORDER_VALUE = 299.0
MAX_ORDER_VALUE = 12999.0


def generate_orders(seed: int = SEED, count: int = TOTAL_RECORDS) -> List[Dict[str, Any]]:
    """Generates a seeded, deterministic list of order records."""
    random.seed(seed)
    orders = []

    for i in range(1, count + 1):
        category = random.choice(CATEGORIES)
        status = random.choice(STATUSES)
        order_value = round(random.uniform(MIN_ORDER_VALUE, MAX_ORDER_VALUE), 2)
        days_since_created = random.randint(0, 30)
        # Empirical probability yielding 10-30% delayed shipments
        delayed_shipment = random.random() < 0.20

        orders.append({
            "record_id": f"ORD-{1000 + i}",
            "category": category,
            "status": status,
            "order_value_inr": order_value,
            "days_since_created": days_since_created,
            "delayed_shipment": delayed_shipment
        })

    return orders


# Generate deterministic ORDERS list
ORDERS: List[Dict[str, Any]] = generate_orders(seed=SEED, count=TOTAL_RECORDS)


def print_dataset_report(orders: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Prints and returns dataset validation statistics."""
    total = len(orders)
    cat_counts = Counter(o["category"] for o in orders)
    stat_counts = Counter(o["status"] for o in orders)
    delayed_count = sum(1 for o in orders if o["delayed_shipment"])
    delayed_pct = (delayed_count / total) * 100 if total > 0 else 0.0

    print("=" * 60)
    print("NYKAA ORDER DATASET REPORT (Task 1)")
    print(f"Total Records: {total} (Requirement: >= 40)")
    print(f"Random Seed: {SEED}")
    print(f"Price Range: INR {MIN_ORDER_VALUE} - INR {MAX_ORDER_VALUE}")
    print("-" * 60)
    print("Category Breakdown (Requirement: each category >= 3):")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  - {cat:12s}: {cnt:2d} records")
    print("-" * 60)
    print("Status Breakdown (Requirement: each status >= 1):")
    for stat, cnt in sorted(stat_counts.items()):
        print(f"  - {stat:12s}: {cnt:2d} records")
    print("-" * 60)
    print(f"Delayed Shipments: {delayed_count}/{total} ({delayed_pct:.2f}%) (Requirement: 10% - 30%)")
    print("=" * 60)

    # Verification checks
    for cat in CATEGORIES:
        assert cat_counts[cat] >= 3, f"Category '{cat}' has {cat_counts[cat]} records (< 3)"
    for stat in STATUSES:
        assert stat_counts[stat] >= 1, f"Status '{stat}' has {stat_counts[stat]} records (< 1)"
    assert 10.0 <= delayed_pct <= 30.0, f"Delayed % {delayed_pct:.2f}% outside [10%, 30%]"
    assert total >= 40, f"Total records {total} (< 40)"

    return {
        "total_records": total,
        "category_counts": dict(cat_counts),
        "status_counts": dict(stat_counts),
        "delayed_shipment_pct": delayed_pct
    }


if __name__ == "__main__":
    print_dataset_report(ORDERS)

