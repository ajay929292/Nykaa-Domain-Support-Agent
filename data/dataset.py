"""Data package alias for root dataset.py."""

from dataset import (
    ORDERS,
    generate_orders,
    print_dataset_report,
    SEED,
    CATEGORIES,
    STATUSES,
    MIN_ORDER_VALUE,
    MAX_ORDER_VALUE
)

__all__ = [
    "ORDERS",
    "generate_orders",
    "print_dataset_report",
    "SEED",
    "CATEGORIES",
    "STATUSES",
    "MIN_ORDER_VALUE",
    "MAX_ORDER_VALUE"
]

if __name__ == "__main__":
    print_dataset_report(ORDERS)

