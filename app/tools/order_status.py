from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "orders.json"


@lru_cache(maxsize=1)
def _load_orders() -> list[dict[str, Any]]:
    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_order_status(order_id: str) -> dict[str, Any]:
    normalized_order_id = order_id.strip().upper()

    if not normalized_order_id:
        return {
            "found": False,
            "message": "Order ID is required.",
        }

    for order in _load_orders():
        if order["order_id"].upper() == normalized_order_id:
            return {
                "found": True,
                "order": order,
            }

    return {
        "found": False,
        "message": f"No order found for order_id '{normalized_order_id}'.",
    }