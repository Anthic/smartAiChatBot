from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "products.json"


@lru_cache(maxsize=1)
def _load_products() -> list[dict[str, Any]]:
    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def search_product(product_name: str) -> dict[str, Any]:
    query = product_name.strip().lower()

    if not query:
        return {
            "found": False,
            "message": "Product name is required.",
            "matches": [],
        }

    matches = [
        product
        for product in _load_products()
        if query in product["name"].lower()
        or query in product.get("category", "").lower()
    ]

    if not matches:
        return {
            "found": False,
            "message": f"No products found for '{product_name}'.",
            "matches": [],
        }

    return {
        "found": True,
        "count": len(matches),
        "matches": matches,
    }