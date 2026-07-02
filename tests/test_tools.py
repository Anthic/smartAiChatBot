from __future__ import annotations

from app.tools.order_status import get_order_status
from app.tools.product_search import search_product


def test_order_status_found():
    response = get_order_status("ORD001")
    assert response["found"] is True
    assert "order" in response
    assert response["order"]["order_id"] == "ORD001"


def test_order_status_not_found():
    response = get_order_status("ORD999")
    assert response["found"] is False
    assert "No order found" in response["message"]


def test_order_status_empty_input():
    response = get_order_status("")
    assert response["found"] is False
    assert response["message"] == "Order ID is required."


def test_product_search_found():
    response = search_product("Wireless Mouse")
    assert response["found"] is True
    assert response["count"] >= 1
    assert any(p["name"] == "Wireless Mouse" for p in response["matches"])


def test_product_search_not_found():
    response = search_product("nonexistent_item_name")
    assert response["found"] is False
    assert not response["matches"]


def test_product_search_empty_input():
    response = search_product("")
    assert response["found"] is False
    assert not response["matches"]
