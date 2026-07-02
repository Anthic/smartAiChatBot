from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.tools.order_status import get_order_status
from app.tools.product_search import search_product


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get the status and estimated delivery date for a customer order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID, for example ORD001.",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_product",
            "description": "Search product catalog by product name or category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "The product name or category to search for.",
                    }
                },
                "required": ["product_name"],
            },
        },
    },
]


_TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "get_order_status": get_order_status,
    "search_product": search_product,
}


def execute_tool(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    tool = _TOOL_REGISTRY.get(tool_name)

    if tool is None:
        return {
            "success": False,
            "error": f"Unknown tool: {tool_name}",
        }

    try:
        result = tool(**tool_args)
    except TypeError as exc:
        return {
            "success": False,
            "error": f"Invalid arguments for tool '{tool_name}': {exc}",
        }

    return {
        "success": True,
        "tool_name": tool_name,
        "result": result,
    }
