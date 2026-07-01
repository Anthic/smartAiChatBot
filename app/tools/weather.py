from __future__ import annotations

from typing import Any


def get_weather(city: str) -> dict[str, Any]:
    normalized_city = city.strip() or "Unknown"

    return {
        "found": True,
        "city": normalized_city,
        "temperature_c": 29,
        "condition": "Partly cloudy",
        "humidity": "68%",
        "wind": "12 km/h",
        "note": "Mock weather data for demo purposes.",
    }