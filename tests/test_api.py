from __future__ import annotations

from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


@patch("app.vector_store.qdrant_client.ping_qdrant")
def test_health_check(mock_ping):
    mock_ping.return_value = {
        "connected": True,
        "collection": "SmartAiChatbot",
        "collection_exists": True,
        "vectors_count": 10,
    }
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["qdrant"]["connected"] is True


@patch("app.api.routes.chat.run_in_threadpool")
def test_chat_endpoint(mock_run):
    from app.agent.router import AgentResponse
    
    # Mock the thread pool output returning AgentResponse
    mock_run.return_value = AgentResponse(
        reply="This is a test chat response.",
        source="llm",
        metadata={"finish_reason": "stop"}
    )

    payload = {
        "session_id": "api_test_session",
        "message": "Hello Agent!"
    }
    
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == "This is a test chat response."
    assert data["session_id"] == "api_test_session"
    assert data["source"] == "llm"  # Matches schema mapping after fix
