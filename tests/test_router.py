from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from app.agent.router import AgentRouter


@pytest.fixture
def mock_mistral_choice():
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = "This is a direct response."
    choice.message.tool_calls = None
    return choice


@pytest.fixture
def mock_mistral_response(mock_mistral_choice):
    response = MagicMock()
    response.choices = [mock_mistral_choice]
    return response


@patch("app.agent.router.Mistral")
def test_router_direct_llm_response(mock_client_class, mock_mistral_response):
    # Set up mock client response
    mock_client = mock_client_class.return_value
    mock_client.chat.complete.return_value = mock_mistral_response

    router = AgentRouter()
    response = router.process("session_direct", "Hello, who are you?")

    assert response.source == "llm"
    assert response.reply == "This is a direct response."
    assert response.metadata == {"finish_reason": "stop"}


@patch("app.agent.router.retrieve")
@patch("app.agent.router.Mistral")
def test_router_fallback_to_rag(mock_client_class, mock_retrieve, mock_mistral_response, mock_mistral_choice):
    # Setup first LLM response to return fallback phrase
    mock_mistral_choice.message.content = "I couldn't find that information in the uploaded documents."
    mock_client = mock_client_class.return_value
    
    # Setup RAG retrieval chunks
    mock_retrieve.return_value = [
        {"text": "RAG information.", "score": 0.95, "source_file": "doc.txt", "chunk_index": 0}
    ]

    # Setup second LLM response (after injecting RAG context)
    second_choice = MagicMock()
    second_choice.finish_reason = "stop"
    second_choice.message.content = "According to the document, here is the answer."
    
    second_response = MagicMock()
    second_response.choices = [second_choice]

    # Side effect: first call returns fallback, second call returns RAG answer
    mock_client.chat.complete.side_effect = [mock_mistral_response, second_response]

    router = AgentRouter()
    response = router.process("session_rag", "What is return policy?")

    assert response.source == "rag"
    assert response.reply == "According to the document, here is the answer."
    assert response.metadata["chunks"] == mock_retrieve.return_value
