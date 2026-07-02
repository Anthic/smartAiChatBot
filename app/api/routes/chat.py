from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from app.agent.router import agent_router
from app.models.chat import ChatRequest, ChatResponse


router = APIRouter(prefix="/chat", tags=["Agent Chat"])


@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_with_agent(payload: ChatRequest) -> ChatResponse:
    """
    Send a message to the AI agent.
    
    The agent will route the message to the appropriate source (RAG, Mock Tool, or Direct LLM).
    """
    try:
        response = await run_in_threadpool(
            agent_router.process,
            payload.session_id,
            payload.message,
            payload.source_files,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent failed to process request: {exc}",
        ) from exc

    return ChatResponse(
        reply=response.reply,
        session_id=payload.session_id,
        source=response.source,
        metadata=response.metadata,
    )
