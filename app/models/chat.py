
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


class ChatRequest(BaseModel) :
    session_id : str = Field(
        ...,
        description="The unique session/conversation ID.",
        examples=["session_123"],
    )
    message : str = Field (
        ...,
        description="The message to send to the agent.",
        examples=["Do you have a wireless mouse in stock?"],        
    )
    source_files: list[str] | None = Field(
        None,
        description="Optional list of filenames to search within.",
        examples=[["return_policy.pdf"]],
    )

class ChatResponse(BaseModel) : 
    reply : str = Field(..., description="The response message from the agent.")
    session_id : str = Field(..., description="The associated session/conversation ID.")

    source : Literal["llm", "tool", "rag"] = Field(
        ..., description="The source of the response.",
        
    )
    metadata: dict[str, Any] | None = Field(
        None,
        description="Optional execution metadata (e.g., tool execution results, retrieved chunks).",
    )