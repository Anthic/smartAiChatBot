from __future__ import annotations

from collections import deque
from typing import Literal, TypedDict

from app.config import settings


Role = Literal["user", "assistant", "system"]

class Message(TypedDict) :
    role : Role
    content :  str
    
class ConversationMemory : 
    """Stores recent conversation turns for one chat session."""
    
    def __init__(self, max_turns : int | None = None)-> None:
        self._message : deque[Message] = deque(
            maxlen= max_turns or settings.MEMORY_MAX_TURNS * 2
        )
    
    def add_turn(self, role : Role, content : str) -> None: 
        content = content.strip()
        if not content:
            return
        self._message.append(
            {
                "role": role,
                "content": content                
            }
        )
        
    def get_history(self) -> list[Message] :
        return list(self._message)
    
    def clean(self) -> None :
        self._message.clear()
        
        
class MemoryStore:
    """In-process session memory registry."""

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationMemory] = {}

    def get_or_create_session(self, session_id: str) -> ConversationMemory:
        normalized_session_id = session_id.strip()

        if not normalized_session_id:
            raise ValueError("session_id cannot be empty.")

        if normalized_session_id not in self._sessions:
            self._sessions[normalized_session_id] = ConversationMemory()

        return self._sessions[normalized_session_id]

    def clear_session(self, session_id: str) -> None:
        memory = self._sessions.get(session_id)
        if memory:
            memory.clear()

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


memory_store = MemoryStore()