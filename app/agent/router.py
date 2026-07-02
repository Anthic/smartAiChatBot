from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any, Literal

from mistralai.client import Mistral

from app.agent.memory import memory_store
from app.agent.prompts import FALLBACK_MESSAGE, SYSTEM_PROMPT, build_rag_prompt
from app.config import settings
from app.rag.retriever import format_chunks_for_prompt, retrieve
from app.tools.registry import TOOL_DEFINITIONS, execute_tool


ResponseSource = Literal["llm", "tool", "rag"]


@dataclass(frozen=True)
class AgentResponse:
    reply: str
    source: ResponseSource
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
class AgentRouter :
    """Routes a user message through tools, RAG, or direct LLM response."""
    
    def __init__(self) -> None:
        self._client = Mistral(api_key=settings.MISTRAL_API_KEY)
        self._model = settings.MISTRAL_CHAT_MODEL
        
    def process(
        self, 
        session_id : str, 
        user_message: str ,
        source_files: list[str] | None = None,
    ) -> AgentResponse:
        message = user_message.strip()
        if not message :
            return AgentResponse(reply="Please send a message.", source="llm")
        
        
        memory = memory_store.get_or_create_session(session_id)

        messages = self._build_base_messages(memory.get_history(), message)
        first_response = self._call_llm(messages=messages, tools=TOOL_DEFINITIONS)

        finish_reason = first_response["finish_reason"]
        assistant_message = first_response["message"]
        
        if finish_reason == "tool_calls":
            tool_response = self._handle_tool_calls(
                messages=messages,
                assistant_message=assistant_message,
            )

            memory.add_turn("user", message)
            memory.add_turn("assistant", tool_response.reply)
            
            return tool_response
    
        direct_reply = self._message_content(assistant_message)
        
        if self._should_try_rag(direct_reply) :
            rag_reply = self._answer_with_rag(
                user_message = message,
                history = memory.get_history(),
                source_files=source_files,
            )
            
            memory.add_turn("user",message)
            memory.add_turn("assistant",rag_reply.reply)
            
            return rag_reply
        
        memory.add_turn("user", message)
        memory.add_turn("assistant", direct_reply)
        
        return AgentResponse(
            reply=direct_reply,
            source="llm",
            metadata={"finish_reason": finish_reason},
        )
    
    
    def _build_base_messages(
        self,
        history: list[dict[str, str]],
        user_message: str,
    ) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": user_message},
        ]    
    
    
    def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools : list[dict[str,Any]] | None = None,
    ) -> dict[str, Any]:
        response = self._client.chat.complete(
            model= self._model,
            messages= messages,
            tools=tools,
            tool_choice="auto" if tools else None,
            temperature=0.2,
        )
        
        choice = response.choices[0]
        
        return {
            "finish_reason": choice.finish_reason,
            "message": choice.message,            
        }
        
        
    def _handle_tool_calls(
        self,
        messages: list[dict[str, Any]],
        assistant_message: Any,
    ) -> AgentResponse:
        tool_calls = list(getattr(assistant_message, "tool_calls", None) or [])

        if not tool_calls:
            return AgentResponse(
                reply="I could not determine which tool to use.",
                source="tool",
                metadata={"tool_results": []},
            )

        messages.append(self._assistant_message_to_dict(assistant_message))

        tool_results: list[dict[str, Any]] = []

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = self._parse_tool_arguments(tool_call.function.arguments)
            tool_result = execute_tool(tool_name, tool_args)

            tool_results.append(
                {
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "tool_result": tool_result,
                }
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": getattr(tool_call, "id", None),
                    "name": tool_name,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
            )

        final_response = self._call_llm(messages=messages)
        final_reply = self._message_content(final_response["message"])

        return AgentResponse(
            reply=final_reply or "I used the tool, but could not create a final answer.",
            source="tool",
            metadata={
                "tool_results": tool_results,
                "finish_reason": final_response["finish_reason"],
            },
        )
    
    
    def _answer_with_rag(
        self,
        user_message: str,
        history: list[dict[str, str]],
        source_files: list[str] | None = None,
    ) -> AgentResponse:
        chunks = retrieve(user_message, source_files=source_files) 

        if not chunks:
            return AgentResponse(
                reply=FALLBACK_MESSAGE,
                source="rag",
                metadata={"chunks": []},
            )

        context = format_chunks_for_prompt(chunks)
        rag_prompt = build_rag_prompt(user_message=user_message, context=context)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": rag_prompt},
        ]

        response = self._call_llm(messages=messages)
        reply = self._message_content(response["message"])

        return AgentResponse(
            reply=reply or FALLBACK_MESSAGE,
            source="rag",
            metadata={
                "chunks": chunks,
                "finish_reason": response["finish_reason"],
            },
        )

    

    @staticmethod
    def _message_content(message: Any) -> str:
        content = getattr(message, "content", "") or ""

        if isinstance(content, str):
            return content.strip()

        return str(content).strip()

    @staticmethod
    def _assistant_message_to_dict(message: Any) -> dict[str, Any]:
        tool_calls = []

        for tool_call in list(getattr(message, "tool_calls", None) or []):
            tool_calls.append(
                {
                    "id": getattr(tool_call, "id", None),
                    "type": getattr(tool_call, "type", "function"),
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            )

        return {
            "role": "assistant",
            "content": getattr(message, "content", None),
            "tool_calls": tool_calls,
        }
 

    @staticmethod
    def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments

        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return {}

            if isinstance(parsed, dict):
                return parsed

        return {}

    @staticmethod
    def _should_try_rag(reply: str) -> bool:
        normalized_reply = reply.lower()

        fallback_signals = [
            FALLBACK_MESSAGE.lower(),
            "uploaded document",
            "uploaded documents",
            "document context",
            "i don't have enough information",
            "i do not have enough information",
        ]

        return any(signal in normalized_reply for signal in fallback_signals)


agent_router = AgentRouter() 
