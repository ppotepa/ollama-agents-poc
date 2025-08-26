"""OpenAI-compatible data models and schemas."""

import time
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ToolCall(BaseModel):
    """OpenAI-compatible tool call structure."""
    id: str
    type: str = "function"
    function: Dict[str, Any]


class ChatMessage(BaseModel):
    """OpenAI-compatible chat message structure."""
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str
    messages: List[ChatMessage]
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = "auto"
    stream: bool = False
    temperature: float = 0.7
    max_tokens: Optional[int] = None


def create_tool_call_payload(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Create a tool call payload in OpenAI format.
    
    Args:
        name: Tool function name
        arguments: Tool function arguments
        
    Returns:
        Tool call payload dictionary
    """
    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments if isinstance(arguments, str) else str(arguments)
        }
    }


def create_tool_response(choices_tool_calls: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    """Create a tool response in OpenAI format.
    
    Args:
        choices_tool_calls: List of tool calls
        model: Model name used
        
    Returns:
        OpenAI-compatible response dictionary
    """
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": choices_tool_calls
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }


def create_text_response(content: str, model: str) -> Dict[str, Any]:
    """Create a text response in OpenAI format.
    
    Args:
        content: Response content
        model: Model name used
        
    Returns:
        OpenAI-compatible response dictionary
    """
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }


def generate_id(prefix: str = "msg") -> str:
    """Generate a unique ID with prefix.
    
    Args:
        prefix: ID prefix
        
    Returns:
        Unique ID string
    """
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def get_current_timestamp() -> int:
    """Get current Unix timestamp.
    
    Returns:
        Current timestamp as integer
    """
    return int(time.time())
