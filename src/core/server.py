#!/usr/bin/env python3
# server.py — Minimal OpenAI-compatible agent shim for Continue Agent Mode
# - Advertises agent + tools support
# - Normalizes hallucinated tool names (e.g., create_new_file -> files:create)
# - Emits proper tool_calls for Continue to apply

import os
import time
import uuid
import json
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

VERSION = "agent-shim-v2-2025-08-20"

# --------------------------------------------------------------------------------------
# FastAPI app + CORS
# --------------------------------------------------------------------------------------
app = FastAPI(title="Continue Agent Shim", version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------------------
# OpenAI-like schemas
# --------------------------------------------------------------------------------------
class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Dict[str, Any]

class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

class ChatRequest(BaseModel):
    model: Optional[str] = "deepseek-agent"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.5
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None  # Important for Agent Mode

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _now() -> int:
    return int(time.time())

def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"

def tool_call_payload(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Build a single tool_call entry (OpenAI format)."""
    return {
        "id": _new_id("call"),
        "type": "function",
        "function": {
            "name": name,
            # OpenAI format expects a JSON-serialized string for arguments
            "arguments": json.dumps(arguments),
        },
    }

def tool_response(choices_tool_calls: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    """Wrap tool calls into a /chat/completions response."""
    return {
        "id": _new_id("chatcmpl"),
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": choices_tool_calls,
            },
            "finish_reason": "tool_calls",
        }],
    }

def text_response(content: str, model: str) -> Dict[str, Any]:
    return {
        "id": _new_id("chatcmpl"),
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
    }

def parse_user_tool_json(s: str) -> Optional[Dict[str, Any]]:
    """If the user pasted a raw tool JSON {name, arguments}, parse it."""
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            return obj
    except Exception:
        pass
    return None

def normalize_tool_call(tool: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Map hallucinated tool names to Continue tools."""
    name = tool.get("name")
    args = tool.get("arguments", {}) or {}

    # Common aliases -> files/terminal
    if name in ("create_new_file", "new_file", "make_file"):
        if "filepath" in args:
            return tool_call_payload("files", {
                "action": "create",
                "filepath": args.get("filepath"),
                "contents": args.get("contents", ""),
            })
        return None

    if name in ("edit_file", "write_file", "append_file"):
        if "filepath" in args:
            return tool_call_payload("files", {
                "action": "write",
                "filepath": args.get("filepath"),
                "contents": args.get("contents", ""),
            })
        return None

    if name in ("open_file", "read_file"):
        if "filepath" in args:
            return tool_call_payload("files", {
                "action": "read",
                "filepath": args.get("filepath"),
            })
        return None

    if name in ("delete_file", "remove_file"):
        if "filepath" in args:
            return tool_call_payload("files", {
                "action": "delete",
                "filepath": args.get("filepath"),
            })
        return None

    if name in ("run_terminal_command", "exec", "terminal", "shell"):
        cmd = args.get("command") or args.get("cmd")
        if cmd:
            return tool_call_payload("terminal", {"command": cmd})
        return None

    # Already a valid Continue tool? Pass it through (ensure arguments are str)
    if name in ("files", "terminal", "code", "docs", "diff", "problems", "folder", "codebase"):
        # Ensure arguments are JSON string
        arguments = args if isinstance(args, dict) else {}
        return {
            "id": _new_id("call"),
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }

    # Unknown tool
    return None

# --------------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "version": VERSION,
        "cwd": os.getcwd(),
        "file": __file__,
    }

@app.get("/v1/capabilities")
def capabilities():
    return {
        "agent_mode": True,
        "tools": ["code", "files", "terminal", "docs", "diff", "problems", "folder", "codebase"],
        "function_calling": True,
        "tool_calling": True,
    }

@app.get("/v1/models")
def list_models():
    # Single logical model exposed: "deepseek-agent"
    return {
        "object": "list",
        "data": [{
            "id": "deepseek-agent",
            "object": "model",
            "created": _now(),
            "owned_by": "local",
            "permission": [],
            # Flags Continue looks for:
            "supports_agent": True,
            "supports_tools": True,
            "supports_tool_calls": True,
            "supports_function_calling": True,
            "capabilities": {
                "agent": True,
                "tools": True,
                "tool_calls": True,
                "functions": True,
                "function_calling": True,
            },
            "tool_resources": ["code", "files", "terminal", "docs", "diff", "problems", "folder", "codebase"],
            "type": "chat.completions",
        }],
    }

@app.post("/v1/chat/completions")
def chat_completions(body: ChatRequest = Body(...)):
    model = body.model or "deepseek-agent"

    # Flatten the latest user message
    user_texts = [m.content for m in body.messages if m.role == "user" and m.content]
    user_prompt = (user_texts[-1] if user_texts else "") or ""

    # 1) If user pasted raw tool JSON, normalize & return as tool_calls
    pasted = parse_user_tool_json(user_prompt.strip())
    if pasted:
        tc = normalize_tool_call(pasted)
        if tc:
            return tool_response([tc], model)
        return text_response("I couldn't parse that tool call. Please provide filepath/contents/command as needed.", model)

    # 2) If tools are enabled, emit helpful tool_calls based on simple intent
    if body.tools:
        low = user_prompt.lower()

        # Heuristic: "create file ..." → files:create
        if ("create" in low and "file" in low) or ("new file" in low):
            # Quick path: try to extract a filename
            # (Continue will still show a picker if path isn't valid.)
            return tool_response([
                tool_call_payload("files", {
                    "action": "create",
                    "filepath": "src/hello.txt",
                    "contents": "Hello World",
                })
            ], model)

        # Heuristic: "run/execute ..." → terminal
        if any(k in low for k in ("run ", "execute", "terminal", "shell", "cmd ")):
            return tool_response([
                tool_call_payload("terminal", {"command": "echo Hello from agent"})
            ], model)

        # Fallback: offer a code snippet via `code` tool
        return tool_response([
            tool_call_payload("code", {
                "content": "// Example\nconsole.log('Hello from agent shim');"
            })
        ], model)

    # 3) No tools: plain text
    return text_response("Agent mode is available. Enable tools to let me edit files or run commands.", model)

# --------------------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    # Tip: run inside a venv to avoid PEP 668 issues:
    # python3 -m venv venv && source venv/bin/activate && pip install fastapi uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
