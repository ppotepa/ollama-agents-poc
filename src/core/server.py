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

# Import the new integration system
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from integrations import IntegrationManager, AgentRegistry, ModelConfigReader

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

# Initialize integration manager, agent registry, and model config reader
integration_manager = IntegrationManager()
agent_registry = AgentRegistry()
model_config_reader = ModelConfigReader()

def get_enhanced_models() -> List[Dict[str, Any]]:
    """
    Get models enhanced with YAML configuration data.
    
    Merges integration models with YAML config to provide:
    - Short names from YAML
    - Enhanced metadata and capabilities
    - Configuration parameters
    """
    # Get models from integrations (Ollama, etc.)
    integration_models = integration_manager.get_all_models()
    
    # Get model configurations from YAML
    yaml_models = model_config_reader.get_all_models()
    
    # Create lookup map for YAML configs by model_id
    yaml_lookup = {config.model_id: config for config in yaml_models}
    
    enhanced_models = []
    
    for integration_model in integration_models:
        model_id = integration_model.get("id", "")
        yaml_config = yaml_lookup.get(model_id)
        
        if yaml_config:
            # Enhance with YAML configuration
            enhanced_model = {
                **integration_model,  # Base integration data
                "short_name": yaml_config.short_name,
                "display_name": yaml_config.name,
                "description": yaml_config.description,
                "provider": yaml_config.provider,
                "yaml_capabilities": yaml_config.capabilities,
                "parameters": yaml_config.parameters,
                "tools": yaml_config.tools,
                "system_message": yaml_config.system_message,
                "supports_coding": yaml_config.supports_coding,
                "supports_file_operations": yaml_config.supports_file_operations,
                "supports_streaming": yaml_config.supports_streaming,
                "has_yaml_config": True
            }
        else:
            # Use integration data only
            enhanced_model = {
                **integration_model,
                "short_name": model_id.split(":")[0] if ":" in model_id else model_id,
                "display_name": integration_model.get("id", "Unknown"),
                "description": f"Model from {integration_model.get('source', 'unknown')} integration",
                "has_yaml_config": False
            }
        
        enhanced_models.append(enhanced_model)
    
    # Add YAML-only models (not found in integrations)
    integration_ids = {model.get("id") for model in integration_models}
    for yaml_config in yaml_models:
        if yaml_config.model_id not in integration_ids:
            # YAML-defined model not available in integrations
            enhanced_model = {
                "id": yaml_config.model_id,
                "object": "model",
                "created": _now(),
                "owned_by": yaml_config.provider,
                "permission": [],
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
                "tool_resources": yaml_config.tools,
                "type": "chat.completions",
                "short_name": yaml_config.short_name,
                "display_name": yaml_config.name,
                "description": yaml_config.description,
                "provider": yaml_config.provider,
                "yaml_capabilities": yaml_config.capabilities,
                "parameters": yaml_config.parameters,
                "tools": yaml_config.tools,
                "system_message": yaml_config.system_message,
                "supports_coding": yaml_config.supports_coding,
                "supports_file_operations": yaml_config.supports_file_operations,
                "supports_streaming": yaml_config.supports_streaming,
                "has_yaml_config": True,
                "source": "yaml_config",
                "available": False,  # Not available in integrations
                "details": {
                    "size": "Unknown",
                    "family": "unknown",
                    "status": "configured but not available"
                }
            }
            enhanced_models.append(enhanced_model)
    
    return enhanced_models

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

@app.get("/health")
def health():
    """Simple health check endpoint for connection modes."""
    return {"status": "ok", "version": VERSION}

@app.get("/v1/integrations/health")
def integrations_health():
    """Get health status of all integrations."""
    return integration_manager.health_check()

@app.get("/v1/integrations")
def list_integrations():
    """List all available integrations."""
    return {
        "integrations": integration_manager.list_integrations(),
        "available": integration_manager.get_available_integrations()
    }

@app.get("/v1/models/all")
def list_all_models():
    """List all models (both available and configured)."""
    enhanced_models = get_enhanced_models()
    
    return {
        "object": "list", 
        "data": enhanced_models,
        "total": len(enhanced_models),
        "available": len([m for m in enhanced_models if m.get("available", True)]),
        "configured": len([m for m in enhanced_models if m.get("has_yaml_config", False)])
    }

@app.get("/v1/models/with-agents")
def list_models_with_agents():
    """List models with their agent implementation status."""
    enhanced_models = get_enhanced_models()
    matches = agent_registry.match_models_with_agents(enhanced_models)
    
    # Separate models with and without agents
    with_agents = []
    without_agents = []
    
    for match in matches:
        model_data = {
            "model_id": match.model_id,
            "model_info": match.model_info,
            "has_agent": match.has_agent,
            "match_confidence": match.match_confidence,
            "match_reason": match.match_reason,
            "short_name": match.model_info.get("short_name", "unknown"),
            "display_name": match.model_info.get("display_name", "Unknown"),
            "has_yaml_config": match.model_info.get("has_yaml_config", False)
        }
        
        if match.has_agent and match.agent_info:
            model_data["agent_info"] = {
                "name": match.agent_info.name,
                "description": match.agent_info.description,
                "capabilities": match.agent_info.capabilities,
                "family": match.agent_info.family
            }
            with_agents.append(model_data)
        else:
            without_agents.append(model_data)
    
    return {
        "with_agents": with_agents,
        "without_agents": without_agents,
        "summary": {
            "total_models": len(matches),
            "with_agents": len(with_agents),
            "without_agents": len(without_agents),
            "coverage_percentage": round((len(with_agents) / len(matches)) * 100, 1) if matches else 0
        }
    }

@app.get("/v1/models/short-names")
def list_model_short_names():
    """List models with their short names for easy reference."""
    enhanced_models = get_enhanced_models()
    
    short_name_list = []
    for model in enhanced_models:
        short_name_list.append({
            "short_name": model.get("short_name", "unknown"),
            "model_id": model.get("id", "unknown"),
            "display_name": model.get("display_name", "Unknown"),
            "available": model.get("available", True),
            "has_config": model.get("has_yaml_config", False),
            "supports_coding": model.get("supports_coding", False),
            "provider": model.get("provider", "unknown")
        })
    
    return {
        "models": sorted(short_name_list, key=lambda x: x["short_name"]),
        "total": len(short_name_list)
    }

@app.get("/v1/agents")
def list_agents():
    """List all available agent implementations."""
    agents = agent_registry.list_agents()
    
    agent_data = []
    for agent in agents:
        agent_data.append({
            "name": agent.name,
            "description": agent.description,
            "family": agent.family,
            "capabilities": agent.capabilities,
            "model_patterns": agent.model_patterns,
            "module_path": agent.module_path
        })
    
    return {
        "agents": agent_data,
        "total": len(agent_data)
    }


@app.post("/v1/agents/query")
def agent_query(request: Dict[str, Any] = Body(...)):
    """Direct agent query endpoint - our native implementation."""
    try:
        agent_name = request.get("agent", "deepcoder")
        query = request.get("query", "")
        
        if not query:
            return {"error": "Query is required", "status": "error"}
        
        print(f"🔍 Agent Query: Processing '{query}' with agent '{agent_name}'")
        
        # Check if this is a repository analysis query and tools are available
        repo_keywords = ["repository", "repo", "structure", "analyze", "analysis", "files", "directories", "languages"]
        is_repo_query = any(keyword in query.lower() for keyword in repo_keywords)
        
        if is_repo_query:
            print(f"🔍 Detected repository analysis query, checking for context...")
            try:
                # Try to use repository context tools directly
                import sys
                import os
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
                from tools.context import get_repository_state, analyze_repository_context, analyze_repo_languages
                
                # Get repository state to see if context is loaded
                repo_state = get_repository_state()
                if "Total Files:" in repo_state and not repo_state.startswith("❌"):
                    print(f"🧠 Repository context available, using tools")
                    
                    # Build comprehensive repository analysis
                    context_analysis = analyze_repository_context()
                    language_analysis = analyze_repo_languages()
                    
                    # Create enhanced prompt with repository data
                    enhanced_query = f"""Based on the repository context data below, please analyze the repository structure:

Repository Overview:
{context_analysis}

Programming Languages:
{language_analysis}

Repository State:
{repo_state}

Original question: {query}

Please provide a comprehensive analysis based on this repository data."""
                    
                    query = enhanced_query
                    print(f"🔧 Enhanced query with repository context data")
                    
            except Exception as e:
                print(f"⚠️ Could not access repository context tools: {e}")
        
        # Import our agent system
        from core.single_query_mode import run_single_query_direct_ollama
        
        # Use our agent system to process the query
        result = run_single_query_direct_ollama(query, agent_name)
        
        if result and not result.startswith("❌"):
            print(f"✅ Agent Query: Success ({len(result)} chars)")
            return {
                "status": "success",
                "agent": agent_name,
                "query": query,
                "response": result,
                "response_length": len(result)
            }
        else:
            print(f"⚠️ Agent Query: Failed or error result")
            return {
                "status": "error",
                "agent": agent_name,
                "query": query,
                "error": result
            }
            
    except Exception as e:
        print(f"❌ Agent Query: Exception: {e}")
        return {
            "status": "error",
            "error": str(e)
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
    """List available models with YAML configuration data."""
    enhanced_models = get_enhanced_models()
    
    # Filter to only show available models (from integrations)
    available_models = [model for model in enhanced_models if model.get("available", True)]
    
    return {
        "object": "list",
        "data": available_models
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

    # 2) Try to process with actual agents first
    try:
        # Import our agent system
        from core.single_query_mode import run_single_query_direct_ollama
        
        print(f"🔍 Server: Processing query '{user_prompt}' with agent '{model}'")
        
        # Use our agent system to process the query
        result = run_single_query_direct_ollama(user_prompt, model)
        
        if result and not result.startswith("❌"):
            print(f"✅ Server: Agent processed successfully ({len(result)} chars)")
            return text_response(result, model)
        else:
            print(f"⚠️ Server: Agent processing failed or returned error")
            
    except Exception as e:
        print(f"⚠️ Server: Agent processing error: {e}")
    
    # 3) If tools are enabled, emit helpful tool_calls based on simple intent
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

    # 4) Fallback: plain text with agent info
    return text_response("Agent mode is available. Enable tools to let me edit files or run commands.", model)

# --------------------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    # Tip: run inside a venv to avoid PEP 668 issues:
    # python3 -m venv venv && source venv/bin/activate && pip install fastapi uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
