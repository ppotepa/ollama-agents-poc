"""API route handlers for the server."""

import sys
import os
from typing import Any, Dict, List

# Add path for integrations
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from src.utils.enhanced_logging import get_logger

from .models import ChatRequest, create_text_response, create_tool_response
from .tool_processor import ToolCallProcessor

try:
    from integrations import AgentRegistry, IntegrationManager, ModelConfigReader
except ImportError:
    AgentRegistry = None
    IntegrationManager = None
    ModelConfigReader = None


class APIRouteHandler:
    """Handles API routes and request processing."""

    def __init__(self, app: FastAPI):
        """Initialize the route handler.
        
        Args:
            app: FastAPI application instance
        """
        self.app = app
        self.logger = get_logger()
        self.tool_processor = ToolCallProcessor()
        
        # Initialize integrations if available
        self.agent_registry = AgentRegistry() if AgentRegistry else None
        self.integration_manager = IntegrationManager() if IntegrationManager else None
        self.model_config = ModelConfigReader() if ModelConfigReader else None
        
        # Register routes
        self._register_routes()

    def _register_routes(self):
        """Register all API routes."""
        
        @self.app.get("/healthz")
        def healthz():
            """Health check endpoint."""
            return {"status": "ok", "service": "continue-agent-shim"}

        @self.app.get("/health")
        def health():
            """Alternative health check endpoint."""
            return healthz()

        @self.app.get("/v1/integrations/health")
        def integrations_health():
            """Integration system health check."""
            return {"integrations_status": "ok", "manager_available": self.integration_manager is not None}

        @self.app.get("/v1/integrations")
        def list_integrations():
            """List available integrations."""
            if self.integration_manager:
                return {"integrations": self.integration_manager.list_integrations()}
            return {"integrations": []}

        @self.app.get("/v1/models/all")
        def list_all_models():
            """List all available models."""
            return {"data": self.get_enhanced_models()}

        @self.app.get("/v1/models")
        def list_models():
            """OpenAI-compatible models endpoint."""
            return {"object": "list", "data": self.get_enhanced_models()}

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatRequest):
            """OpenAI-compatible chat completions endpoint."""
            return await self.handle_chat_completion(request)

    def get_enhanced_models(self) -> List[Dict[str, Any]]:
        """Get enhanced model list with tool support information.
        
        Returns:
            List of enhanced model dictionaries
        """
        try:
            if self.model_config:
                base_models = self.model_config.get_available_models()
            else:
                # Fallback models
                base_models = [
                    {"id": "qwen2.5-coder:7b-instruct", "name": "Qwen2.5 Coder 7B"},
                    {"id": "qwen2.5:7b", "name": "Qwen2.5 7B"},
                    {"id": "llama3.2:3b", "name": "Llama 3.2 3B"},
                    {"id": "phi3:mini", "name": "Phi-3 Mini"}
                ]

            enhanced_models = []
            for model in base_models:
                model_id = model.get("id", model.get("name", "unknown"))
                
                enhanced_model = {
                    "id": model_id,
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": "ollama",
                    "name": model.get("name", model_id),
                    "description": f"Enhanced {model_id} with tool support",
                    
                    # Tool support metadata
                    "supports_tools": True,
                    "supports_function_calling": True,
                    "tool_types": ["function"],
                    
                    # Continue-specific metadata
                    "continue_compatible": True,
                    "agent_mode": True,
                    "streaming": True,
                    
                    # Model capabilities
                    "capabilities": {
                        "completion": True,
                        "chat": True,
                        "tools": True,
                        "streaming": True
                    },
                    
                    # Available tools
                    "available_tools": [tool["function"]["name"] for tool in self.tool_processor.get_available_tools()]
                }
                
                enhanced_models.append(enhanced_model)

            self.logger.debug(f"Enhanced {len(enhanced_models)} models with tool support")
            return enhanced_models

        except Exception as e:
            self.logger.error(f"Error getting enhanced models: {e}")
            return []

    async def handle_chat_completion(self, request: ChatRequest) -> Dict[str, Any]:
        """Handle chat completion requests.
        
        Args:
            request: Chat completion request
            
        Returns:
            OpenAI-compatible response
        """
        try:
            self.logger.info(f"Chat completion request for model: {request.model}")
            
            # Get the last user message
            user_messages = [msg for msg in request.messages if msg.role == "user"]
            if not user_messages:
                return create_text_response("No user message provided", request.model)
            
            last_message = user_messages[-1]
            user_text = last_message.content or ""

            # Check if tools are requested and available
            has_tools = request.tools and len(request.tools) > 0
            tool_choice = request.tool_choice
            
            # Try to extract tool calls from user message
            extracted_tools = self.tool_processor.extract_tool_calls_from_text(user_text)
            
            if extracted_tools and has_tools:
                # Return tool calls response
                self.logger.info(f"Returning {len(extracted_tools)} tool calls")
                return create_tool_response(extracted_tools, request.model)
            
            elif has_tools and tool_choice == "auto":
                # Analyze if tools should be used
                should_use_tools = self._should_use_tools(user_text, request.tools)
                
                if should_use_tools:
                    # Generate appropriate tool calls
                    suggested_tools = self._suggest_tools(user_text, request.tools)
                    if suggested_tools:
                        self.logger.info(f"Suggesting {len(suggested_tools)} tool calls")
                        return create_tool_response(suggested_tools, request.model)
            
            # Use agent registry if available for text generation
            if self.agent_registry:
                try:
                    agent = self.agent_registry.get_agent(request.model)
                    if agent:
                        response_text = await self._generate_with_agent(agent, user_text)
                        return create_text_response(response_text, request.model)
                except Exception as e:
                    self.logger.error(f"Agent generation failed: {e}")
            
            # Fallback response
            return create_text_response(
                f"This is a simulated response from {request.model}. "
                f"Your message: {user_text[:100]}...",
                request.model
            )

        except Exception as e:
            self.logger.error(f"Error handling chat completion: {e}")
            return create_text_response(f"Error processing request: {str(e)}", request.model)

    def _should_use_tools(self, text: str, available_tools: List[Dict[str, Any]]) -> bool:
        """Determine if tools should be used based on user text.
        
        Args:
            text: User message text
            available_tools: List of available tools
            
        Returns:
            True if tools should be used
        """
        text_lower = text.lower()
        
        # Keywords that suggest tool usage
        tool_keywords = [
            "create", "write", "edit", "read", "file", "search", "run", "execute",
            "command", "terminal", "directory", "folder", "save", "open"
        ]
        
        return any(keyword in text_lower for keyword in tool_keywords)

    def _suggest_tools(self, text: str, available_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suggest appropriate tools based on user text.
        
        Args:
            text: User message text
            available_tools: List of available tools
            
        Returns:
            List of suggested tool calls
        """
        suggestions = []
        text_lower = text.lower()
        
        # Simple keyword-based tool suggestion
        if "create" in text_lower and "file" in text_lower:
            suggestions.append(self.tool_processor.normalize_tool_call({
                "name": "files:create",
                "arguments": {"path": "example.txt", "content": "Generated content"}
            }))
        
        elif "read" in text_lower and "file" in text_lower:
            suggestions.append(self.tool_processor.normalize_tool_call({
                "name": "files:read", 
                "arguments": {"path": "example.txt"}
            }))
        
        elif "run" in text_lower or "execute" in text_lower:
            suggestions.append(self.tool_processor.normalize_tool_call({
                "name": "terminal:run",
                "arguments": {"command": "echo 'Hello World'"}
            }))

        return [s for s in suggestions if s is not None]

    async def _generate_with_agent(self, agent, text: str) -> str:
        """Generate response using an agent.
        
        Args:
            agent: Agent instance
            text: Input text
            
        Returns:
            Generated response text
        """
        try:
            if hasattr(agent, "agenerate"):
                return await agent.agenerate(text)
            elif hasattr(agent, "generate"):
                return agent.generate(text)
            elif hasattr(agent, "query"):
                return agent.query(text)
            else:
                return f"Agent {agent} processed: {text}"
        except Exception as e:
            self.logger.error(f"Agent generation error: {e}")
            return f"Agent processing failed: {str(e)}"
