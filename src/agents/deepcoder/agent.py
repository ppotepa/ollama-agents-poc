"""DeepCoderAgent - concrete implementation using Ollama backend."""
from __future__ import annotations

from typing import Any, Dict, List
from src.agents.base.base_agent import AbstractAgent
from src.utils.animations import stream_text

try:  # LangChain optional
    from langchain_ollama import ChatOllama
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False
    # Define dummy classes to avoid import errors
    class ChatOllama: pass
    class AgentExecutor: pass
    def create_tool_calling_agent(*args, **kwargs): return None
    class ChatPromptTemplate: 
        @staticmethod
        def from_messages(*args): return None

from src.tools.registry import get_registered_tools  # registry list
import src.tools  # noqa: F401  # trigger side-effect imports & registration


class DeepCoderAgent(AbstractAgent):
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        self._agent_executor = None
        self._supports_tools = None  # Cache tool support detection
        
    def _check_tool_support(self) -> bool:
        """Check if the current model supports tool calling."""
        if self._supports_tools is not None:
            return self._supports_tools
            
        if not LANGCHAIN_AVAILABLE or not self._llm:
            self._supports_tools = False
            return False
        
        model_id = self.config.get("model_id", "unknown")
        
        try:
            # Test tool support with a simple function call
            test_prompt = "Hello, can you help me?"
            test_result = self._llm.invoke(test_prompt)
            
            # If we can create a basic agent without tools, the model works
            # But we need to test actual tool calling capability
            try:
                # Try to create a minimal tool-calling setup
                from langchain.tools import tool
                
                @tool
                def test_tool() -> str:
                    """Test tool for capability detection."""
                    return "test"
                
                test_agent = create_tool_calling_agent(
                    self._llm, 
                    [test_tool], 
                    ChatPromptTemplate.from_messages([("human", "{input}")])
                )
                
                if test_agent:
                    self._supports_tools = True
                    stream_text(f"✅ Model {model_id} supports tools")
                    
                    # Log successful tool support
                    try:
                        from src.utils.enhanced_logging import get_logger
                        logger = get_logger()
                        logger.log_model_compatibility(
                            model=model_id,
                            supports_tools=True,
                            capabilities=["tool_calling", "agent_executor"],
                            error_details=None
                        )
                    except Exception:
                        pass  # Don't fail if logging fails
                else:
                    self._supports_tools = False
                    stream_text(f"⚠️  Model {model_id} does not support tools - using LLM fallback")
                    
            except Exception as tool_error:
                self._supports_tools = False
                error_msg = str(tool_error)
                stream_text(f"⚠️  Tool support detection failed for {model_id}: {error_msg}")
                stream_text("📝 Continuing with direct LLM mode (no tools)")
                
                # Log tool support failure
                try:
                    from src.utils.enhanced_logging import get_logger
                    logger = get_logger()
                    logger.log_model_compatibility(
                        model=model_id,
                        supports_tools=False,
                        capabilities=["llm_only"],
                        error_details=error_msg
                    )
                except Exception:
                    pass  # Don't fail if logging fails
                
        except Exception as e:
            self._supports_tools = False
            stream_text(f"❌ Model connectivity test failed: {e}")
            
        return self._supports_tools

    def stream(self, prompt: str, on_token):  # type: ignore[override]
        if not self._loaded:
            self.load()
        if not LANGCHAIN_AVAILABLE or self._llm is None:
            return super().stream(prompt, on_token)
        final_tokens: List[str] = []
        try:
            for chunk in self._llm.stream(prompt):  # type: ignore[attr-defined]
                text = getattr(chunk, "content", None) or str(chunk)
                if text:
                    on_token(text)
                    final_tokens.append(text)
        except Exception as e:
            stream_text(f"❌ Streaming error: {e}")
            return ""
        return "".join(final_tokens)
    
    def stream_with_context(self, prompt: str, on_token, context: Dict[str, Any]):
        """Stream response with additional context from interceptor."""
        if context and 'interceptor_analysis' in context:
            interceptor_data = context['interceptor_analysis']
            
            # Create enhanced prompt with interceptor context
            enhanced_prompt = f"""INTERCEPTOR ANALYSIS:
Intent: {interceptor_data['detected_intent']} (confidence: {interceptor_data['confidence']:.2%})
Commands executed: {interceptor_data['execution_stats']['total_commands']} 
Successful commands: {interceptor_data['execution_stats']['successful_commands']}
Data gathered: {interceptor_data['execution_stats']['total_data_gathered']} chars
Context types: {', '.join(interceptor_data['context_types'])}

COMMAND EXECUTION DETAILS:
{chr(10).join([f"- {cmd['command']}: {'✅' if cmd['success'] else '❌'} ({cmd['duration']:.3f}s, {cmd['result_length']} chars)" for cmd in interceptor_data['commands_executed']])}

USER QUERY:
{prompt}"""
            
            return self.stream(enhanced_prompt, on_token)
        else:
            return self.stream(prompt, on_token)
    
    def _build_llm(self) -> Any:  # noqa: D401
        if not LANGCHAIN_AVAILABLE:
            return None
        model_id = self.config.get("backend_image") or self.config.get("model_id")
        params = self.config.get("parameters", {})
        llm = ChatOllama(
            model=model_id,
            streaming=True,
            **{k: v for k, v in params.items() if isinstance(k, str)}
        )
        return llm

    def _build_tools(self) -> List[Any]:  # noqa: D401
        desired = set(self.config.get("tools", []))
        selected: List[Any] = []
        for t in get_registered_tools():
            name = getattr(t, "name", None)
            if name and name in desired:
                selected.append(t)
        if desired and not selected:
            stream_text("⚠️ No matching tools found for this agent configuration")
        elif selected:
            stream_text(f"🔧 Loaded {len(selected)} tools: {[getattr(t, 'name', str(t)) for t in selected]}")
        return selected

    def _build_agent_executor(self):
        """Build an agent executor that can use tools."""
        if not LANGCHAIN_AVAILABLE or not self._llm or not self._tools:
            stream_text("⚠️  Cannot build agent executor: missing requirements")
            return None
        
        # Check if model supports tools before attempting to create executor
        if not self._check_tool_support():
            stream_text(f"📝 Model {self.config.get('model_id', 'unknown')} does not support tools")
            stream_text("🔄 Continuing in direct LLM mode (tools disabled)")
            return None
        
        try:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", self.config.get("system_message", "You are a helpful AI assistant.")),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}")
            ])
            
            agent = create_tool_calling_agent(self._llm, self._tools, prompt_template)
            self._agent_executor = AgentExecutor(
                agent=agent,
                tools=self._tools,
                verbose=False,
                max_iterations=3,
                handle_parsing_errors=True
            )
            
            stream_text(f"✅ Agent executor created with {len(self._tools)} tools")
            return self._agent_executor
            
        except Exception as e:
            self._supports_tools = False  # Mark as unsupported for future calls
            stream_text(f"❌ Failed to build agent executor: {e}")
            stream_text("🔄 Falling back to direct LLM mode")
            return None

    def load(self) -> None:
        """Load the agent and create agent executor with tools if supported."""
        super().load()
        
        if LANGCHAIN_AVAILABLE and self._llm:
            # Only try to build agent executor if we have tools and the model supports them
            if self._tools and len(self._tools) > 0:
                stream_text(f"🔧 Attempting to configure {len(self._tools)} tools...")
                self._build_agent_executor()
                
                if not self._agent_executor:
                    stream_text("📝 Tools not supported by this model - continuing in LLM-only mode")
            else:
                stream_text("📝 No tools configured - running in LLM-only mode")
        else:
            stream_text("📝 LangChain not available - running in basic mode")

    def run(self, prompt: str) -> str:
        if not self._loaded:
            self.load()
        
        # If we have a proper agent executor with tools, use it
        if self._agent_executor:
            try:
                result = self._agent_executor.invoke({"input": prompt})
                return result.get("output", "No response generated")
            except Exception as e:
                stream_text(f"❌ Agent executor error: {e}")
                stream_text("🔄 Falling back to direct LLM mode")
                # Mark tools as unsupported for future calls
                self._supports_tools = False
                self._agent_executor = None
        
        # Fallback: direct LLM call (no tools)
        if LANGCHAIN_AVAILABLE and self._llm:
            try:
                # Create enhanced prompt that acknowledges tool limitations
                enhanced_prompt = f"""You are a helpful AI assistant. While you don't have access to tools like file reading or command execution, please provide the best analysis you can based on the information given.

USER QUERY: {prompt}

Please analyze this query and provide helpful insights based on your knowledge. If the query seems to require file access or command execution, explain what you would look for and provide general guidance."""

                response = self._llm.invoke(enhanced_prompt)
                content = getattr(response, "content", str(response))
                
                # Add a note about tool limitations
                if any(keyword in prompt.lower() for keyword in ['file', 'read', 'analyze', 'code', 'project']):
                    content += "\n\n📝 Note: This model doesn't support tools for file access. For complete file analysis, consider using a tool-compatible model like llama3.1 or qwen2.5-coder."
                
                return content
            except Exception as e:
                stream_text(f"❌ LLM error: {e}")
                return f"Error: Unable to process query - {e}"
        
        return "Agent not available: LangChain not installed or model not accessible"


def create_agent(agent_id: str, config: Dict[str, Any]) -> DeepCoderAgent:
    return DeepCoderAgent(agent_id, config)


__all__ = ["DeepCoderAgent", "create_agent"]
