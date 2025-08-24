"""DeepCoderAgent - concrete implementation using Ollama backend."""
from __future__ import annotations

from typing import Any, Dict, List
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
            return self.stream(prompt, on_token)s.base.base_agent import AbstractAgent
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
            return None
        
        try:
            # Create a prompt template that includes tool usage instructions
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_message + """

You have access to the following tools. Use them when needed to answer questions about repository structure, file contents, or analysis:

{tools}

When you need to analyze a repository, always use the repository context tools like:
- analyze_repository_context: Get overall repository structure and statistics
- analyze_repo_languages: Get programming language breakdown  
- analyze_repo_directories: Get directory structure analysis
- get_repository_state: Get current repository state
- get_file_content: Read specific files
- search_files: Search for files by name, extension, or content

Always use these tools before asking the user for information that might be available in the repository."""),
                ("user", "{input}"),
                ("assistant", "{agent_scratchpad}")
            ])

            # Create tool-calling agent
            agent = create_tool_calling_agent(self._llm, self._tools, prompt)
            
            # Create agent executor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self._tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=10
            )
            
            return agent_executor
            
        except Exception as e:
            stream_text(f"⚠️ Could not create agent executor: {e}")
            return None

    def load(self):
        """Load the agent and create agent executor with tools."""
        super().load()
        
        # Create agent executor if tools are available
        if self._tools and LANGCHAIN_AVAILABLE:
            self._agent_executor = self._build_agent_executor()
            if self._agent_executor:
                stream_text(f"✅ Agent executor created with {len(self._tools)} tools")

    def run(self, prompt: str) -> str:
        """Run the agent, using tool-calling agent executor if available."""
        if not self._loaded:
            self.load()
        
        # Use agent executor if available (supports tools)
        if self._agent_executor:
            try:
                result = self._agent_executor.invoke({"input": prompt})
                return str(result.get("output", "No output received"))
            except Exception as e:
                stream_text(f"⚠️ Agent executor error: {e}")
                # Fall back to direct LLM call
        
        # Fallback to direct LLM call (no tools)
        if self._llm is None:
            return "⚠️ LLM backend not available. Install dependencies."
        
        try:
            if hasattr(self._llm, "invoke"):
                result = self._llm.invoke(prompt)
                if isinstance(result, dict) and "content" in result:
                    return str(result["content"])
                return str(result)
            return str(self._llm(prompt))
        except Exception as e:
            return f"❌ Error running agent: {e}"

    @property
    def system_message(self) -> str:
        return self.config.get("system_message", "You are an AI coding assistant.")

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


def create_agent(agent_id: str, config: Dict[str, Any]) -> DeepCoderAgent:
    return DeepCoderAgent(agent_id, config)


__all__ = ["DeepCoderAgent", "create_agent"]
