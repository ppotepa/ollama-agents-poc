"""Universal Agent Factory - Creates agents dynamically based on model configuration."""
from __future__ import annotations

from typing import Any

from src.agents.base.base_agent import AbstractAgent
from src.utils.animations import stream_text
from src.utils.enhanced_logging import get_logger

try:  # LangChain optional
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import ChatOllama
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False
    # Define dummy classes to avoid import errors
    class ChatOllama:
        pass

    class AgentExecutor:
        pass

    def create_tool_calling_agent(*args, **kwargs):
        return None

    class ChatPromptTemplate:
        @staticmethod
        def from_messages(*args):
            return None

import src.tools  # noqa: F401  # trigger side-effect imports & registration
from src.tools.registry import get_registered_tools  # registry list


class UniversalAgent(AbstractAgent):
    """Universal agent that can dynamically adapt to any model configuration."""

    def __init__(self, agent_id: str, config: dict[str, Any], model_id: str | None = None, streaming: bool = True):
        super().__init__(agent_id, config)
        self._agent_executor = None
        self._model_id = model_id or config.get("model_id", agent_id)
        self._streaming = streaming
        self._agent_type = self._determine_agent_type()
        self.logger = get_logger()

    def _determine_agent_type(self) -> str:
        """Determine the agent type based on model_id and configuration."""
        model_lower = self._model_id.lower()

        # Coding-focused agents
        if any(keyword in model_lower for keyword in ['coder', 'code', 'deepcoder']):
            return 'coding'

        # Instruction-following agents
        if 'instruct' in model_lower:
            return 'instruct'

        # Chat-optimized agents
        if 'chat' in model_lower:
            return 'chat'

        # General purpose
        return 'general'

    def get_optimized_system_message(self) -> str:
        """Get system message optimized for the agent type."""
        base_message = self.config.get("system_message", "")

        # Add model swapping awareness to all agent types
        model_swap_awareness = """

🔄 MODEL SWAPPING CAPABILITY:
You have the ability to request switching to a different AI model if you encounter a task that might be better suited for another model. Use this capability when:
- You lack specific capabilities needed for the task
- The task requires specialized knowledge (e.g., coding, creative writing, analysis)
- You're hesitating or unsure about your ability to complete the task effectively
- The user's request is outside your optimal performance area

Available models you can switch to:
- qwen2.5-coder:7b: Specialized for programming and coding tasks
- deepcoder:14b: Advanced coding assistant with comprehensive tools
- qwen2.5:7b-instruct-q4_K_M: Strong analytical and instruction-following capabilities
- gemma:7b-instruct-q4_K_M: Good for general tasks and creative work
- codellama:13b-instruct: Coding-focused with strong programming abilities
- mistral:7b-instruct: Balanced model for various tasks

To request a model swap, use the request_model_swap tool with:
- reason: Clear explanation of why you need to switch
- preferred_model: Specific model ID (optional)
- task_type: Type of task (coding, analysis, creative, etc.)

Example: request_model_swap("I need coding capabilities to write Python functions", "qwen2.5-coder:7b", "coding")"""

        if self._agent_type == 'coding':
            return f"""{base_message}

You are a specialized coding assistant with expertise in software development, debugging, and code optimization.

CORE CAPABILITIES:
- Write, review, and improve code in multiple programming languages
- Debug errors and provide fixes with explanations
- Analyze codebases and suggest improvements
- Perform file operations and project management
- Execute code and interpret results

CODING PRINCIPLES:
1. Always provide complete, working code solutions
2. Include error handling and edge cases
3. Follow language best practices and conventions
4. Explain complex logic and design decisions
5. Suggest optimizations when appropriate

When given a coding task:
1. Analyze the requirements thoroughly
2. Break down complex problems into smaller steps
3. Implement clean, readable, and efficient solutions
4. Test the solution and handle potential errors
5. Provide clear explanations of the implementation{model_swap_awareness}"""

        elif self._agent_type == 'instruct':
            return f"""{base_message}

You are an instruction-following assistant optimized for clear, structured responses.

RESPONSE GUIDELINES:
1. Follow instructions precisely and completely
2. Provide step-by-step explanations when needed
3. Structure responses with clear headings and bullet points
4. Give practical, actionable advice
5. Ask clarifying questions when instructions are ambiguous

COMMUNICATION STYLE:
- Be direct and concise while remaining helpful
- Use examples to illustrate concepts
- Break down complex topics into digestible parts
- Acknowledge limitations and suggest alternatives when needed{model_swap_awareness}"""

        elif self._agent_type == 'chat':
            return f"""{base_message}

You are a conversational assistant designed for natural, engaging interactions.

CONVERSATION PRINCIPLES:
1. Maintain context throughout the conversation
2. Ask follow-up questions to better understand needs
3. Provide personalized responses based on user preferences
4. Use a friendly, approachable tone
5. Remember key details from earlier in the conversation

RESPONSE STYLE:
- Be conversational and engaging
- Show empathy and understanding
- Offer multiple perspectives when appropriate
- Encourage continued dialogue and exploration{model_swap_awareness}"""

        else:  # general
            return f"""{base_message}

You are a helpful general-purpose AI assistant with broad knowledge and capabilities.

CORE PRINCIPLES:
1. Provide accurate, helpful, and comprehensive responses
2. Adapt your communication style to the user's needs
3. Ask clarifying questions when needed
4. Offer practical solutions and alternatives
5. Be honest about limitations and uncertainties

RESPONSE APPROACH:
- Tailor responses to the specific question or task
- Provide context and background when helpful
- Use clear, accessible language
- Include examples and illustrations when appropriate
- Suggest next steps or related topics when relevant{model_swap_awareness}"""

    def stream(self, prompt: str, on_token):  # type: ignore[override]
        if not self._loaded:
            self.load()
        if not LANGCHAIN_AVAILABLE or self._llm is None:
            return super().stream(prompt, on_token)

        # If streaming is disabled, use non-streaming mode
        if not self._streaming:
            try:
                response = self._llm.invoke(prompt)
                content = getattr(response, "content", None) or str(response)
                if content:
                    on_token(content)  # Send all content at once
                return content
            except Exception as e:
                stream_text(f"❌ Non-streaming error: {e}")
                return ""

        # Use streaming mode with deduplication
        final_tokens: list[str] = []
        seen_tokens = set()
        try:
            for chunk in self._llm.stream(prompt):  # type: ignore[attr-defined]
                # Extract content from LangChain AIMessageChunk
                text = ""
                if hasattr(chunk, "content") and chunk.content:
                    text = str(chunk.content)  # Use content property, not text method

                if text and text.strip() and text not in seen_tokens:
                    seen_tokens.add(text)
                    on_token(text)
                    final_tokens.append(text)
        except Exception as e:
            stream_text(f"❌ Streaming error: {e}")
            return ""
        return "".join(final_tokens)

    def stream_with_context(self, prompt: str, on_token, context: dict[str, Any]):
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
            self.logger.warning("LangChain not available, falling back to basic mode")
            return None

        # Use model_id from configuration or agent initialization
        model_id = self._model_id
        params = self.config.get("parameters", {})

        # Apply optimizations based on agent type
        if self._agent_type == 'coding':
            # Coding agents need more context and precision
            params.setdefault('temperature', 0.1)
            params.setdefault('num_ctx', 8192)
            params.setdefault('repeat_penalty', 1.1)
        elif self._agent_type == 'chat':
            # Chat agents can be more creative
            params.setdefault('temperature', 0.7)
            params.setdefault('num_ctx', 4096)
        else:
            # Default balanced settings
            params.setdefault('temperature', 0.3)
            params.setdefault('num_ctx', 4096)

        try:
            llm = ChatOllama(
                model=model_id,
                streaming=self._streaming,
                **{k: v for k, v in params.items() if isinstance(k, str)}
            )
            self.logger.info(f"Initialized {self._agent_type} agent with model {model_id} (streaming: {self._streaming})")
            return llm
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM for {model_id}: {e}")
            return None

    def _build_tools(self) -> list[Any]:  # noqa: D401
        desired = set(self.config.get("tools", []))
        selected: list[Any] = []
        available_tools = get_registered_tools()

        for t in available_tools:
            name = getattr(t, "name", None)
            if name and name in desired:
                selected.append(t)

        # If no tools specified, provide default tools based on agent type
        if not desired and not selected and self._agent_type == 'coding':
            # Coding agents get file and execution tools by default
            coding_tools = ['read_file', 'write_file', 'list_files', 'run_python', 'analyze_repository']
            for t in available_tools:
                name = getattr(t, "name", None)
                if name and name in coding_tools:
                    selected.append(t)

        # If no specific tools requested, use all available tools
        if desired and not selected and available_tools:
            self.logger.info(f"No specific tools matched, using all {len(available_tools)} available tools")
            selected = available_tools

        if selected:
            tool_names = [getattr(t, 'name', str(t)) for t in selected]
            print(f"🔧 Loaded {len(selected)} tools: {tool_names}", flush=True)
            self.logger.info(f"Agent {self._model_id} loaded tools: {tool_names}")
        else:
            # Only show the warning if we explicitly tried to find tools but couldn't
            if desired or available_tools:
                print("⚠️ No matching tools found for this agent configuration", flush=True)
            print("� No tools configured - running in LLM-only mode (tools functionality is limited)", flush=True)

        return selected

    def _build_agent_executor(self):
        """Build an agent executor that can use tools."""
        if not LANGCHAIN_AVAILABLE or not self._llm or not self._tools:
            return None

        try:
            # Debug the tools to ensure they're properly structured
            for i, tool in enumerate(self._tools):
                tool_name = getattr(tool, "name", f"Tool #{i}")
                tool_type = type(tool).__name__
                has_get = hasattr(tool, "get")
                self.logger.info(f"Tool {i}: {tool_name} ({tool_type}), has get method: {has_get}")

            # Use optimized system message
            system_message = self.get_optimized_system_message()

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}")
            ])

            # Create proper tool objects with dict-like interface
            processed_tools = []

            # First make sure we have proper tools with .get() method
            class ToolAdapter:
                """Adapter to make any tool compatible with LangChain's expectations."""
                def __init__(self, tool):
                    self.tool = tool
                    self.name = getattr(tool, "name", str(tool))
                    self.description = getattr(tool, "description", "")
                    self.args_schema = getattr(tool, "args_schema", None)
                    self.return_direct = getattr(tool, "return_direct", False)
                    self.coroutine = getattr(tool, "coroutine", None)
                    self.__name__ = getattr(tool, "__name__", self.name)

                    # Copy other attributes
                    for attr in dir(tool):
                        if not attr.startswith('_') and not hasattr(self, attr):
                            setattr(self, attr, getattr(tool, attr))

                def __call__(self, *args, **kwargs):
                    return self.tool(*args, **kwargs)

                def get(self, key, default=None):
                    """Implement dict-like .get() method."""
                    return getattr(self, key, default)

                # Implement dictionary-like interface
                def __getitem__(self, key):
                    value = getattr(self, key, None)
                    if value is None:
                        raise KeyError(key)
                    return value

            # Process each tool
            for tool in self._tools:
                # Skip malformed tools
                if not hasattr(tool, "name") or not callable(tool):
                    self.logger.warning(f"Skipping malformed tool: {tool}")
                    continue

                # Wrap the tool to ensure it has .get method
                wrapped_tool = ToolAdapter(tool)
                processed_tools.append(wrapped_tool)
                self.logger.info(f"Tool {tool.name} adapted with dict-like interface")

            if not processed_tools:
                self.logger.error("No valid tools available after processing")
                return None

            # Verify each tool has the required methods
            for tool in processed_tools:
                if not hasattr(tool, "get") or not callable(tool.get):
                    self.logger.error(f"Tool {tool.name} missing get() method after adaptation")
                    # Fix it on the fly
                    tool.get = lambda key, default=None: getattr(tool, key, default)

            # Convert list to a dict to ensure LangChain can use .get()
            try:
                # Create a more robust tool adapter
                class EnhancedToolAdapter:
                    """Enhanced adapter for tools to work with LangChain."""
                    def __init__(self, tool):
                        self._tool = tool
                        self.name = getattr(tool, "name", str(tool))
                        self.description = getattr(tool, "description", "")
                        self.args_schema = getattr(tool, "args_schema", None)
                        self.return_direct = getattr(tool, "return_direct", False)
                        self.__name__ = self.name

                        # Ensure tool has all required methods/attributes
                        self._ensure_attributes()

                    def _ensure_attributes(self):
                        """Ensure the tool has all required attributes."""
                        # Add any required attributes LangChain might expect
                        required_attrs = [
                            "name", "description", "args_schema",
                            "return_direct", "__name__"
                        ]
                        for attr in required_attrs:
                            if not hasattr(self, attr):
                                setattr(self, attr, None)

                    def __call__(self, *args, **kwargs):
                        """Execute the tool."""
                        return self._tool(*args, **kwargs)

                    def get(self, key, default=None):
                        """Get an attribute with dict-like interface."""
                        return getattr(self, key, default)

                    def __getitem__(self, key):
                        """Support dict-like access."""
                        if hasattr(self, key):
                            return getattr(self, key)
                        raise KeyError(key)

                    def __str__(self):
                        """String representation."""
                        return f"Tool({self.name})"

                    def __repr__(self):
                        """Detailed representation."""
                        return f"Tool(name='{self.name}', description='{self.description}')"

                # Create enhanced tool adapters
                enhanced_tools = []
                for tool in processed_tools:
                    try:
                        enhanced_tool = EnhancedToolAdapter(tool)
                        enhanced_tools.append(enhanced_tool)
                        self.logger.info(f"Created enhanced adapter for tool {tool.name}")
                    except Exception as e:
                        self.logger.error(f"Failed to create enhanced adapter for {tool.name}: {e}")
                        # Fall back to the original tool
                        enhanced_tools.append(tool)

                # Create a tool dictionary for any internal lookups
                tool_dict = {tool.name: tool for tool in enhanced_tools}

                # Make the list itself have a .get method to handle LangChain's expectations
                class ToolList(list):
                    def get(self, key, default=None):
                        if key in tool_dict:
                            return tool_dict[key]
                        return default

                # Create a list with .get method
                final_tools = ToolList(enhanced_tools)

                # Create the agent with our enhanced tools
                agent = create_tool_calling_agent(self._llm, final_tools, prompt_template)
                self._agent_executor = AgentExecutor(
                    agent=agent,
                    tools=final_tools,
                    verbose=False,
                    max_iterations=3,
                    handle_parsing_errors=True
                )
            except Exception as e:
                error_str = str(e)
                if "'list' object has no attribute 'get'" in error_str:
                    self.logger.error("Tool format error: A list is being treated as a dictionary, fixing tool structure")
                    # Create a more detailed log of the tool structure
                    for i, tool in enumerate(processed_tools):
                        self.logger.info(f"Tool {i} details: {dir(tool)}")

                    # Last-resort fix: monkey patch the list class just for this instance
                    def get_method(self, key, default=None):
                        for item in self:
                            if hasattr(item, 'name') and item.name == key:
                                return item
                        return default

                    # Try again with our emergency fix
                    self.logger.info("Attempting emergency fix for tool list")
                    final_tools = processed_tools.copy()
                    final_tools.get = get_method.__get__(final_tools)

                    # Last attempt with emergency fix
                    agent = create_tool_calling_agent(self._llm, final_tools, prompt_template)
                    self._agent_executor = AgentExecutor(
                        agent=agent,
                        tools=final_tools,
                        verbose=False,
                        max_iterations=3,
                        handle_parsing_errors=True
                    )
                elif "'ToolAdapter' object has no attribute" in error_str or "is not a module, class, method, or function" in error_str:
                    self.logger.error(f"Tool adaptation issue: {error_str}")

                    # If we're having trouble with tool adaptation, try a completely different approach:
                    # Use the raw StructuredTool objects directly

                    # Get the original tools from the tool registry
                    from src.tools.registry import get_registered_tools
                    registry_tools = get_registered_tools()

                    # Filter to match the names we want
                    desired_names = [t.name for t in self._tools]
                    raw_tools = [t for t in registry_tools if hasattr(t, 'name') and t.name in desired_names]

                    if raw_tools:
                        self.logger.info(f"Using {len(raw_tools)} raw tools from registry")

                        # Create agent with unmodified tools
                        try:
                            agent = create_tool_calling_agent(self._llm, raw_tools, prompt_template)
                            self._agent_executor = AgentExecutor(
                                agent=agent,
                                tools=raw_tools,
                                verbose=False,
                                max_iterations=3,
                                handle_parsing_errors=True
                            )
                            self.logger.info("Created agent executor with raw tools")
                        except Exception as inner_e:
                            self.logger.error(f"Failed with raw tools too: {inner_e}")
                            # Even raw tools failed, operate in LLM-only mode
                            self._agent_executor = None
                    else:
                        self.logger.error("Could not find matching tools in registry")
                        self._agent_executor = None
                        print("🛑 Failed to initialize tools - running in LLM-only mode", flush=True)
                else:
                    self.logger.error(f"Unexpected error building agent executor: {e}")
                    # Don't raise - we'll operate in LLM-only mode
                    self._agent_executor = None
                    print("🛑 Failed to initialize tools - running in LLM-only mode", flush=True)
                    # Return None without raising an exception

            if hasattr(self, 'capabilities') and self.capabilities.get('verbose'):
                stream_text(f"✅ {self._agent_type.title()} agent executor created with {len(self._tools)} tools")

            self.logger.info(f"Built agent executor for {self._model_id} ({self._agent_type} type)")
            return self._agent_executor
        except Exception as e:
            stream_text(f"❌ Failed to build agent executor: {e}")
            self.logger.error(f"Failed to build agent executor for {self._model_id}: {e}")
            return None

    def load(self) -> None:
        """Load the agent and create agent executor with tools."""
        super().load()
        if LANGCHAIN_AVAILABLE and self._llm and self._tools:
            self._build_agent_executor()

    def run(self, prompt: str) -> str:
        if not self._loaded:
            self.load()

        # If tools failed to load but we have LLM, let user know tools are not available
        if not self._agent_executor and self._llm and self._tools:
            # Log the issue
            self.logger.warning("Agent executor not available despite tools being configured")

            # Notify user about the fallback mode
            tool_names = [getattr(t, 'name', str(t)) for t in self._tools]
            message = (
                f"\n🛑 IMPORTANT: Tools could not be properly initialized. "
                f"Running in LLM-only mode instead of using {len(self._tools)} tools ({', '.join(tool_names)}).\n\n"
                f"If you want to use a tool, you can still try entering commands like:\n"
                f"- write_file path=\"D:\\containers\\ollama\\hello.txt\" content=\"Hello, Ollama!\"\n"
                f"- read_file path=\"D:\\containers\\ollama\\hello.txt\"\n"
                f"- list_files path=\"D:\\containers\\ollama\"\n\n"
            )

            # Include this notice in the prompt so the LLM knows tools aren't available
            prompt = message + prompt

        # If we have a proper agent executor, use it
        if self._agent_executor:
            try:
                result = self._agent_executor.invoke({"input": prompt})
                return result.get("output", "No response generated")
            except Exception as e:
                stream_text(f"❌ Agent executor error: {e}")
                self.logger.error(f"Agent executor error for {self._model_id}: {e}")
                # Fallback to direct LLM

        # Fallback: direct LLM call
        if LANGCHAIN_AVAILABLE and self._llm:
            try:
                response = self._llm.invoke(prompt)
                return getattr(response, "content", str(response))
            except Exception as e:
                stream_text(f"❌ LLM error: {e}")
                self.logger.error(f"LLM error for {self._model_id}: {e}")

        return f"Agent {self._model_id} not available or failed to respond"

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model."""
        return {
            'model_id': self._model_id,
            'agent_type': self._agent_type,
            'config': self.config,
            'has_tools': len(self._tools or []) > 0,
            'has_agent_executor': self._agent_executor is not None
        }


def create_universal_agent(agent_id: str, config: dict[str, Any], model_id: str | None = None, streaming: bool = True) -> UniversalAgent:
    """Create a universal agent that can adapt to any model configuration."""
    return UniversalAgent(agent_id, config, model_id, streaming)


__all__ = ["UniversalAgent", "create_universal_agent"]
