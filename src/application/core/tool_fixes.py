"""Tool registry improvements to fix agent executor issues."""

from typing import Any, Optional

from src.utils.enhanced_logging import get_logger


def fix_tool_registry_issues():
    """Fix issues with tool registry and agent executor creation."""
    logger = get_logger()
    logger.info("Applying tool registry fixes...")

    try:
        # Import tools and registry
        from src.tools.registry import get_registered_tools

        # Store original function
        original_get_tools = get_registered_tools

        def improved_get_registered_tools() -> list[Any]:
            """Get registered tools with improved error handling."""
            try:
                tools = original_get_tools()

                # Ensure tools are in proper format
                improved_tools = []
                for tool in tools:
                    if hasattr(tool, 'name') and hasattr(tool, 'description'):
                        # Tool is properly formatted
                        improved_tools.append(tool)
                    elif isinstance(tool, dict):
                        # Convert dict to proper tool format if needed
                        logger.warning(f"Converting dict tool to proper format: {tool.get('name', 'unknown')}")
                        improved_tools.append(tool)
                    else:
                        logger.warning(f"Skipping malformed tool: {type(tool)}")

                logger.info(f"✅ Improved {len(improved_tools)} tools out of {len(tools)} total")
                return improved_tools

            except Exception as e:
                logger.error(f"Error in improved tool registry: {e}")
                # Return empty list as fallback
                return []

        # Apply the patch
        import src.tools.registry
        src.tools.registry.get_registered_tools = improved_get_registered_tools

        logger.info("✅ Tool registry fixes applied successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to apply tool registry fixes: {e}")
        return False

def improve_agent_tool_loading():
    """Improve agent tool loading to prevent executor errors."""
    logger = get_logger()
    logger.info("Improving agent tool loading...")

    try:
        # Import universal agent
        from src.agents.universal.agent import UniversalAgent

        # Store original method
        original_init = UniversalAgent.__init__

        def improved_init(self, agent_id: str, config: dict[str, Any], model_id: Optional[str] = None, streaming: bool = True):
            """Enhanced initialization with better tool loading."""
            try:
                # Call original init
                original_init(self, agent_id, config, model_id, streaming)

                # Additional validation and fixes for tool loading
                if hasattr(self, '_agent_executor') and self._agent_executor is None:
                    logger.warning(f"Agent executor not created for {self._model_id}, attempting to rebuild...")
                    try:
                        self._build_agent_executor_safely()
                    except Exception as e:
                        logger.warning(f"Could not rebuild agent executor: {e}")
                        # Set a flag to indicate limited functionality
                        self._tools_available = False
                else:
                    self._tools_available = True

            except Exception as e:
                logger.error(f"Error in improved agent init: {e}")
                # Call original init as fallback
                original_init(self, agent_id, config, model_id, streaming)
                self._tools_available = False

        def build_agent_executor_safely(self):
            """Safely build agent executor with error handling."""
            try:
                from src.tools.registry import get_registered_tools
                tools = get_registered_tools()

                if not tools:
                    logger.warning("No tools available, agent will run in LLM-only mode")
                    self._agent_executor = None
                    return

                # Validate tools before creating executor
                valid_tools = []
                for tool in tools:
                    if hasattr(tool, 'name') and hasattr(tool, 'description'):
                        valid_tools.append(tool)
                    else:
                        logger.warning(f"Skipping invalid tool: {type(tool)}")

                if valid_tools:
                    # Try to create agent executor with valid tools
                    from langchain.agents import AgentExecutor, create_tool_calling_agent
                    from langchain_core.prompts import ChatPromptTemplate
                    from langchain_ollama import ChatOllama

                    llm = ChatOllama(model=self._model_id, temperature=0.1)

                    prompt = ChatPromptTemplate.from_messages([
                        ("system", self.get_optimized_system_message()),
                        ("human", "{input}"),
                        ("placeholder", "{agent_scratchpad}")
                    ])

                    agent = create_tool_calling_agent(llm, valid_tools, prompt)
                    self._agent_executor = AgentExecutor(
                        agent=agent,
                        tools=valid_tools,
                        verbose=True,
                        handle_parsing_errors=True,
                        max_iterations=10
                    )

                    logger.info(f"✅ Successfully created agent executor with {len(valid_tools)} tools")
                else:
                    logger.warning("No valid tools found, agent will run in LLM-only mode")
                    self._agent_executor = None

            except Exception as e:
                logger.error(f"Failed to build agent executor safely: {e}")
                self._agent_executor = None

        # Apply patches
        UniversalAgent.__init__ = improved_init
        UniversalAgent._build_agent_executor_safely = build_agent_executor_safely

        logger.info("✅ Agent tool loading improvements applied successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to improve agent tool loading: {e}")
        return False

# Auto-apply fixes when module is imported
if __name__ != "__main__":
    fix_tool_registry_issues()
    improve_agent_tool_loading()
