"""Enhanced collaborative system fixes for context preservation and model selection."""

from typing import Any

from src.utils.enhanced_logging import get_logger


def apply_collaborative_system_fixes():
    """Apply all fixes to the collaborative system."""
    logger = get_logger()
    logger.info("Applying collaborative system fixes...")

    # Import and patch the collaborative system
    try:
        from src.core.collaborative_system import CollaborativeAgentSystem

        # Store original methods
        original_collaborative_execution = CollaborativeAgentSystem.collaborative_execution

        # Define improved collaborative execution with better error handling
        def improved_collaborative_execution(self, query: str, working_directory: str = ".", max_steps: int = None) -> dict[str, Any]:
            """Enhanced collaborative execution with better model handling."""
            logger = get_logger()

            try:
                # Check if the current agent model exists
                current_model = getattr(self.main_agent, '_model_id', 'unknown')

                # If current model is "universal" or doesn't exist, switch to a reliable model first
                if current_model == "universal" or not self._verify_model_exists(current_model):
                    logger.warning(f"Model '{current_model}' not available, switching to reliable fallback")

                    # Try reliable models in order of preference
                    reliable_models = [
                        "qwen2.5-coder:7b",
                        "qwen2.5:7b-instruct-q4_K_M",
                        "phi3:small",
                        "llama3:8b"
                    ]

                    switched = False
                    for model in reliable_models:
                        if self._verify_model_exists(model):
                            logger.info(f"Switching to reliable model: {model}")
                            if self._switch_main_agent(model):
                                switched = True
                                break

                    if not switched:
                        logger.error("No reliable models available, execution may fail")

                # Call original execution with current agent
                return original_collaborative_execution(self, query, working_directory, max_steps)

            except Exception as e:
                logger.error(f"Error in enhanced collaborative execution: {e}")
                # Fallback to original method
                return original_collaborative_execution(self, query, working_directory, max_steps)

        def verify_model_exists(self, model_name: str) -> bool:
            """Verify if a model exists in Ollama."""
            try:
                from src.core.model_discovery import model_exists
                return model_exists(model_name)
            except Exception:
                # If we can't verify, assume it exists to avoid blocking execution
                return True

        # Apply patches
        CollaborativeAgentSystem.collaborative_execution = improved_collaborative_execution
        CollaborativeAgentSystem._verify_model_exists = verify_model_exists

        logger.info("✅ Collaborative system fixes applied successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to apply collaborative system fixes: {e}")
        return False

def improve_single_query_mode():
    """Improve the single query mode to handle model failures better."""
    logger = get_logger()
    logger.info("Improving single query mode...")

    try:
        from src.core import single_query_mode

        # Store original function
        original_run_collaborative = single_query_mode.run_collaborative_query

        def improved_run_collaborative_query(query: str, agent_name: str, max_iterations: int = 5, streaming: bool = True) -> str:
            """Enhanced collaborative query execution with better error handling."""

            # Handle "universal" as special multi-model agent
            if agent_name == "universal":
                logger.info("Creating Universal Multi-Agent for dynamic model switching")
                try:
                    # For collaborative query, we need to return a model name that maps to our universal agent
                    # We'll handle this in the agent factory instead
                    agent_name = "universal-multi"  # Special identifier for universal multi-agent
                except Exception as e:
                    logger.warning(f"Failed to import Universal Multi-Agent: {e}, using fallback")
                    agent_name = "qwen2.5-coder:7b"  # Fallback to reliable model
            else:
                # Map other problematic agent names to reliable ones
                agent_mapping = {
                    "default": "qwen2.5:7b-instruct-q4_K_M",
                    "unknown": "qwen2.5-coder:7b"
                }

                # Use mapped agent name if available
                if agent_name in agent_mapping:
                    logger.info(f"Mapping agent '{agent_name}' to '{agent_mapping[agent_name]}'")
                    agent_name = agent_mapping[agent_name]

            # Verify agent exists before proceeding
            try:
                from src.core.model_discovery import model_exists
                if not model_exists(agent_name):
                    logger.warning(f"Agent '{agent_name}' not found, using fallback")
                    agent_name = "qwen2.5-coder:7b"  # Reliable fallback
            except Exception:
                pass  # If we can't verify, proceed anyway

            # Call original function with improved agent name
            return original_run_collaborative(query, agent_name, max_iterations, streaming)

        # Apply patch
        single_query_mode.run_collaborative_query = improved_run_collaborative_query

        logger.info("✅ Single query mode improvements applied successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to improve single query mode: {e}")
        return False

# Auto-apply fixes when module is imported
if __name__ != "__main__":
    apply_collaborative_system_fixes()
    improve_single_query_mode()
