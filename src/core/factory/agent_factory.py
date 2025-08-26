"""Agent factory for creating and managing agents."""

import os
import sys
from typing import Any, Dict, Optional


def get_agent_instance(agent_name: str, streaming: bool = True, with_tools: bool = True):
    """
    Get the appropriate agent instance based on the agent name.

    Args:
        agent_name: The name of the agent to create
        streaming: Whether to enable streaming responses
        with_tools: Whether to enable tools for the agent
    """
    try:
        # Use the new DynamicAgentFactory for creating agents
        from src.core.agent_factory import get_agent_factory

        factory = get_agent_factory(streaming=streaming)
        agent = factory.get_or_create_agent(agent_name)

        if agent:
            print(f"✅ Created agent using DynamicAgentFactory: {agent_name} (streaming: {streaming})")

            # If tools are explicitly disabled, update the agent
            if not with_tools and hasattr(agent, '_tools'):
                agent._tools = []
                print(f"🛑 Tools explicitly disabled for {agent_name}")

            return agent

        # Fallback to legacy system if DynamicAgentFactory fails
        print("⚠️ DynamicAgentFactory failed, falling back to legacy system")

    except Exception as e:
        print(f"⚠️ Error with DynamicAgentFactory: {e}")
        # Continue to legacy fallback

    # Legacy fallback system
    try:
        # Add parent directory to path for imports
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

        from integrations.model_config_reader import ModelConfigReader

        # Load the model configuration to get agent details
        config_reader = ModelConfigReader('src/config/models.yaml')
        model_config = config_reader.get_model(agent_name)

        if not model_config:
            # Try to find by model ID or partial match
            for model in config_reader.get_all_models():
                if (agent_name in model.model_id or
                    model.model_id.startswith(agent_name) or
                    model.model_id == agent_name or  # Exact model ID match
                    agent_name in model.short_name):
                    model_config = model
                    break

        if not model_config:
            raise ValueError(f"No configuration found for agent '{agent_name}'")

        # Convert model config to agent config format
        agent_config = {
            "name": model_config.name,
            "model_id": model_config.model_id,
            "backend_image": model_config.model_id,
            "parameters": model_config.parameters,
            "tools": model_config.tools,
            "system_message": model_config.system_message,
            "supports_coding": model_config.supports_coding
        }

        # Use UniversalAgent for all agent types
        try:
            from src.agents.universal.agent import create_universal_agent
            return create_universal_agent(agent_name, agent_config, model_config.model_id, streaming)
        except ImportError:
            print("⚠️  Warning: UniversalAgent not available, falling back to simple agent")
            # Import SimpleQueryAgent from single_query_mode
            from ..single_query_mode import SimpleQueryAgent
            return SimpleQueryAgent(agent_name, agent_config)

    except Exception as e:
        print(f"⚠️  Warning: Could not load proper agent, using fallback: {e}")
        # Create a basic fallback config with proper model mapping
        return _create_fallback_agent(agent_name, streaming)


def _create_fallback_agent(agent_name: str, streaming: bool = True):
    """Create a fallback agent when normal creation fails."""
    model_mapping = {
        'phi3_mini': 'phi3:mini',
        'deepcoder': 'deepcoder:14b',
        'deepcoder:14b': 'deepcoder:14b',  # Allow direct model ID
        'qwen2.5-coder:7b': 'qwen2.5-coder:7b',
        'qwen2.5:7b-instruct-q4_K_M': 'qwen2.5:7b-instruct-q4_K_M',
        'qwen2.5:3b-instruct-q4_K_M': 'qwen2.5:3b-instruct-q4_K_M',
        'gemma:7b-instruct-q4_K_M': 'gemma:7b-instruct-q4_K_M',
        'codellama:13b-instruct': 'codellama:13b-instruct',
        'mistral:7b-instruct': 'mistral:7b-instruct',
        'assistant': 'qwen2.5:3b-instruct-q4_K_M',  # Use available model
        'coder': 'qwen2.5-coder:7b'
    }

    fallback_config = {
        'name': agent_name.title(),
        'model_id': model_mapping.get(agent_name, agent_name),
        'backend_image': model_mapping.get(agent_name, agent_name),
        'parameters': {'temperature': 0.7, 'num_ctx': 8192},
        'tools': [],
        'system_message': "You are an AI assistant.",
        'supports_coding': True
    }

    # Try UniversalAgent first, then fallback to SimpleQueryAgent
    try:
        from src.agents.universal.agent import create_universal_agent
        return create_universal_agent(agent_name, fallback_config, fallback_config['model_id'], streaming)
    except ImportError:
        from ..single_query_mode import SimpleQueryAgent
        return SimpleQueryAgent(agent_name, fallback_config)
