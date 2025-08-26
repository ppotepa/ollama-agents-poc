"""Dynamic Agent Factory - Creates and manages agents dynamically."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.agents.base.base_agent import AbstractAgent
from src.agents.universal.agent import create_universal_agent
from src.utils.enhanced_logging import get_logger


class DynamicAgentFactory:
    """Factory for creating agents dynamically based on model configurations."""

    def __init__(self, models_config_path: str | None = None, streaming: bool = True):
        """Initialize the dynamic agent factory.

        Args:
            models_config_path: Path to models.yaml configuration file
            streaming: Whether agents should use streaming mode by default
        """
        self.logger = get_logger()
        self.models_config = self._load_models_config(models_config_path)
        self._streaming = streaming
        self._agent_cache: dict[str, AbstractAgent] = {}

    def _load_models_config(self, config_path: str | None = None) -> dict[str, Any]:
        """Load models configuration from YAML file."""
        if not config_path:
            # Default path relative to this file
            config_path = Path(__file__).parent.parent / "config" / "models.yaml"

        try:
            with open(config_path, encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self.logger.info(f"Loaded models configuration from {config_path}")
                return config.get('models', {})
        except Exception as e:
            self.logger.error(f"Failed to load models config from {config_path}: {e}")
            return {}

    def _find_model_config(self, model_id: str) -> tuple[str | None, dict[str, Any] | None]:
        """Find model configuration by model_id.

        Args:
            model_id: The model identifier to search for

        Returns:
            Tuple of (config_key, config_dict) or (None, None) if not found
        """
        # Direct key match
        if model_id in self.models_config:
            return model_id, self.models_config[model_id]

        # Search by model_id field
        for config_key, config in self.models_config.items():
            if config.get('model_id') == model_id:
                return config_key, config

        # Partial match (e.g., "qwen2.5-coder:7b" matches "qwen2.5-coder")
        for config_key, config in self.models_config.items():
            config_model_id = config.get('model_id', config_key)
            if model_id.startswith(config_model_id) or config_model_id.startswith(model_id):
                self.logger.info(f"Partial match: {model_id} -> {config_model_id}")
                return config_key, config

        return None, None

    def create_agent(self, model_id: str, force_recreate: bool = False) -> AbstractAgent | None:
        """Create an agent for the specified model.

        Args:
            model_id: The model identifier
            force_recreate: If True, recreate even if cached

        Returns:
            The created agent or None if creation failed
        """
        # Check cache first
        if not force_recreate and model_id in self._agent_cache:
            self.logger.debug(f"Returning cached agent for {model_id}")
            return self._agent_cache[model_id]

        # Handle special Universal Multi-Agent
        if model_id in ["universal", "universal-multi"]:
            try:
                from src.core.universal_multi_agent import create_universal_multi_agent
                agent = create_universal_multi_agent()
                self._agent_cache[model_id] = agent
                self.logger.info(f"Created Universal Multi-Agent for {model_id}")
                return agent
            except Exception as e:
                self.logger.error(f"Failed to create Universal Multi-Agent: {e}")
                # Fallback to regular agent with reliable model
                model_id = "qwen2.5-coder:7b"

        # Find model configuration
        config_key, config = self._find_model_config(model_id)

        if not config:
            self.logger.warning(f"No configuration found for model {model_id}")
            # Create minimal config for unknown models
            config = {
                'model_id': model_id,
                'name': f"Unknown Model ({model_id})",
                'provider': 'ollama',
                'capabilities': {
                    'general_qa': True,
                    'streaming': True
                },
                'parameters': {
                    'temperature': 0.3,
                    'num_ctx': 4096
                },
                'tools': []
            }
            config_key = model_id

        try:
            # Create universal agent with streaming configuration
            agent = create_universal_agent(config_key, config, model_id, self._streaming)

            # Cache the agent
            self._agent_cache[model_id] = agent

            self.logger.info(f"Created agent for model {model_id} (streaming: {self._streaming})")
            return agent

        except Exception as e:
            self.logger.error(f"Failed to create agent for {model_id}: {e}")
            return None

    def get_or_create_agent(self, model_id: str) -> AbstractAgent | None:
        """Get existing agent or create new one."""
        return self.create_agent(model_id, force_recreate=False)

    def switch_agent(self, current_agent: AbstractAgent | None, new_model_id: str) -> AbstractAgent | None:
        """Switch from current agent to a new model.

        Args:
            current_agent: The current agent (can be None)
            new_model_id: The new model to switch to

        Returns:
            The new agent or None if switch failed
        """
        # Log the switch
        current_model = getattr(current_agent, '_model_id', 'None') if current_agent else 'None'
        self.logger.info(f"Switching agent: {current_model} -> {new_model_id}")

        # Create new agent
        new_agent = self.create_agent(new_model_id)

        if new_agent:
            self.logger.info(f"Successfully switched to agent {new_model_id}")
        else:
            self.logger.error(f"Failed to switch to agent {new_model_id}")

        return new_agent

    def get_agent_info(self, model_id: str) -> dict[str, Any] | None:
        """Get information about an agent without creating it."""
        config_key, config = self._find_model_config(model_id)
        if config:
            return {
                'model_id': model_id,
                'config_key': config_key,
                'name': config.get('name', model_id),
                'provider': config.get('provider', 'unknown'),
                'capabilities': config.get('capabilities', {}),
                'tools': config.get('tools', []),
                'cached': model_id in self._agent_cache
            }
        return None

    def clear_cache(self) -> None:
        """Clear the agent cache."""
        self.logger.info(f"Clearing agent cache ({len(self._agent_cache)} agents)")
        self._agent_cache.clear()

    def get_cached_agents(self) -> dict[str, AbstractAgent]:
        """Get all cached agents."""
        return self._agent_cache.copy()


# Global factory instance
_global_factory: DynamicAgentFactory | None = None


def get_agent_factory(streaming: bool = True) -> DynamicAgentFactory:
    """Get the global agent factory instance."""
    global _global_factory
    if _global_factory is None:
        _global_factory = DynamicAgentFactory(streaming=streaming)
    return _global_factory


def create_agent_for_model(model_id: str) -> AbstractAgent | None:
    """Convenience function to create an agent for a model."""
    factory = get_agent_factory()
    return factory.create_agent(model_id)


__all__ = ["DynamicAgentFactory", "get_agent_factory", "create_agent_for_model"]
