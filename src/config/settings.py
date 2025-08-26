"""
Configuration management for DeepCoder
"""

from pathlib import Path
from typing import Any

# Try to import yaml, fallback to JSON parsing if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("⚠️ PyYAML not installed. Install with: pip install pyyaml")


class ConfigManager:
    """Manages configuration loading and agent definitions.

    Backwards compatible: if only models.yaml exists it is mapped into agents.
    """

    def __init__(self, config_dir: str = None):
        if config_dir is None:
            # Default to config directory relative to this file
            self.config_dir = Path(__file__).parent
        else:
            self.config_dir = Path(config_dir)

        self.agents_config = None
        self._load_agents_config()

    def _load_agents_config(self):
        """Load agents configuration from YAML (or migrate from models)."""
        agents_file = self.config_dir / "agents.yaml"
        models_file = self.config_dir / "models.yaml"  # legacy
        try:
            target_file = agents_file if agents_file.exists() else models_file
            with open(target_file, encoding='utf-8') as f:
                raw = yaml.safe_load(f) if YAML_AVAILABLE else self._get_default_agents()
            if raw and 'models' in raw and 'agents' not in raw:
                raw = {'agents': raw['models']}
            self.agents_config = raw
        except FileNotFoundError:
            print(f"⚠️ Agents configuration file not found: {agents_file}")
            self.agents_config = self._get_default_agents()
        except Exception as e:
            print(f"⚠️ Error parsing agents configuration: {e}")
            self.agents_config = self._get_default_agents()

    def _get_default_agents(self):
        return {
            "agents": {
                "deepcoder": {
                    "name": "DeepCoder Basic",
                    "backend_image": "deepcoder:14b",
                    "description": "Advanced AI coding assistant",
                    "capabilities": {"coding": True, "streaming": True},
                    "tools": []
                }
            }
        }

    def get_available_agents(self) -> dict[str, dict[str, Any]]:
        return self.agents_config.get("agents", {})

    def get_agent_config(self, agent_name: str) -> dict[str, Any]:
        return self.get_available_agents().get(agent_name, {})

    def list_agent_names(self) -> list[str]:
        return list(self.get_available_agents().keys())

    # Compatibility wrappers
    def get_available_models(self):  # pragma: no cover
        return self.get_available_agents()

    def get_model_config(self, model_name: str):  # pragma: no cover
        return self.get_agent_config(model_name)

    def is_valid_model(self, model_name: str):  # pragma: no cover
        return self.is_valid_agent(model_name)

    def is_valid_agent(self, agent_name: str) -> bool:
        return agent_name in self.get_available_agents()


# Global config manager instance
config_manager = ConfigManager()
