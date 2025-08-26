"""Model configuration management and loading."""

from pathlib import Path
from typing import Any, Dict

import yaml

from src.utils.enhanced_logging import get_logger

try:
    from src.core.model_config_updater import get_available_models, update_models_config, validate_models_config
    from src.core.model_discovery import OllamaModelDiscovery
    from src.core.model_registry import get_model_registry
except ImportError:
    # Fallback if modules are not available
    get_available_models = None
    validate_models_config = None
    update_models_config = None
    OllamaModelDiscovery = None
    get_model_registry = None


class ModelConfigManager:
    """Manages model configuration loading and updating."""

    def __init__(self, auto_update: bool = True):
        """Initialize the config manager.
        
        Args:
            auto_update: Whether to automatically update configuration
        """
        self.logger = get_logger()
        self.auto_update = auto_update
        self.model_discovery = OllamaModelDiscovery() if OllamaModelDiscovery else None
        self.model_registry = get_model_registry() if get_model_registry else None

    def load_models_config(self) -> Dict[str, Any]:
        """Load models configuration with auto-update capability.
        
        Returns:
            Models configuration dictionary
        """
        config_path = Path("src/config/models.yaml")

        # Check if configuration exists and is valid
        if self.auto_update and update_models_config and (not config_path.exists() or not validate_models_config()):
            self.logger.info("Updating models configuration...")
            try:
                update_models_config()
                self.logger.info("Models configuration updated successfully")
            except Exception as e:
                self.logger.error(f"Failed to update models config: {e}")

        # Load configuration from file
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                self.logger.error(f"Failed to load models config from {config_path}: {e}")

        # Fallback to dynamic configuration
        return self.get_dynamic_config()

    def get_dynamic_config(self) -> Dict[str, Any]:
        """Generate dynamic configuration when static config is unavailable.
        
        Returns:
            Dynamically generated configuration
        """
        config = {"models": []}

        try:
            # Use model registry if available
            if self.model_registry:
                models = self.model_registry.get_all_models()
                for model in models:
                    config["models"].append({
                        "name": model.name,
                        "supports_tools": model.supports_tools,
                        "capabilities": model.capabilities,
                        "size_b": model.size_b
                    })
            else:
                # Fallback configuration with common models
                config = self._get_fallback_config()

        except Exception as e:
            self.logger.error(f"Failed to generate dynamic config: {e}")
            config = self._get_fallback_config()

        return config

    def _get_fallback_config(self) -> Dict[str, Any]:
        """Get minimal fallback configuration.
        
        Returns:
            Fallback configuration dictionary
        """
        return {
            "models": [
                {
                    "name": "qwen2.5-coder:7b-instruct",
                    "supports_tools": True,
                    "capabilities": ["coding", "programming", "general", "instruction_following"],
                    "size_b": 7.0
                },
                {
                    "name": "qwen2.5:7b",
                    "supports_tools": True,
                    "capabilities": ["general", "analysis", "conversation"],
                    "size_b": 7.0
                },
                {
                    "name": "llama3.2:3b",
                    "supports_tools": True,
                    "capabilities": ["general", "conversation", "instruction_following"],
                    "size_b": 3.0
                },
                {
                    "name": "llama3.2:1b",
                    "supports_tools": False,
                    "capabilities": ["general", "conversation"],
                    "size_b": 1.0
                },
                {
                    "name": "phi3:mini",
                    "supports_tools": False,
                    "capabilities": ["general", "conversation"],
                    "size_b": 3.8
                }
            ]
        }

    def refresh_configuration(self) -> Dict[str, Any]:
        """Force refresh of the models configuration.
        
        Returns:
            Refreshed configuration
        """
        self.logger.info("Refreshing models configuration...")
        
        # Force update if auto-update is enabled
        if self.auto_update and update_models_config:
            try:
                update_models_config()
                self.logger.info("Configuration update completed")
            except Exception as e:
                self.logger.error(f"Configuration update failed: {e}")
        
        # Reload configuration
        return self.load_models_config()

    def ensure_model_available(self, model_name: str) -> bool:
        """Ensure a model is available, pulling it if necessary.
        
        Args:
            model_name: Name of the model to ensure availability
            
        Returns:
            True if model is available, False otherwise
        """
        try:
            # Check if model is already available
            if get_available_models:
                available = get_available_models()
                if model_name in available:
                    return True

            # Try to pull the model if discovery is available
            if self.model_discovery:
                self.logger.info(f"Attempting to pull model: {model_name}")
                success = self.model_discovery.pull_model(model_name)
                if success:
                    self.logger.info(f"Successfully pulled model: {model_name}")
                    return True
                else:
                    self.logger.warning(f"Failed to pull model: {model_name}")

            return False

        except Exception as e:
            self.logger.error(f"Error ensuring model availability for {model_name}: {e}")
            return False

    def validate_config(self) -> bool:
        """Validate the current models configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            config = self.load_models_config()
            
            # Basic validation
            if not isinstance(config, dict):
                return False
            
            models = config.get("models", [])
            if not isinstance(models, list):
                return False
            
            # Validate each model entry
            for model in models:
                if not isinstance(model, dict):
                    return False
                
                required_fields = ["name"]
                for field in required_fields:
                    if field not in model:
                        return False
            
            return True

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
