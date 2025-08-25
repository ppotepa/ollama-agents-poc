"""Model capability checking and validation."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from src.utils.enhanced_logging import get_logger

try:
    from src.core.model_config_updater import get_available_models, validate_models_config, update_models_config
    from src.core.model_discovery import OllamaModelDiscovery
except ImportError:
    # Fallback if modules are not available
    get_available_models = None
    validate_models_config = None
    update_models_config = None
    OllamaModelDiscovery = None


class ModelCapabilityChecker:
    """Validates model capabilities and provides intelligent model selection."""
    
    def __init__(self, auto_update: bool = True, auto_pull: bool = False):
        """Initialize the capability checker.
        
        Args:
            auto_update: Whether to automatically update model configuration
            auto_pull: Whether to automatically pull missing models
        """
        self.logger = get_logger()
        self.auto_pull = auto_pull
        self.model_discovery = OllamaModelDiscovery() if OllamaModelDiscovery else None
        self.models_config = self._load_models_config(auto_update)
        
        # Cache for capability lookups
        self._tool_support_cache = {}
        self._capability_cache = {}
        
    def _load_models_config(self, auto_update: bool = True) -> Dict[str, Any]:
        """Load models configuration with auto-update capability."""
        config_path = Path("src/config/models.yaml")
        
        # Check if configuration exists and is valid
        if auto_update and update_models_config and (not config_path.exists() or not validate_models_config()):
            self.logger.info("Updating model configuration from Ollama instance...")
            if update_models_config(force_refresh=True):
                self.logger.info("Model configuration updated successfully")
            else:
                self.logger.warning("Failed to update model configuration, using fallback")
        
        # Load configuration
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                self.logger.debug(f"Loaded models config from {config_path}")
                return config.get('models', {})
            else:
                self.logger.warning("No models configuration found, using dynamic discovery")
                return self._get_dynamic_config()
                
        except Exception as e:
            self.logger.error(f"Failed to load models configuration: {e}")
            return self._get_dynamic_config()
    
    def _get_dynamic_config(self) -> Dict[str, Any]:
        """Get dynamic configuration from model discovery."""
        try:
            if get_available_models:
                available_models = get_available_models()
                config = {}
                
                for model in available_models:
                    config[model["model_id"]] = {
                        "supports_tools": model["supports_tools"],
                        "capabilities": model["capabilities"],
                        "size_b": model["size_b"],
                        "type": model["type"]
                    }
                
                return config
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"Failed to get dynamic configuration: {e}")
            return {}
    
    def supports_tools(self, model_name: str) -> bool:
        """Check if a model supports tools.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if the model supports tools, False otherwise
        """
        if model_name in self._tool_support_cache:
            return self._tool_support_cache[model_name]
        
        # Check configuration
        model_config = self.models_config.get(model_name, {})
        supports_tools = model_config.get('supports_tools', False)
        
        # Cache the result
        self._tool_support_cache[model_name] = supports_tools
        
        self.logger.debug(f"Model {model_name} supports tools: {supports_tools}")
        return supports_tools
    
    def get_model_capabilities(self, model_name: str) -> List[str]:
        """Get the capabilities of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of capabilities
        """
        if model_name in self._capability_cache:
            return self._capability_cache[model_name]
        
        model_config = self.models_config.get(model_name, {})
        capabilities = model_config.get('capabilities', ['general_qa'])
        
        # Cache the result
        self._capability_cache[model_name] = capabilities
        
        return capabilities
    
    def get_model_size(self, model_name: str) -> float:
        """Get the size of a model in billions of parameters.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Size in billions of parameters
        """
        model_config = self.models_config.get(model_name, {})
        return model_config.get('size_b', 1.0)
    
    def get_best_model_for_task(self, task_type: str, requires_tools: bool = False) -> Optional[str]:
        """Get the best model for a specific task type.
        
        Args:
            task_type: Type of task (e.g., 'coding', 'analysis', 'qa')
            requires_tools: Whether the task requires tool support
            
        Returns:
            Best model name or None if no suitable model found
        """
        suitable_models = []
        
        for model_name, config in self.models_config.items():
            # Check tool requirement
            if requires_tools and not config.get('supports_tools', False):
                continue
            
            # Check capabilities
            capabilities = config.get('capabilities', [])
            if task_type in capabilities or task_type in config.get('use_cases', []):
                model_size = config.get('size_b', 1.0)
                suitable_models.append((model_name, model_size))
        
        if not suitable_models:
            # Fallback: find any model that supports tools if required
            if requires_tools:
                for model_name, config in self.models_config.items():
                    if config.get('supports_tools', False):
                        return model_name
            return None
        
        # Sort by size (prefer larger models for better quality)
        suitable_models.sort(key=lambda x: x[1], reverse=True)
        
        # Return the largest suitable model
        return suitable_models[0][0]
    
    def get_alternative_model(self, original_model: str, requires_tools: bool = False) -> Optional[str]:
        """Get an alternative model when the original doesn't meet requirements.
        
        Args:
            original_model: The originally requested model
            requires_tools: Whether tool support is required
            
        Returns:
            Alternative model name or None
        """
        original_config = self.models_config.get(original_model, {})
        original_type = original_config.get('type', 'general')
        
        # Find models of similar type
        alternatives = []
        for model_name, config in self.models_config.items():
            if model_name == original_model:
                continue
            
            # Check tool requirement
            if requires_tools and not config.get('supports_tools', False):
                continue
            
            # Prefer same type
            if config.get('type') == original_type:
                model_size = config.get('size_b', 1.0)
                alternatives.append((model_name, model_size, 1))  # Priority 1 for same type
            else:
                model_size = config.get('size_b', 1.0)
                alternatives.append((model_name, model_size, 2))  # Priority 2 for different type
        
        if not alternatives:
            return None
        
        # Sort by priority then by size
        alternatives.sort(key=lambda x: (x[2], -x[1]))
        
        return alternatives[0][0]
    
    def validate_model_for_tools(self, model_name: str) -> Tuple[bool, Optional[str]]:
        """Validate if a model can be used for tool-requiring tasks.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            Tuple of (is_valid, alternative_model)
        """
        if self.supports_tools(model_name):
            return True, None
        else:
            # Find an alternative
            alternative = self.get_alternative_model(model_name, requires_tools=True)
            return False, alternative
    
    def get_available_models(self) -> List[str]:
        """Get list of all available models.
        
        Returns:
            List of model names
        """
        return list(self.models_config.keys())
    
    def get_tool_supporting_models(self) -> List[str]:
        """Get list of models that support tools.
        
        Returns:
            List of model names that support tools
        """
        tool_models = []
        for model_name, config in self.models_config.items():
            if config.get('supports_tools', False):
                tool_models.append(model_name)
        return tool_models
    
    def get_models_with_capability(self, capability: str) -> List[str]:
        """Get list of models that have a specific capability.
        
        Args:
            capability: The capability to search for (e.g., 'analysis', 'coding', 'reasoning')
        
        Returns:
            List of model names that have the specified capability
        """
        matching_models = []
        for model_name, config in self.models_config.items():
            model_capabilities = config.get('capabilities', [])
            if capability in model_capabilities:
                matching_models.append(model_name)
        return matching_models
    
    def ensure_model_available(self, model_name: str) -> bool:
        """Ensure a model is available, pulling it if necessary and auto_pull is enabled.
        
        Args:
            model_name: Name of the model to ensure availability
            
        Returns:
            True if model is available, False otherwise
        """
        # Check if model is already in our configuration
        if model_name in self.models_config:
            return True
        
        # If auto_pull is enabled and we have model discovery, try to pull
        if self.auto_pull and self.model_discovery:
            self.logger.info(f"Model {model_name} not found, attempting to pull...")
            if self.model_discovery.ensure_model_available(model_name, auto_pull=True):
                # Refresh configuration after successful pull
                self.refresh_configuration()
                return model_name in self.models_config
        
        return False
    
    def refresh_configuration(self) -> None:
        """Refresh the models configuration from the Ollama instance."""
        if update_models_config:
            if update_models_config(force_refresh=True):
                self.models_config = self._load_models_config(auto_update=False)
                # Clear caches
                self._tool_support_cache.clear()
                self._capability_cache.clear()
                self.logger.info("Model configuration refreshed")
            else:
                self.logger.warning("Failed to refresh model configuration")


# Global instance
_capability_checker = None


def get_capability_checker() -> ModelCapabilityChecker:
    """Get the global capability checker instance."""
    global _capability_checker
    if _capability_checker is None:
        _capability_checker = ModelCapabilityChecker()
    return _capability_checker


def reset_capability_checker():
    """Reset the global capability checker (useful for tests or config changes)."""
    global _capability_checker
    _capability_checker = None
