"""Model capability validation and checking."""

from pathlib import Path
from typing import Any, Dict, List, Optional

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


class ModelCapabilityValidator:
    """Validates model capabilities and provides capability information."""

    def __init__(self, models_config: Dict[str, Any]):
        """Initialize the capability validator.
        
        Args:
            models_config: Models configuration dictionary
        """
        self.logger = get_logger()
        self.models_config = models_config
        self._capability_cache = {}

    def supports_tools(self, model_name: str) -> bool:
        """Check if a model supports tool usage.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model supports tools, False otherwise
        """
        # Check cache first
        if model_name in self._capability_cache:
            return self._capability_cache[model_name].get('supports_tools', False)

        supports_tools = False
        
        # Check in models configuration
        for model_config in self.models_config.get("models", []):
            if model_config.get("name") == model_name:
                supports_tools = model_config.get("supports_tools", False)
                break
        
        # Cache the result
        if model_name not in self._capability_cache:
            self._capability_cache[model_name] = {}
        self._capability_cache[model_name]['supports_tools'] = supports_tools
        
        self.logger.debug(f"Model {model_name} tool support: {supports_tools}")
        return supports_tools

    def get_model_capabilities(self, model_name: str) -> List[str]:
        """Get list of capabilities for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of capability strings
        """
        # Check cache first
        if model_name in self._capability_cache:
            cached_caps = self._capability_cache[model_name].get('capabilities')
            if cached_caps is not None:
                return cached_caps

        capabilities = []
        
        # Find model in configuration
        for model_config in self.models_config.get("models", []):
            if model_config.get("name") == model_name:
                capabilities = model_config.get("capabilities", [])
                break
        
        # Default capabilities based on model name patterns
        if not capabilities:
            capabilities = self._infer_capabilities_from_name(model_name)
        
        # Cache the result
        if model_name not in self._capability_cache:
            self._capability_cache[model_name] = {}
        self._capability_cache[model_name]['capabilities'] = capabilities
        
        return capabilities

    def get_model_size(self, model_name: str) -> float:
        """Get the size of a model in billions of parameters.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model size in billions of parameters
        """
        # Check in models configuration
        for model_config in self.models_config.get("models", []):
            if model_config.get("name") == model_name:
                return model_config.get("size_b", 7.0)  # Default to 7B
        
        # Try to infer from model name
        return self._infer_size_from_name(model_name)

    def validate_model_for_tools(self, model_name: str) -> tuple[bool, Optional[str]]:
        """Validate if a model can be used for tool-based tasks.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.supports_tools(model_name):
            return False, f"Model {model_name} does not support tool usage"
        
        # Check if model is available
        available_models = self.get_available_models()
        if model_name not in available_models:
            return False, f"Model {model_name} is not available"
        
        return True, None

    def get_models_with_capability(self, capability: str) -> List[str]:
        """Get all models that have a specific capability.
        
        Args:
            capability: Capability to search for
            
        Returns:
            List of model names with the capability
        """
        matching_models = []
        
        for model_config in self.models_config.get("models", []):
            model_name = model_config.get("name")
            capabilities = model_config.get("capabilities", [])
            
            if capability.lower() in [cap.lower() for cap in capabilities]:
                matching_models.append(model_name)
        
        return matching_models

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        if get_available_models:
            return get_available_models()
        
        # Fallback: return models from config
        return [model.get("name") for model in self.models_config.get("models", [])]

    def get_tool_supporting_models(self) -> List[str]:
        """Get all models that support tool usage.
        
        Returns:
            List of model names that support tools
        """
        tool_models = []
        
        for model_config in self.models_config.get("models", []):
            model_name = model_config.get("name")
            if model_config.get("supports_tools", False):
                tool_models.append(model_name)
        
        # If no models marked as supporting tools, use heuristics
        if not tool_models:
            all_models = self.get_available_models()
            for model in all_models:
                if self._likely_supports_tools(model):
                    tool_models.append(model)
        
        return tool_models

    def _infer_capabilities_from_name(self, model_name: str) -> List[str]:
        """Infer capabilities from model name patterns."""
        capabilities = ["general"]
        
        model_lower = model_name.lower()
        
        if "coder" in model_lower or "code" in model_lower:
            capabilities.extend(["coding", "programming", "debugging"])
        
        if "instruct" in model_lower:
            capabilities.append("instruction_following")
        
        if "chat" in model_lower:
            capabilities.append("conversation")
        
        if "math" in model_lower:
            capabilities.append("mathematics")
        
        if "vision" in model_lower:
            capabilities.append("visual_processing")
        
        return capabilities

    def _infer_size_from_name(self, model_name: str) -> float:
        """Infer model size from name patterns."""
        # Common size patterns in model names
        size_patterns = {
            "1b": 1.0, "2b": 2.0, "3b": 3.0, "6.7b": 6.7, "7b": 7.0,
            "8b": 8.0, "13b": 13.0, "14b": 14.0, "30b": 30.0, "70b": 70.0
        }
        
        model_lower = model_name.lower()
        for pattern, size in size_patterns.items():
            if pattern in model_lower:
                return size
        
        # Default size if not found
        return 7.0

    def _likely_supports_tools(self, model_name: str) -> bool:
        """Heuristic to determine if a model likely supports tools."""
        model_lower = model_name.lower()
        
        # Models that typically support tools
        tool_indicators = ["coder", "instruct", "chat"]
        size_indicators = ["7b", "8b", "13b", "14b"]  # Larger models more likely
        
        has_tool_indicator = any(indicator in model_lower for indicator in tool_indicators)
        has_size_indicator = any(indicator in model_lower for indicator in size_indicators)
        
        return has_tool_indicator or has_size_indicator

    def refresh_cache(self):
        """Clear the capability cache to force refresh."""
        self._capability_cache.clear()
        self.logger.debug("Capability cache cleared")
