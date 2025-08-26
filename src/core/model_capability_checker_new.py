"""Model capability checking and validation - Modular facade."""

# Backwards compatibility facade that delegates to specialized model capability components
from .model_capabilities import ModelConfigManager, ModelCapabilityValidator, ModelSelector


class ModelCapabilityChecker:
    """Validates model capabilities and provides intelligent model selection."""

    def __init__(self, auto_update: bool = True, auto_pull: bool = False, max_model_size_b: float = 14.0):
        """Initialize the capability checker.

        Args:
            auto_update: Whether to automatically update model configuration
            auto_pull: Whether to automatically pull missing models
            max_model_size_b: Maximum model size in billions of parameters
        """
        self.auto_pull = auto_pull
        
        # Initialize components
        self.config_manager = ModelConfigManager(auto_update=auto_update)
        self.models_config = self.config_manager.load_models_config()
        self.validator = ModelCapabilityValidator(self.models_config)
        self.selector = ModelSelector(self.validator, max_model_size_b)

    def supports_tools(self, model_name: str) -> bool:
        """Check if a model supports tool usage."""
        return self.validator.supports_tools(model_name)

    def get_model_capabilities(self, model_name: str) -> list[str]:
        """Get list of capabilities for a model."""
        return self.validator.get_model_capabilities(model_name)

    def get_model_size(self, model_name: str) -> float:
        """Get the size of a model in billions of parameters."""
        return self.validator.get_model_size(model_name)

    def get_best_model_for_task(self, task_type: str, requires_tools: bool = False):
        """Get the best model for a specific task type."""
        return self.selector.get_best_model_for_task(task_type, requires_tools)

    def get_alternative_model(self, original_model: str, requires_tools: bool = False):
        """Get an alternative model to the original one."""
        return self.selector.get_alternative_model(original_model, requires_tools)

    def validate_model_for_tools(self, model_name: str) -> tuple[bool, str]:
        """Validate if a model can be used for tool-based tasks."""
        return self.validator.validate_model_for_tools(model_name)

    def get_available_models(self) -> list[str]:
        """Get list of available models."""
        return self.validator.get_available_models()

    def get_tool_supporting_models(self) -> list[str]:
        """Get all models that support tool usage."""
        return self.validator.get_tool_supporting_models()

    def get_models_with_capability(self, capability: str) -> list[str]:
        """Get all models that have a specific capability."""
        return self.validator.get_models_with_capability(capability)

    def ensure_model_available(self, model_name: str) -> bool:
        """Ensure a model is available, pulling it if necessary."""
        return self.config_manager.ensure_model_available(model_name)

    def get_default_model(self) -> str:
        """Get the default model for general use."""
        return self.selector.get_default_model()

    def refresh_configuration(self) -> None:
        """Force refresh of the models configuration."""
        self.models_config = self.config_manager.refresh_configuration()
        self.validator = ModelCapabilityValidator(self.models_config)
        self.selector = ModelSelector(self.validator, self.selector.max_model_size_b)


# Global instance management for backwards compatibility
_capability_checker_instance = None


def get_capability_checker() -> ModelCapabilityChecker:
    """Get a singleton instance of the model capability checker."""
    global _capability_checker_instance
    if _capability_checker_instance is None:
        _capability_checker_instance = ModelCapabilityChecker()
    return _capability_checker_instance


def reset_capability_checker():
    """Reset the global capability checker instance."""
    global _capability_checker_instance
    _capability_checker_instance = None
