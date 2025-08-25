"""Model capability checking and validation."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from src.utils.enhanced_logging import get_logger

try:
    from src.core.model_config_updater import get_available_models, validate_models_config, update_models_config
    from src.core.model_discovery import OllamaModelDiscovery
    from src.core.model_registry import get_model_registry
except ImportError:
    # Fallback if modules are not available
    get_available_models = None
    validate_models_config = None
    update_models_config = None
    OllamaModelDiscovery = None
    get_model_registry = None


class ModelCapabilityChecker:
    """Validates model capabilities and provides intelligent model selection."""
    
    def __init__(self, auto_update: bool = True, auto_pull: bool = False, max_model_size_b: float = 14.0):
        """Initialize the capability checker.
        
        Args:
            auto_update: Whether to automatically update model configuration
            auto_pull: Whether to automatically pull missing models
            max_model_size_b: Maximum model size in billions of parameters
        """
        self.logger = get_logger()
        self.auto_pull = auto_pull
        self.max_model_size_b = max_model_size_b
        self.model_discovery = OllamaModelDiscovery() if OllamaModelDiscovery else None
        self.model_registry = get_model_registry() if get_model_registry else None
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
        
        # Check configuration first
        model_config = self.models_config.get(model_name, {})
        supports_tools_metadata = model_config.get('supports_tools', False)
        
        # If metadata says it supports tools, verify through testing
        if supports_tools_metadata:
            try:
                # Use the tool support testing mechanism
                from src.core.model_tool_support import test_model_tool_support
                supports_tools = test_model_tool_support(model_name)
                if not supports_tools:
                    self.logger.warning(f"Model {model_name} claims tool support in metadata but failed verification")
            except Exception as e:
                self.logger.error(f"Error testing tool support for {model_name}: {e}")
                # Fall back to metadata if testing fails
                supports_tools = supports_tools_metadata
        else:
            # If metadata says it doesn't support tools, trust that
            supports_tools = False
        
        # Cache the result
        self._tool_support_cache[model_name] = supports_tools
        
        self.logger.debug(f"Model {model_name} supports tools: {supports_tools} (verified)")
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
        
        # Handle both list and dictionary capability formats
        if isinstance(capabilities, dict):
            # If it's a dictionary, extract keys where value is True
            capabilities = [key for key, value in capabilities.items() if value]
        elif not isinstance(capabilities, list):
            # If it's neither list nor dict, default to empty list
            capabilities = ['general_qa']
            
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
        """Get the best model for a specific task type using the model registry.
        
        Args:
            task_type: Type of task (coding, file_operations, general_qa, etc.)
            requires_tools: Whether the task requires tool support
            
        Returns:
            Model name if found, None otherwise
        """
        # First try using the new model registry
        if self.model_registry:
            try:
                # Get the best model with size constraints
                best_model = self.model_registry.get_model_with_variant(
                    task_type=task_type,
                    requires_tools=requires_tools,
                    max_size_b=self.max_model_size_b
                )
                
                if best_model:
                    # Verify the model exists before recommending
                    from src.core.model_discovery import model_exists
                    if model_exists(best_model):
                        # If tools are required, verify the model actually supports them
                        if requires_tools:
                            from src.core.model_tool_support import test_model_tool_support
                            if test_model_tool_support(best_model):
                                self.logger.info(f"Registry recommended model {best_model} for {task_type} with verified tool support")
                                return best_model
                            else:
                                self.logger.warning(f"Registry model {best_model} claims tool support but verification failed")
                        else:
                            # Tools not required, can use as is
                            self.logger.info(f"Registry recommended model {best_model} for {task_type} (tools not required)")
                            return best_model
                    else:
                        self.logger.debug(f"Registry model {best_model} not available locally")
                
            except Exception as e:
                self.logger.warning(f"Error using model registry: {e}")
        
        # Fallback to original logic with size constraints
        result = self._get_best_model_fallback(task_type, requires_tools)
        
        # If no suitable model found but tools are required, try one last fallback to hardcoded models
        if result is None and requires_tools:
            self.logger.warning(f"No suitable model found for {task_type} with tool support, trying hardcoded fallbacks")
            
            # Try a list of known tool-supporting models
            hardcoded_tool_models = [
                "llama3:8b", "phi3:small", "qwen2.5:7b", "mistral:7b", "phi3:mini",
                "qwen2.5-coder:7b", "hermes3:8b", "nemotron-mini:4b"
            ]
            
            for model in hardcoded_tool_models:
                from src.core.model_discovery import model_exists
                if model_exists(model):
                    self.logger.info(f"Using hardcoded fallback model: {model}")
                    return model
            
            # Last resort fallback - use any available model
            self.logger.warning("No suitable model found, using default model")
            return self.get_default_model()
        
        return result
    
    def _get_best_model_fallback(self, task_type: str, requires_tools: bool = False) -> Optional[str]:
        """Fallback method for model selection using original logic."""
        suitable_models = []
        
        # Create model discovery instance to check existence
        discovery = None
        try:
            from src.core.model_discovery import OllamaModelDiscovery
            discovery = OllamaModelDiscovery()
        except ImportError:
            self.logger.warning("Could not import OllamaModelDiscovery to verify model existence")
            
        # Import tool support checker
        from src.core.model_tool_support import test_model_tool_support
            
        # First pass - find models that match the task and size constraints
        for model_name, config in self.models_config.items():
            # Apply size constraint
            model_size = config.get('size_b', 1.0)
            if model_size > self.max_model_size_b:
                self.logger.debug(f"Skipping {model_name}: size {model_size}B > {self.max_model_size_b}B")
                continue
                
            # Skip models that don't exist if we can check
            if discovery and not discovery.model_exists(model_name):
                self.logger.debug(f"Skipping non-existent model: {model_name}")
                continue
                
            # Check tool requirement in metadata first
            if requires_tools and not config.get('supports_tools', False):
                continue
                
            # Check capabilities - handle both list and dict formats
            capabilities = config.get('capabilities', [])
            use_cases = config.get('use_cases', [])
            
            # Check if the task type matches capabilities or use cases
            if (isinstance(capabilities, list) and task_type in capabilities) or \
               (isinstance(capabilities, dict) and capabilities.get(task_type, False)) or \
               task_type in use_cases:
                # If tools required, verify actual tool support
                if requires_tools:
                    if test_model_tool_support(model_name):
                        self.logger.debug(f"Model {model_name} supports {task_type} with confirmed tool support")
                        suitable_models.append((model_name, model_size, 1))  # Priority 1: Task match + confirmed tool support
                    else:
                        self.logger.debug(f"Model {model_name} supports {task_type} but failed tool support verification")
                else:
                    suitable_models.append((model_name, model_size, 2))  # Priority 2: Task match but tools not required
        
        # If no suitable models with task match, try fallbacks
        if not suitable_models:
            self.logger.debug(f"No models found for task {task_type}, trying fallbacks")
            
            # Find any model that supports tools if required
            if requires_tools:
                for model_name, config in self.models_config.items():
                    model_size = config.get('size_b', 1.0)
                    if model_size > self.max_model_size_b:
                        continue
                        
                    # Verify model exists if possible
                    if discovery and not discovery.model_exists(model_name):
                        continue
                        
                    # Check for tool support in metadata
                    if config.get('supports_tools', False):
                        # Verify actual tool support
                        if test_model_tool_support(model_name):
                            self.logger.info(f"Found fallback model with confirmed tool support: {model_name}")
                            return model_name
                
                # If no models with verified tool support, but we need tools,
                # try using any model that claims to support tools (but might not actually do so)
                self.logger.warning("No models with verified tool support found, trying any model that claims tool support")
                for model_name, config in self.models_config.items():
                    model_size = config.get('size_b', 1.0)
                    if model_size > self.max_model_size_b:
                        continue
                        
                    # Verify model exists if possible
                    if discovery and not discovery.model_exists(model_name):
                        continue
                        
                    # Check for tool support in metadata
                    if config.get('supports_tools', False):
                        self.logger.warning(f"Using {model_name} which claims tool support but was not verified")
                        return model_name
            
            # If still no suitable models
            return None
        
        # Sort by priority (1 is better than 2) and then by size (larger models preferred)
        suitable_models.sort(key=lambda x: (x[2], -x[1]))
        
        # Return the best suitable model
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
            List of model names that support tools with verification
        """
        tool_models = []
        from src.core.model_tool_support import test_model_tool_support
        
        for model_name, config in self.models_config.items():
            # First check metadata
            if config.get('supports_tools', False):
                # Then verify with actual testing
                try:
                    if test_model_tool_support(model_name):
                        tool_models.append(model_name)
                    else:
                        self.logger.warning(f"Model {model_name} claims tool support but failed verification")
                except Exception as e:
                    self.logger.error(f"Error testing tool support for {model_name}: {e}")
        
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
    
    def get_default_model(self) -> str:
        """Get a safe default model that's likely to be available.
        
        Returns:
            Name of a default model to use as fallback
        """
        # First try using model registry for fallbacks
        if self.model_registry:
            try:
                fallback_models = self.model_registry.get_fallback_models(max_size_b=self.max_model_size_b)
                if fallback_models:
                    # Try each fallback in order
                    for model in fallback_models:
                        from src.core.model_discovery import model_exists
                        if model_exists(model):
                            # Verify model actually supports tools
                            from src.core.model_tool_support import test_model_tool_support
                            if test_model_tool_support(model):
                                self.logger.info(f"Selected default model with confirmed tool support: {model}")
                                return model
                            else:
                                self.logger.debug(f"Model exists but doesn't support tools: {model}")
            except Exception as e:
                self.logger.warning(f"Error getting registry fallbacks: {e}")
        
        # Hardcoded fallbacks within size constraints - curated list of models known to work with tools
        default_options = [
            "llama3:8b", "phi3:small", "mistral:7b", "phi3:mini",
            "qwen2.5:7b", "llama3.1:8b", "granite3.3:8b", "qwen2.5-coder:7b",
            "hermes3:8b", "nemotron-mini:4b"
        ]
        
        # Check if any of our default options exist
        for model in default_options:
            from src.core.model_discovery import model_exists
            if model_exists(model):
                # Verify model actually supports tools
                from src.core.model_tool_support import test_model_tool_support
                if test_model_tool_support(model):
                    self.logger.info(f"Selected default model with confirmed tool support: {model}")
                    return model
                else:
                    self.logger.debug(f"Model exists but doesn't support tools: {model}")
        
        # If none of the defaults are available, get the first available model within size constraint
        available_models = self.get_available_models()
        if available_models:
            for model in available_models:
                # Try to extract size info and check constraint
                try:
                    if ":" in model:
                        size_part = model.split(":")[1]
                        if size_part.endswith("b"):
                            size_b = float(size_part[:-1])
                            if size_b <= self.max_model_size_b:
                                # Verify model actually supports tools
                                from src.core.model_tool_support import test_model_tool_support
                                if test_model_tool_support(model):
                                    self.logger.info(f"Selected first available model with tool support: {model}")
                                    return model
                    else:
                        # If no size info, verify tool support
                        from src.core.model_tool_support import test_model_tool_support
                        if test_model_tool_support(model):
                            self.logger.info(f"Selected model without size info but with tool support: {model}")
                            return model
                except (ValueError, IndexError):
                    continue
            
            # If no tool-supporting models, try one more time without tool support check
            self.logger.warning("No models with confirmed tool support found, using first available model")
            return available_models[0]
            
        # Final fallback - smallest reasonable model
        return "qwen2.5:3b"
    
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
