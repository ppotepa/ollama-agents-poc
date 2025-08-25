"""Model capability checker - determines which models support tools and specific capabilities."""

import yaml
import os
from typing import Dict, List, Set, Optional
from pathlib import Path

from src.utils.enhanced_logging import get_logger


class ModelCapabilityChecker:
    """Model capability checking and validation."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from src.utils.enhanced_logging import get_logger
from src.core.model_config_updater import get_available_models, validate_models_config, update_models_config


class ModelCapabilityChecker:
    """Validates model capabilities and provides intelligent model selection."""
    
    def __init__(self, auto_update: bool = True):
        """Initialize the capability checker.
        
        Args:
            auto_update: Whether to automatically update model configuration
        """
        self.logger = get_logger()
        self.models_config = self._load_models_config(auto_update)
        
    def _load_models_config(self, auto_update: bool = True) -> Dict[str, Any]:
        """Load models configuration with auto-update capability."""
        config_path = Path("src/config/models.yaml")
        
        # Check if configuration exists and is valid
        if auto_update and (not config_path.exists() or not validate_models_config()):
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
            
        except Exception as e:
            self.logger.error(f"Failed to get dynamic configuration: {e}")
            return {}
    
    def __init__(self, models_config_path: Optional[str] = None):
        """Initialize the capability checker.
        
        Args:
            models_config_path: Path to models.yaml configuration file
        """
        self.logger = get_logger()
        
        if models_config_path is None:
            # Default path - go up to src directory then into config
            script_dir = Path(__file__).parent.parent  # src/core -> src
            models_config_path = script_dir / "config" / "models.yaml"
        
        self.models_config_path = models_config_path
        self.models_config = self._load_models_config()
        
        # Cache for capability lookups
        self._tool_support_cache = {}
        self._capability_cache = {}
    
    def _load_models_config(self) -> Dict:
        """Load the models configuration."""
        try:
            with open(self.models_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self.logger.debug(f"Loaded models config from {self.models_config_path}")
                return config.get('models', {})
        except Exception as e:
            self.logger.error(f"Failed to load models config from {self.models_config_path}: {e}")
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
        
        # Check direct model name
        if model_name in self.models_config:
            tools = self.models_config[model_name].get('tools', [])
            supports = len(tools) > 0
            self._tool_support_cache[model_name] = supports
            return supports
        
        # Check by model_id
        for model_key, model_config in self.models_config.items():
            if model_config.get('model_id') == model_name:
                tools = model_config.get('tools', [])
                supports = len(tools) > 0
                self._tool_support_cache[model_name] = supports
                return supports
        
        # Default to False if model not found
        self.logger.warning(f"Model {model_name} not found in configuration, assuming no tool support")
        self._tool_support_cache[model_name] = False
        return False
    
    def get_tool_supporting_models(self) -> List[str]:
        """Get list of all models that support tools.
        
        Returns:
            List of model names that support tools
        """
        tool_models = []
        
        for model_key, model_config in self.models_config.items():
            tools = model_config.get('tools', [])
            if len(tools) > 0:
                model_id = model_config.get('model_id', model_key)
                tool_models.append(model_id)
        
        return tool_models
    
    def get_non_tool_models(self) -> List[str]:
        """Get list of models that don't support tools.
        
        Returns:
            List of model names that don't support tools
        """
        non_tool_models = []
        
        for model_key, model_config in self.models_config.items():
            tools = model_config.get('tools', [])
            if len(tools) == 0:
                model_id = model_config.get('model_id', model_key)
                non_tool_models.append(model_id)
        
        return non_tool_models
    
    def has_capability(self, model_name: str, capability: str) -> bool:
        """Check if a model has a specific capability.
        
        Args:
            model_name: Name of the model to check
            capability: Capability to check for (e.g., 'coding', 'general_qa')
            
        Returns:
            True if the model has the capability, False otherwise
        """
        cache_key = f"{model_name}_{capability}"
        if cache_key in self._capability_cache:
            return self._capability_cache[cache_key]
        
        # Check direct model name
        if model_name in self.models_config:
            capabilities = self.models_config[model_name].get('capabilities', {})
            has_cap = capabilities.get(capability, False)
            self._capability_cache[cache_key] = has_cap
            return has_cap
        
        # Check by model_id
        for model_key, model_config in self.models_config.items():
            if model_config.get('model_id') == model_name:
                capabilities = model_config.get('capabilities', {})
                has_cap = capabilities.get(capability, False)
                self._capability_cache[cache_key] = has_cap
                return has_cap
        
        # Default to False if model not found
        self._capability_cache[cache_key] = False
        return False
    
    def get_models_with_capability(self, capability: str) -> List[str]:
        """Get models that have a specific capability.
        
        Args:
            capability: Capability to search for
            
        Returns:
            List of model names with the capability
        """
        matching_models = []
        
        for model_key, model_config in self.models_config.items():
            capabilities = model_config.get('capabilities', {})
            if capabilities.get(capability, False):
                model_id = model_config.get('model_id', model_key)
                matching_models.append(model_id)
        
        return matching_models
    
    def get_best_model_for_task(self, task_type: str, requires_tools: bool = False) -> Optional[str]:
        """Get the best model for a specific task type.
        
        Args:
            task_type: Type of task (coding, research, etc.)
            requires_tools: Whether the task requires tool support
            
        Returns:
            Best model name for the task, or None if no suitable model found
        """
        # Define capability mappings for task types
        capability_mappings = {
            'coding': 'coding',
            'research': 'general_qa',
            'file_analysis': 'file_operations',
            'system_operation': 'general_qa',
            'data_processing': 'general_qa',
            'creative': 'general_qa',
            'general_qa': 'general_qa'
        }
        
        required_capability = capability_mappings.get(task_type.lower(), 'general_qa')
        
        # Get models with the required capability
        capable_models = self.get_models_with_capability(required_capability)
        
        if requires_tools:
            # Filter to only models that support tools
            tool_models = self.get_tool_supporting_models()
            suitable_models = [m for m in capable_models if m in tool_models]
        else:
            suitable_models = capable_models
        
        if not suitable_models:
            self.logger.warning(f"No suitable models found for task {task_type} (tools required: {requires_tools})")
            return None
        
        # Prioritize models based on task type
        model_priorities = self._get_model_priorities_for_task(task_type)
        
        # Sort by priority
        prioritized_models = []
        for priority_model in model_priorities:
            if priority_model in suitable_models:
                prioritized_models.append(priority_model)
        
        # Add any remaining models
        for model in suitable_models:
            if model not in prioritized_models:
                prioritized_models.append(model)
        
        return prioritized_models[0] if prioritized_models else None
    
    def _get_model_priorities_for_task(self, task_type: str) -> List[str]:
        """Get model priorities for a specific task type.
        
        Args:
            task_type: Type of task
            
        Returns:
            List of models in priority order
        """
        # Define model priorities by task type
        priorities = {
            'coding': [
                'qwen2.5-coder:7b',
                'deepseek-coder:6.7b',
                'codellama:13b-instruct',
                'deepcoder:14b',
                'qwen2.5:7b-instruct-q4_K_M'
            ],
            'research': [
                'qwen2.5:7b-instruct-q4_K_M',
                'llama3.3:70b-instruct-q2_K',
                'mistral:7b-instruct',
                'gemma:7b-instruct-q4_K_M'
            ],
            'file_analysis': [
                'qwen2.5:7b-instruct-q4_K_M',
                'deepcoder:14b',
                'gemma:7b-instruct-q4_K_M'
            ],
            'system_operation': [
                'qwen2.5:7b-instruct-q4_K_M',
                'phi3:mini',
                'gemma:7b-instruct-q4_K_M'
            ],
            'general_qa': [
                'qwen2.5:7b-instruct-q4_K_M',
                'gemma:7b-instruct-q4_K_M',
                'mistral:7b-instruct'
            ]
        }
        
        return priorities.get(task_type.lower(), priorities['general_qa'])
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get complete information about a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        # Check direct model name
        if model_name in self.models_config:
            return self.models_config[model_name]
        
        # Check by model_id
        for model_key, model_config in self.models_config.items():
            if model_config.get('model_id') == model_name:
                return model_config
        
        return {}
    
    def validate_model_for_tools(self, model_name: str, required_tools: List[str]) -> bool:
        """Validate if a model supports all required tools.
        
        Args:
            model_name: Name of the model to check
            required_tools: List of required tools
            
        Returns:
            True if model supports all required tools
        """
        model_info = self.get_model_info(model_name)
        available_tools = model_info.get('tools', [])
        
        # Check if all required tools are available
        for tool in required_tools:
            if tool not in available_tools:
                return False
        
        return True
    
    def get_alternative_model(self, current_model: str, task_type: str, requires_tools: bool = False) -> Optional[str]:
        """Get an alternative model for the same task type.
        
        Args:
            current_model: Current model that's not working
            task_type: Type of task
            requires_tools: Whether the task requires tools
            
        Returns:
            Alternative model name, or None if no alternative found
        """
        # Get all suitable models for the task
        all_suitable = []
        
        capability_mappings = {
            'coding': 'coding',
            'research': 'general_qa',
            'file_analysis': 'file_operations',
            'system_operation': 'general_qa',
            'data_processing': 'general_qa',
            'creative': 'general_qa',
            'general_qa': 'general_qa'
        }
        
        required_capability = capability_mappings.get(task_type.lower(), 'general_qa')
        capable_models = self.get_models_with_capability(required_capability)
        
        if requires_tools:
            tool_models = self.get_tool_supporting_models()
            all_suitable = [m for m in capable_models if m in tool_models and m != current_model]
        else:
            all_suitable = [m for m in capable_models if m != current_model]
        
        if not all_suitable:
            return None
        
        # Return first suitable alternative
        priorities = self._get_model_priorities_for_task(task_type)
        for priority_model in priorities:
            if priority_model in all_suitable:
                return priority_model
        
        return all_suitable[0] if all_suitable else None


# Global instance
_capability_checker = None


def get_capability_checker() -> ModelCapabilityChecker:
    """Get the global capability checker instance."""
    global _capability_checker
    if _capability_checker is None:
        _capability_checker = ModelCapabilityChecker()
    return _capability_checker


__all__ = ['ModelCapabilityChecker', 'get_capability_checker']
