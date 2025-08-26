#!/usr/bin/env python3
"""
YAML Configuration Reader for Models

Handles reading and parsing the models.yaml configuration file.
Provides structured access to model definitions with metadata.
"""

import os
from dataclasses import dataclass
from typing import Any, Optional

import yaml


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    short_name: str
    name: str
    model_id: str
    provider: str
    description: str
    capabilities: dict[str, bool]
    parameters: dict[str, Any]
    tools: list[str]
    system_message: Optional[str] = None

    @property
    def supports_coding(self) -> bool:
        """Check if model supports coding capabilities."""
        if isinstance(self.capabilities, dict):
            return self.capabilities.get("coding", False)
        elif isinstance(self.capabilities, list):
            return "coding" in self.capabilities
        return False

    @property
    def supports_file_operations(self) -> bool:
        """Check if model supports file operations."""
        if isinstance(self.capabilities, dict):
            return self.capabilities.get("file_operations", False)
        elif isinstance(self.capabilities, list):
            return "file_operations" in self.capabilities
        return False

    @property
    def supports_streaming(self) -> bool:
        """Check if model supports streaming."""
        if isinstance(self.capabilities, dict):
            return self.capabilities.get("streaming", False)
        elif isinstance(self.capabilities, list):
            return "streaming" in self.capabilities
        return False


class ModelConfigReader:
    """
    Reads and manages model configurations from YAML files.

    Responsibilities:
    - Load models.yaml configuration
    - Parse model definitions into structured objects
    - Provide lookup and filtering capabilities
    - Handle configuration validation
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the model config reader.

        Args:
            config_path: Path to models.yaml file (auto-detected if None)
        """
        self.config_path = config_path or self._find_config_path()
        self._models: dict[str, ModelConfig] = {}
        self._loaded = False

    def _find_config_path(self) -> str:
        """Find the models.yaml configuration file."""
        # Try multiple possible locations
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "models.yaml"),
            os.path.join(os.path.dirname(__file__), "..", "config", "models.yaml"),
            os.path.join(os.path.dirname(__file__), "..", "..", "src", "config", "models.yaml"),
            "src/config/models.yaml",
            "models.yaml"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return os.path.abspath(path)

        raise FileNotFoundError(f"Could not find models.yaml in any of these locations: {possible_paths}")

    def load_models(self) -> dict[str, ModelConfig]:
        """
        Load models from the YAML configuration file.

        Returns:
            Dictionary of model configurations keyed by short name
        """
        if self._loaded:
            return self._models

        try:
            with open(self.config_path, encoding='utf-8') as f:
                config = yaml.safe_load(f)

            models_data = config.get('models', {})
            self._models.clear()

            for short_name, model_data in models_data.items():
                try:
                    model_config = ModelConfig(
                        short_name=short_name,
                        name=model_data.get('name', short_name),
                        model_id=model_data.get('model_id', short_name),
                        provider=model_data.get('provider', 'unknown'),
                        description=model_data.get('description', ''),
                        capabilities=model_data.get('capabilities', {}),
                        parameters=model_data.get('parameters', {}),
                        tools=model_data.get('tools', []),
                        system_message=model_data.get('system_message')
                    )
                    self._models[short_name] = model_config
                except Exception as e:
                    print(f"Warning: Failed to parse model '{short_name}': {e}")

            self._loaded = True
            print(f"Loaded {len(self._models)} models from {self.config_path}")

        except FileNotFoundError:
            print(f"Warning: Models config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            print(f"Warning: Failed to parse YAML config: {e}")
        except Exception as e:
            print(f"Warning: Failed to load models config: {e}")

        return self._models

    def get_model(self, short_name: str) -> Optional[ModelConfig]:
        """
        Get a specific model configuration.

        Args:
            short_name: Short name of the model

        Returns:
            ModelConfig if found, None otherwise
        """
        if not self._loaded:
            self.load_models()
        return self._models.get(short_name)

    def get_all_models(self) -> list[ModelConfig]:
        """
        Get all model configurations.

        Returns:
            List of all model configurations
        """
        if not self._loaded:
            self.load_models()
        return list(self._models.values())

    def get_models_by_provider(self, provider: str) -> list[ModelConfig]:
        """
        Get models filtered by provider.

        Args:
            provider: Provider name (e.g., 'ollama')

        Returns:
            List of models from the specified provider
        """
        if not self._loaded:
            self.load_models()
        return [model for model in self._models.values() if model.provider == provider]

    def get_coding_models(self) -> list[ModelConfig]:
        """
        Get models that support coding capabilities.

        Returns:
            List of coding-capable models
        """
        if not self._loaded:
            self.load_models()
        return [model for model in self._models.values() if model.supports_coding]

    def find_model_by_id(self, model_id: str) -> Optional[ModelConfig]:
        """
        Find a model by its model_id.

        Args:
            model_id: The model identifier (e.g., 'deepcoder:14b')

        Returns:
            ModelConfig if found, None otherwise
        """
        if not self._loaded:
            self.load_models()

        for model in self._models.values():
            if model.model_id == model_id:
                return model
        return None

    def get_model_choices(self) -> list[dict[str, str]]:
        """
        Get model choices suitable for UI selection.

        Returns:
            List of dictionaries with 'short_name', 'name', and 'description'
        """
        if not self._loaded:
            self.load_models()

        choices = []
        for model in self._models.values():
            choices.append({
                "short_name": model.short_name,
                "name": model.name,
                "description": model.description,
                "model_id": model.model_id,
                "provider": model.provider
            })

        return sorted(choices, key=lambda x: x['name'])

    def reload(self):
        """Reload the configuration from disk."""
        self._loaded = False
        self._models.clear()
        self.load_models()

    def validate_config(self) -> list[str]:
        """
        Validate the loaded configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        if not self._loaded:
            self.load_models()

        errors = []

        for short_name, model in self._models.items():
            # Check required fields
            if not model.model_id:
                errors.append(f"Model '{short_name}' missing model_id")
            if not model.provider:
                errors.append(f"Model '{short_name}' missing provider")

            # Check capabilities structure
            if not isinstance(model.capabilities, dict):
                errors.append(f"Model '{short_name}' capabilities must be a dictionary")

            # Check parameters structure
            if not isinstance(model.parameters, dict):
                errors.append(f"Model '{short_name}' parameters must be a dictionary")

            # Check tools structure
            if not isinstance(model.tools, list):
                errors.append(f"Model '{short_name}' tools must be a list")

        return errors
