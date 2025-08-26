"""
Model Configuration Parser for Ollama Models
============================================

This module parses the models-full-list.json file and provides intelligent
model selection based on capabilities, size constraints, and tool support.
"""

import json
import os
from pathlib import Path
from typing import Optional

from src.core.model_tool_support import test_model_tool_support
from src.utils.enhanced_logging import get_logger


class ModelDescriptor:
    """Comprehensive model descriptor with all capabilities and constraints."""

    def __init__(self, name: str, model_data: dict[str, str], position: int = 0):
        """Initialize from JSON model data."""
        self.name = name
        self.tag = model_data.get("Tag / filename", name)
        self.position = position  # Position in the original list (lower = higher priority)
        self.sizes_variants = model_data.get("Sizes / variants", "")
        self.supports_tools = model_data.get("Supports Tools", "No").lower() == "yes"
        self.supports_vision = model_data.get("Supports Vision", "No").lower() == "yes"
        self.supports_thinking = model_data.get("Supports Thinking", "No").lower() == "yes"
        self.is_embedding = model_data.get("Embedding", "No").lower() == "yes"
        self.context_length = model_data.get("Context (if stated)", "")
        self.disk_size = model_data.get("Example disk size (latest)", "")
        self.notes = model_data.get("Notes", "")
        self.pull_command = model_data.get("How to pull (example)", "")

        # Computed properties
        self.size_variants_list = self._parse_size_variants()
        self.max_size_b = self._compute_max_size()
        self.size_b = self.max_size_b  # Alias for compatibility
        self.is_coding_focused = self._is_coding_focused()
        self.is_instruction_tuned = self._is_instruction_tuned()
        self.is_multimodal = self.supports_vision

    def _parse_size_variants(self) -> list[str]:
        """Parse size variants from the sizes_variants string."""
        if not self.sizes_variants or self.sizes_variants == "-":
            return []

        # Clean up the string and split
        variants = self.sizes_variants.replace(" ", "").lower()
        # Handle different formats: "1b, 3b, 7b" or "8x7b, 8x22b" or "16x17b, 128x17b (MoE)"
        variants = variants.replace("(moe)", "").replace("(", "").replace(")", "")

        return [v.strip() for v in variants.split(",") if v.strip()]

    def _compute_max_size(self) -> float:
        """Compute the maximum size in billions of parameters."""
        max_size = 0.0

        for variant in self.size_variants_list:
            try:
                # Handle different formats
                if "x" in variant:
                    # MoE format like "8x7b" - use the second number as it's the effective size
                    parts = variant.split("x")
                    if len(parts) == 2 and parts[1].endswith("b"):
                        size = float(parts[1][:-1])
                        max_size = max(max_size, size)
                elif variant.endswith("b"):
                    # Standard format like "7b"
                    size = float(variant[:-1])
                    max_size = max(max_size, size)
                elif variant.endswith("m"):
                    # Million parameters format like "335m"
                    size = float(variant[:-1]) / 1000.0  # Convert to billions
                    max_size = max(max_size, size)
            except (ValueError, IndexError):
                continue

        return max_size

    def _is_coding_focused(self) -> bool:
        """Determine if this model is coding-focused."""
        coding_keywords = [
            "code", "coder", "coding", "codestral", "codellama",
            "codegemma", "deepseek-coder", "devstral"
        ]
        tag_lower = self.name.lower()
        notes_lower = self.notes.lower()

        return any(keyword in tag_lower or keyword in notes_lower for keyword in coding_keywords)

    def _is_instruction_tuned(self) -> bool:
        """Determine if this model is instruction-tuned."""
        instruct_keywords = ["instruct", "instruction", "chat", "assistant"]
        return any(keyword in self.name.lower() or keyword in self.notes.lower()
                  for keyword in instruct_keywords)

    def supports_tools_capability(self) -> bool:
        """Check if this model supports tool use.

        This uses both the metadata and actual testing to confirm tool support.
        """
        # If metadata explicitly says it doesn't support tools, trust that
        if not self.supports_tools:
            return False

        # Otherwise verify with actual testing if possible
        try:
            # Use the tag for testing since that's what Ollama uses
            model_tag = self.tag
            if ":" not in model_tag and self.pull_command:
                # Extract the tag from the pull command if available
                model_tag = self.pull_command.split()[-1]

            return test_model_tool_support(model_tag)
        except Exception:
            # If testing fails, fall back to metadata
            return self.supports_tools

    def get_task_compatibility_score(self, task_type: str) -> float:
        """Get compatibility score for a specific task type."""
        score = 0.0

        if task_type == "code_analysis":
            if self.is_coding_focused:
                score += 0.5
            if self.supports_tools:
                score += 0.3
            if self.is_instruction_tuned:
                score += 0.2
        elif task_type == "tool_use":
            if self.supports_tools:
                score += 0.6
            if self.is_instruction_tuned:
                score += 0.4
        elif task_type == "general":
            if self.is_instruction_tuned:
                score += 0.4
            if not self.is_embedding:
                score += 0.3
            if self.supports_tools:
                score += 0.3
        else:
            # Unknown task, prefer general-purpose models
            if self.is_instruction_tuned:
                score += 0.5
            if not self.is_embedding:
                score += 0.5

        return min(score, 1.0)

    def get_position_priority_score(self) -> float:
        """Get priority score based on position in the original list (lower position = higher priority)."""
        # Normalize position to a 0-1 score where earlier positions get higher scores
        # Assuming max ~100 models, earlier models get higher priority
        return max(0.0, 1.0 - (self.position / 100.0))

    def meets_size_constraint(self, max_size_b: float = 14.0) -> bool:
        """Check if this model meets the size constraint."""
        return self.max_size_b <= max_size_b

    def meets_size_constraint(self, max_size_b: float = 14.0) -> bool:
        """Check if the model meets the size constraint."""
        return self.max_size_b <= max_size_b

    def get_recommended_variant(self, max_size_b: float = 14.0) -> Optional[str]:
        """Get the recommended variant within size constraints."""
        suitable_variants = []

        for variant in self.size_variants_list:
            try:
                if "x" in variant:
                    # MoE format
                    parts = variant.split("x")
                    if len(parts) == 2 and parts[1].endswith("b"):
                        size = float(parts[1][:-1])
                        if size <= max_size_b:
                            suitable_variants.append((variant, size))
                elif variant.endswith("b"):
                    # Standard format
                    size = float(variant[:-1])
                    if size <= max_size_b:
                        suitable_variants.append((variant, size))
                elif variant.endswith("m"):
                    # Million parameters
                    size = float(variant[:-1]) / 1000.0
                    if size <= max_size_b:
                        suitable_variants.append((variant, size))
            except (ValueError, IndexError):
                continue

        if not suitable_variants:
            return None

        # Return the largest suitable variant
        suitable_variants.sort(key=lambda x: x[1], reverse=True)
        return suitable_variants[0][0]


class ModelRegistry:
    """Registry for managing and querying model descriptors."""

    def __init__(self, models_json_path: str = "models-full-list.json"):
        """Initialize the model registry."""
        self.logger = get_logger()
        self.models_json_path = self._resolve_models_json_path(models_json_path)
        self.models: dict[str, ModelDescriptor] = {}
        self._tool_support_cache = {}  # Cache for models tested for tool support
        self._load_models()

    def _resolve_models_json_path(self, models_json_path: str) -> Path:
        """Resolve models JSON file path with multiple fallbacks."""
        # Try direct path first
        direct_path = Path(models_json_path)
        if direct_path.exists():
            return direct_path

        # Try relative to project root
        root_paths = [
            Path(os.path.join(os.path.dirname(__file__), "..", "..", models_json_path)),  # From src/core
            Path(os.path.join(os.getcwd(), models_json_path)),  # From current working dir
        ]

        for path in root_paths:
            if path.exists():
                return path

        # Return the original path - will be checked for existence later
        self.logger.warning(f"Models file not found at any standard locations: {models_json_path}")
        return direct_path

    def load_from_file(self, file_path: str) -> None:
        """Load model registry from a JSON file."""
        self.models_json_path = Path(file_path)
        self._load_models()

    def _load_models(self):
        """Load models from the JSON file."""
        if not self.models_json_path.exists():
            self.logger.warning(f"Models JSON file not found: {self.models_json_path}")
            return

        try:
            with open(self.models_json_path, encoding='utf-8') as f:
                models_data = json.load(f)

            # Enumerate to preserve position (order in the original list)
            for position, (model_name, model_info) in enumerate(models_data.items()):
                try:
                    descriptor = ModelDescriptor(model_name, model_info, position)
                    # Use the tag from JSON as the key, not the model_name
                    self.models[descriptor.tag] = descriptor

                except Exception as e:
                    self.logger.warning(f"Failed to parse model {model_name}: {e}")
                    continue

            self.logger.info(f"Loaded {len(self.models)} model descriptors")

        except Exception as e:
            self.logger.error(f"Failed to load models JSON: {e}")

    def get_models_for_task(self,
                           task_type: str,
                           requires_tools: bool = False,
                           max_size_b: float = 14.0,
                           exclude_embedding: bool = True) -> list[tuple[str, ModelDescriptor]]:
        """Get suitable models for a specific task type."""
        suitable_models = []

        for tag, model in self.models.items():
            # Skip if it doesn't meet basic requirements
            if exclude_embedding and model.is_embedding:
                continue

            if not model.meets_size_constraint(max_size_b):
                continue

            if requires_tools and not model.supports_tools:
                continue

            # Task-specific filtering with better scoring
            task_score = model.get_task_compatibility_score(task_type)
            position_score = model.get_position_priority_score()

            if task_type in ["code_analysis", "coding"]:
                if model.is_coding_focused or (model.supports_tools and model.is_instruction_tuned):
                    suitable_models.append((tag, model, task_score, position_score))

            elif task_type in ["tool_use", "file_operations"]:
                if model.supports_tools:
                    suitable_models.append((tag, model, task_score, position_score))

            elif task_type == "vision":
                if model.supports_vision:
                    suitable_models.append((tag, model, task_score, position_score))

            elif task_type == "reasoning":
                if model.supports_thinking or "reasoning" in model.notes.lower():
                    suitable_models.append((tag, model, task_score, position_score))

            elif task_type in ["general", "general_qa"]:
                if model.is_instruction_tuned and not model.is_embedding:
                    suitable_models.append((tag, model, task_score, position_score))

            else:  # Default: any suitable model
                if not model.is_embedding:
                    suitable_models.append((tag, model, task_score, position_score))

        # Sort by task compatibility score first, then by position priority, then by size
        suitable_models.sort(key=lambda x: (x[2], x[3], x[1].max_size_b), reverse=True)

        # Return just tag and model (remove scores)
        return [(tag, model) for tag, model, task_score, position_score in suitable_models]

    def get_best_model_for_task(self,
                               task_type: str,
                               requires_tools: bool = False,
                               max_size_b: float = 14.0) -> Optional[str]:
        """Get the best model for a specific task."""
        suitable_models = self.get_models_for_task(task_type, requires_tools, max_size_b)

        if not suitable_models:
            return None

        # Return the tag of the best model
        best_model = suitable_models[0]
        return best_model[0]

    def get_model_with_variant(self,
                              task_type: str,
                              requires_tools: bool = False,
                              max_size_b: float = 14.0) -> Optional[str]:
        """Get the best model with its recommended variant."""
        suitable_models = self.get_models_for_task(task_type, requires_tools, max_size_b)

        if not suitable_models:
            return None

        best_tag, best_model = suitable_models[0]
        variant = best_model.get_recommended_variant(max_size_b)

        if variant:
            return f"{best_tag}:{variant}"
        else:
            return best_tag

    def get_fallback_models(self, max_size_b: float = 14.0) -> list[str]:
        """Get a list of reliable fallback models."""
        fallback_candidates = [
            "qwen2.5", "llama3.1", "mistral", "qwen2.5-coder",
            "granite3.3", "hermes3", "nemotron-mini", "phi3:mini",
            "phi3:small", "llama3:8b", "mistral:7b"
        ]

        available_fallbacks = []
        for candidate in fallback_candidates:
            # Check if the candidate is a base model we have or a specific variant
            if ":" in candidate:
                base_name = candidate.split(":")[0]
                variant = candidate.split(":")[1]
                if base_name in self.models:
                    model = self.models[base_name]
                    if model.meets_size_constraint(max_size_b) and test_model_tool_support(candidate):
                        available_fallbacks.append(candidate)
            elif candidate in self.models:
                model = self.models[candidate]
                if model.meets_size_constraint(max_size_b) and model.supports_tools_capability():
                    variant = model.get_recommended_variant(max_size_b)
                    if variant:
                        full_name = f"{candidate}:{variant}"
                        # Verify the specific variant supports tools
                        if test_model_tool_support(full_name):
                            available_fallbacks.append(full_name)
                    else:
                        # Base model without variant
                        if test_model_tool_support(candidate):
                            available_fallbacks.append(candidate)

        return available_fallbacks

    def get_model(self, model_name: str) -> Optional[ModelDescriptor]:
        """Get a model descriptor by name."""
        return self.models.get(model_name)

    def get_best_model_for_task(self, task_type: str, max_size_b: float = 14.0) -> Optional[ModelDescriptor]:
        """Get the best model for a specific task."""
        suitable_models = self.get_models_for_task(task_type, max_size_b=max_size_b)
        return suitable_models[0][1] if suitable_models else None

    def validate_model_for_tools(self, model_tag: str) -> tuple[bool, str]:
        """Validate if a model supports tools."""
        base_tag = model_tag.split(':')[0]  # Remove variant

        if base_tag not in self.models:
            return False, f"Model {base_tag} not found in registry"

        model = self.models[base_tag]

        if not model.supports_tools:
            return False, f"Model {base_tag} does not support tools"

        if not model.meets_size_constraint():
            return False, f"Model {base_tag} exceeds size constraint ({model.max_size_b}B > 14B)"

        return True, f"Model {base_tag} supports tools and meets constraints"


# Global instance
_model_registry = None

def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry

def reset_model_registry():
    """Reset the global model registry instance."""
    global _model_registry
    _model_registry = None
