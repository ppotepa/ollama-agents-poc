"""Model selection logic for choosing the best model for tasks."""

from typing import List, Optional

from src.utils.enhanced_logging import get_logger

from .validator import ModelCapabilityValidator


class ModelSelector:
    """Intelligent model selection based on task requirements."""

    def __init__(self, validator: ModelCapabilityValidator, max_model_size_b: float = 14.0):
        """Initialize the model selector.
        
        Args:
            validator: Model capability validator instance
            max_model_size_b: Maximum model size in billions of parameters
        """
        self.logger = get_logger()
        self.validator = validator
        self.max_model_size_b = max_model_size_b

    def get_best_model_for_task(self, task_type: str, requires_tools: bool = False) -> Optional[str]:
        """Get the best model for a specific task type.
        
        Args:
            task_type: Type of task (e.g., 'coding', 'general', 'analysis')
            requires_tools: Whether the task requires tool support
            
        Returns:
            Best model name or None if no suitable model found
        """
        try:
            self.logger.debug(f"Finding best model for task: {task_type}, tools: {requires_tools}")
            
            # Get all available models
            available_models = self.validator.get_available_models()
            if not available_models:
                self.logger.warning("No available models found")
                return None

            # Filter by tool support if required
            if requires_tools:
                tool_models = self.validator.get_tool_supporting_models()
                available_models = [m for m in available_models if m in tool_models]
                
                if not available_models:
                    self.logger.warning("No tool-supporting models available")
                    return self._get_best_model_fallback(task_type, requires_tools)

            # Score models based on task type and capabilities
            scored_models = []
            for model in available_models:
                score = self._score_model_for_task(model, task_type, requires_tools)
                if score > 0:
                    scored_models.append((model, score))

            if not scored_models:
                return self._get_best_model_fallback(task_type, requires_tools)

            # Sort by score (descending) and return the best
            scored_models.sort(key=lambda x: x[1], reverse=True)
            best_model = scored_models[0][0]
            
            self.logger.info(f"Selected model {best_model} for task {task_type} (score: {scored_models[0][1]:.2f})")
            return best_model

        except Exception as e:
            self.logger.error(f"Error selecting model for task {task_type}: {e}")
            return self._get_best_model_fallback(task_type, requires_tools)

    def get_alternative_model(self, original_model: str, requires_tools: bool = False) -> Optional[str]:
        """Get an alternative model to the original one.
        
        Args:
            original_model: Original model that needs to be replaced
            requires_tools: Whether the alternative must support tools
            
        Returns:
            Alternative model name or None
        """
        available_models = self.validator.get_available_models()
        
        # Remove the original model from consideration
        candidates = [m for m in available_models if m != original_model]
        
        if requires_tools:
            tool_models = self.validator.get_tool_supporting_models()
            candidates = [m for m in candidates if m in tool_models]

        if not candidates:
            return None

        # Score candidates and return the best
        scored_candidates = []
        for model in candidates:
            # Get capabilities of original model for comparison
            original_caps = self.validator.get_model_capabilities(original_model)
            original_size = self.validator.get_model_size(original_model)
            
            candidate_caps = self.validator.get_model_capabilities(model)
            candidate_size = self.validator.get_model_size(model)
            
            # Score based on capability overlap and size similarity
            capability_overlap = len(set(original_caps) & set(candidate_caps))
            size_diff = abs(original_size - candidate_size)
            
            # Prefer models with similar capabilities and size
            score = capability_overlap - (size_diff * 0.1)
            scored_candidates.append((model, score))

        if scored_candidates:
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            return scored_candidates[0][0]

        return None

    def get_default_model(self) -> str:
        """Get the default model for general use.
        
        Returns:
            Default model name
        """
        # Priority order for default models
        preferred_defaults = [
            "qwen2.5-coder:7b-instruct",
            "qwen2.5:7b", 
            "llama3.2:3b",
            "phi3:mini"
        ]
        
        available_models = self.validator.get_available_models()
        
        # Return first available preferred model
        for model in preferred_defaults:
            if model in available_models:
                return model
        
        # Fallback to any available model
        if available_models:
            return available_models[0]
        
        # Last resort fallback
        return "qwen2.5:7b"

    def _score_model_for_task(self, model_name: str, task_type: str, requires_tools: bool) -> float:
        """Score a model for a specific task.
        
        Args:
            model_name: Name of the model to score
            task_type: Type of task
            requires_tools: Whether tools are required
            
        Returns:
            Score for the model (higher is better)
        """
        score = 0.0
        
        # Get model properties
        capabilities = self.validator.get_model_capabilities(model_name)
        model_size = self.validator.get_model_size(model_name)
        supports_tools = self.validator.supports_tools(model_name)
        
        # Base score from capabilities matching task
        task_lower = task_type.lower()
        
        if task_lower in ["coding", "programming", "code"]:
            if "coding" in capabilities or "programming" in capabilities:
                score += 3.0
            if "coder" in model_name.lower():
                score += 2.0
        
        elif task_lower in ["analysis", "analytical", "reasoning"]:
            if "analysis" in capabilities or "reasoning" in capabilities:
                score += 3.0
            if model_size >= 7.0:  # Larger models better for analysis
                score += 1.0
        
        elif task_lower in ["general", "conversation", "chat"]:
            if "general" in capabilities or "conversation" in capabilities:
                score += 2.0
            if "instruct" in model_name.lower():
                score += 1.0
        
        elif task_lower in ["math", "mathematics", "calculation"]:
            if "mathematics" in capabilities or "math" in capabilities:
                score += 3.0
            if model_size >= 7.0:  # Larger models better for math
                score += 1.0
        
        # Tool support bonus/penalty
        if requires_tools:
            if supports_tools:
                score += 2.0
            else:
                score = 0.0  # Eliminate models that don't support tools when required
        
        # Size considerations (prefer models within size limit)
        if model_size <= self.max_model_size_b:
            score += 1.0
        else:
            score -= 2.0  # Penalty for oversized models
        
        # Performance heuristics based on model name
        if "instruct" in model_name.lower():
            score += 0.5  # Instruction-tuned models are generally better
        
        if "chat" in model_name.lower():
            score += 0.3  # Chat models are good for interactive tasks
        
        return max(0.0, score)

    def _get_best_model_fallback(self, task_type: str, requires_tools: bool = False) -> Optional[str]:
        """Fallback model selection when primary method fails.
        
        Args:
            task_type: Type of task
            requires_tools: Whether tools are required
            
        Returns:
            Fallback model name
        """
        self.logger.debug(f"Using fallback selection for task: {task_type}")
        
        # Task-specific fallbacks
        fallback_models = {
            "coding": ["qwen2.5-coder:7b-instruct", "deepseek-coder:6.7b", "llama3.2:3b"],
            "analysis": ["qwen2.5:7b", "llama3.2:3b", "phi3:mini"],
            "general": ["qwen2.5:7b", "llama3.2:3b", "phi3:mini"],
            "math": ["qwen2.5:7b", "llama3.2:3b"],
            "conversation": ["llama3.2:3b", "qwen2.5:7b", "phi3:mini"]
        }
        
        # Get fallback list for task type
        candidates = fallback_models.get(task_type.lower(), fallback_models["general"])
        
        # Check which candidates are available
        available_models = self.validator.get_available_models()
        
        for model in candidates:
            if model in available_models:
                # Check tool support if required
                if requires_tools and not self.validator.supports_tools(model):
                    continue
                return model
        
        # Ultimate fallback
        return self.get_default_model()

    def get_models_by_preference(self, task_type: str, requires_tools: bool = False) -> List[str]:
        """Get models ranked by preference for a task.
        
        Args:
            task_type: Type of task
            requires_tools: Whether tools are required
            
        Returns:
            List of models ranked by preference (best first)
        """
        available_models = self.validator.get_available_models()
        
        if requires_tools:
            tool_models = self.validator.get_tool_supporting_models()
            available_models = [m for m in available_models if m in tool_models]
        
        # Score all models
        scored_models = []
        for model in available_models:
            score = self._score_model_for_task(model, task_type, requires_tools)
            if score > 0:
                scored_models.append((model, score))
        
        # Sort by score (descending)
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        return [model for model, score in scored_models]
