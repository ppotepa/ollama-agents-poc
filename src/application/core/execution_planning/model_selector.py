"""Model Selector - Selects optimal models for execution steps."""

from typing import Any

from src.core.agent_resolver import ModelCandidate
from src.core.model_capability_checker import get_capability_checker
from src.core.task_decomposer import Subtask, TaskType
from src.utils.enhanced_logging import get_logger


class ModelSelector:
    """Selects optimal models for execution steps based on capabilities and requirements."""

    def __init__(self, agent_resolver):
        """Initialize the model selector.

        Args:
            agent_resolver: Agent resolver for model recommendations
        """
        self.logger = get_logger()
        self.agent_resolver = agent_resolver
        self.capability_checker = get_capability_checker()

    def select_optimal_model(self, subtask: Subtask,
                            model_candidates: list[ModelCandidate]) -> tuple[str, float]:
        """Select the optimal model for a subtask from candidates.

        Args:
            subtask: The subtask requiring model assignment
            model_candidates: Available model candidates

        Returns:
            Tuple of (selected_model_name, confidence_score)
        """
        if not model_candidates:
            self.logger.warning(f"No model candidates for subtask {subtask.id}")
            return "llama3.2:3b", 0.1  # Fallback model

        # Score each candidate based on subtask requirements
        scored_candidates = []
        
        for candidate in model_candidates:
            score = self._calculate_model_score(subtask, candidate)
            scored_candidates.append((candidate, score))

        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        best_candidate, best_score = scored_candidates[0]
        
        self.logger.debug(f"Selected {best_candidate.model_name} for {subtask.id} "
                         f"(score: {best_score:.3f})")
        
        return best_candidate.model_name, best_score

    def _calculate_model_score(self, subtask: Subtask, candidate: ModelCandidate) -> float:
        """Calculate score for a model candidate based on subtask requirements.

        Args:
            subtask: The subtask to score for
            candidate: The model candidate to score

        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.0
        
        # Base compatibility score from agent resolver
        score += candidate.compatibility_score * 0.4

        # Task type compatibility
        task_type_score = self._get_task_type_compatibility(subtask.task_type, candidate)
        score += task_type_score * 0.25

        # Capability requirements matching
        capability_score = self._get_capability_match_score(subtask, candidate)
        score += capability_score * 0.2

        # Model size vs complexity matching
        complexity_score = self._get_complexity_match_score(subtask, candidate)
        score += complexity_score * 0.1

        # Tool/function calling requirements
        tool_score = self._get_tool_capability_score(subtask, candidate)
        score += tool_score * 0.05

        return min(score, 1.0)  # Cap at 1.0

    def _get_task_type_compatibility(self, task_type: TaskType, candidate: ModelCandidate) -> float:
        """Get task type compatibility score.

        Args:
            task_type: Type of task
            candidate: Model candidate

        Returns:
            Compatibility score 0.0-1.0
        """
        # Task type preferences for different model types
        task_preferences = {
            TaskType.CODE_ANALYSIS: {
                "code": 0.9, "programming": 0.8, "analysis": 0.7
            },
            TaskType.CODE_GENERATION: {
                "code": 1.0, "programming": 0.9, "generation": 0.8
            },
            TaskType.DEBUGGING: {
                "code": 0.9, "debugging": 1.0, "analysis": 0.7
            },
            TaskType.TESTING: {
                "testing": 1.0, "code": 0.8, "qa": 0.7
            },
            TaskType.DOCUMENTATION: {
                "documentation": 1.0, "writing": 0.8, "text": 0.7
            },
            TaskType.FILE_OPERATIONS: {
                "tools": 0.9, "system": 0.8, "automation": 0.7
            },
            TaskType.DATA_PROCESSING: {
                "data": 1.0, "analysis": 0.8, "processing": 0.7
            },
            TaskType.RESEARCH: {
                "research": 1.0, "analysis": 0.8, "reasoning": 0.7
            },
            TaskType.GENERAL_QA: {
                "qa": 0.8, "general": 0.7, "reasoning": 0.6
            }
        }

        if task_type not in task_preferences:
            return 0.5  # Neutral score

        preferences = task_preferences[task_type]
        model_name_lower = candidate.model_name.lower()
        
        # Check model name for task-relevant keywords
        max_score = 0.0
        for keyword, score in preferences.items():
            if keyword in model_name_lower:
                max_score = max(max_score, score)

        # Default score based on model characteristics
        if max_score == 0.0:
            if "code" in model_name_lower and task_type in [TaskType.CODE_ANALYSIS, TaskType.CODE_GENERATION, TaskType.DEBUGGING]:
                max_score = 0.7
            elif "chat" in model_name_lower or "instruct" in model_name_lower:
                max_score = 0.6
            else:
                max_score = 0.5

        return max_score

    def _get_capability_match_score(self, subtask: Subtask, candidate: ModelCandidate) -> float:
        """Get capability matching score.

        Args:
            subtask: Subtask with required capabilities
            candidate: Model candidate

        Returns:
            Capability match score 0.0-1.0
        """
        if not subtask.required_capabilities:
            return 0.8  # No specific requirements

        # Check model capabilities against requirements
        model_capabilities = self.capability_checker.get_model_capabilities(candidate.model_name)
        
        matched_capabilities = 0
        for required_cap in subtask.required_capabilities:
            if required_cap in model_capabilities:
                matched_capabilities += 1

        if len(subtask.required_capabilities) == 0:
            return 0.8
        
        match_ratio = matched_capabilities / len(subtask.required_capabilities)
        return match_ratio

    def _get_complexity_match_score(self, subtask: Subtask, candidate: ModelCandidate) -> float:
        """Get complexity matching score.

        Args:
            subtask: Subtask with complexity estimate
            candidate: Model candidate

        Returns:
            Complexity match score 0.0-1.0
        """
        complexity = subtask.estimated_complexity
        
        # Estimate model capability based on size/name
        model_capability = self._estimate_model_capability(candidate)
        
        # Perfect match gets highest score
        complexity_diff = abs(complexity - model_capability)
        
        # Score decreases with difference
        if complexity_diff < 0.1:
            return 1.0
        elif complexity_diff < 0.3:
            return 0.8
        elif complexity_diff < 0.5:
            return 0.6
        else:
            return 0.4

    def _estimate_model_capability(self, candidate: ModelCandidate) -> float:
        """Estimate model capability level based on characteristics.

        Args:
            candidate: Model candidate

        Returns:
            Estimated capability 0.0-1.0
        """
        model_name = candidate.model_name.lower()
        
        # Size-based estimates
        if "1b" in model_name or "1.1b" in model_name:
            return 0.2
        elif "3b" in model_name:
            return 0.4
        elif "7b" in model_name or "8b" in model_name:
            return 0.6
        elif "13b" in model_name or "14b" in model_name:
            return 0.8
        elif "30b" in model_name or "34b" in model_name or "70b" in model_name:
            return 1.0
        
        # Name-based estimates
        if "llama" in model_name:
            if "3.2" in model_name or "3.1" in model_name:
                return 0.7
            elif "3" in model_name:
                return 0.6
            elif "2" in model_name:
                return 0.5
        elif "mistral" in model_name or "mixtral" in model_name:
            return 0.7
        elif "qwen" in model_name or "deepseek" in model_name:
            return 0.8
        
        return 0.5  # Default estimate

    def _get_tool_capability_score(self, subtask: Subtask, candidate: ModelCandidate) -> float:
        """Get tool/function calling capability score.

        Args:
            subtask: Subtask that may require tools
            candidate: Model candidate

        Returns:
            Tool capability score 0.0-1.0
        """
        # Check if subtask requires tools
        requires_tools = self._subtask_requires_tools(subtask)
        
        if not requires_tools:
            return 1.0  # No tool requirements
        
        # Check if model supports function calling
        model_name = candidate.model_name.lower()
        
        # Models known to support function calling
        function_capable_models = [
            "llama3.2", "llama3.1", "mistral", "qwen", "deepseek"
        ]
        
        for capable_model in function_capable_models:
            if capable_model in model_name:
                return 1.0
        
        # Default assumption - most modern models support basic tool usage
        return 0.7

    def _subtask_requires_tools(self, subtask: Subtask) -> bool:
        """Check if subtask requires tool/function calling capabilities.

        Args:
            subtask: Subtask to check

        Returns:
            True if tools are required
        """
        tool_indicators = [
            "file", "directory", "create", "write", "read", "execute",
            "run", "install", "build", "compile", "test", "debug",
            "search", "find", "analyze", "process"
        ]

        description_lower = subtask.description.lower()
        return any(indicator in description_lower for indicator in tool_indicators)
