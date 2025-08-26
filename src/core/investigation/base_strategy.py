"""Base investigation strategy interface and common functionality."""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .types import InvestigationPlan, InvestigationStrategy, InvestigationStep


class BaseInvestigationStrategy(ABC):
    """Base class for all investigation strategies."""

    def __init__(self, strategy_type: InvestigationStrategy):
        """Initialize the strategy."""
        self.strategy_type = strategy_type
        self.created_at = time.time()
        self.usage_stats = {
            "plans_created": 0,
            "adaptations_made": 0,
            "total_steps_generated": 0
        }

    @abstractmethod
    def create_investigation_plan(self, query: str, context: Dict[str, Any]) -> InvestigationPlan:
        """Create an investigation plan for the given query and context.
        
        Args:
            query: The investigation query
            context: Available context information
            
        Returns:
            InvestigationPlan with ordered steps
        """
        pass

    def adapt_plan(self, plan: InvestigationPlan, 
                   completed_steps: List[str], 
                   step_results: Dict[str, Any],
                   new_context: Dict[str, Any]) -> InvestigationPlan:
        """Adapt an existing plan based on results and new context.
        
        Args:
            plan: Original investigation plan
            completed_steps: List of completed step IDs
            step_results: Results from completed steps
            new_context: Updated context information
            
        Returns:
            Adapted investigation plan
        """
        self.usage_stats["adaptations_made"] += 1
        
        # Default implementation: remove completed steps and adjust remaining
        remaining_steps = [
            step for step in plan.steps 
            if step.step_id not in completed_steps
        ]
        
        # Recalculate total duration
        total_duration = sum(step.estimated_duration for step in remaining_steps)
        
        return InvestigationPlan(
            investigation_id=f"{plan.investigation_id}_adapted_{int(time.time())}",
            query=plan.query,
            strategy=plan.strategy,
            steps=remaining_steps,
            total_estimated_duration=total_duration,
            success_criteria=plan.success_criteria,
            fallback_strategies=plan.fallback_strategies,
            created_at=time.time()
        )

    def _generate_step_id(self, base_name: str, index: int) -> str:
        """Generate a unique step ID."""
        return f"{self.strategy_type.value}_{base_name}_{index}_{int(time.time())}"

    def _estimate_duration(self, step_type: str, complexity: str = "medium") -> float:
        """Estimate duration for a step based on type and complexity.
        
        Args:
            step_type: Type of investigation step
            complexity: Complexity level (low, medium, high)
            
        Returns:
            Estimated duration in seconds
        """
        base_durations = {
            "file_analysis": 30.0,
            "code_review": 60.0,
            "structure_analysis": 45.0,
            "dependency_check": 20.0,
            "implementation_review": 90.0,
            "documentation_review": 25.0,
            "test_analysis": 40.0,
            "configuration_check": 15.0,
            "performance_analysis": 120.0,
            "security_review": 180.0
        }
        
        complexity_multipliers = {
            "low": 0.7,
            "medium": 1.0,
            "high": 1.5,
            "very_high": 2.0
        }
        
        base_duration = base_durations.get(step_type, 60.0)
        multiplier = complexity_multipliers.get(complexity, 1.0)
        
        return base_duration * multiplier

    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get statistics about this strategy's usage."""
        return {
            "strategy_type": self.strategy_type.value,
            "created_at": self.created_at,
            "uptime": time.time() - self.created_at,
            "usage_stats": self.usage_stats.copy()
        }

    def _analyze_query_complexity(self, query: str) -> str:
        """Analyze query complexity to inform step generation.
        
        Args:
            query: The investigation query
            
        Returns:
            Complexity level: low, medium, high, very_high
        """
        query_lower = query.lower()
        
        # Count complexity indicators
        complexity_indicators = {
            "high": ["analyze", "investigate", "comprehensive", "detailed", "thorough"],
            "medium": ["check", "review", "examine", "look", "find"],
            "low": ["list", "show", "display", "get", "simple"]
        }
        
        scores = {"high": 0, "medium": 0, "low": 0}
        
        for level, indicators in complexity_indicators.items():
            for indicator in indicators:
                if indicator in query_lower:
                    scores[level] += 1
        
        # Determine complexity based on indicators and query length
        if scores["high"] >= 2 or len(query.split()) > 15:
            return "very_high"
        elif scores["high"] >= 1 or scores["medium"] >= 3:
            return "high"
        elif scores["medium"] >= 1 or len(query.split()) > 8:
            return "medium"
        else:
            return "low"

    def _extract_query_keywords(self, query: str) -> List[str]:
        """Extract relevant keywords from the query.
        
        Args:
            query: The investigation query
            
        Returns:
            List of extracted keywords
        """
        import re
        
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
            "with", "by", "from", "up", "about", "into", "through", "during", "before",
            "after", "above", "below", "between", "among", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "can", "what", "where",
            "when", "why", "how", "which", "who", "whom", "whose"
        }
        
        # Extract words (alphanumeric + underscore)
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', query.lower())
        
        # Filter out stop words and short words
        keywords = [
            word for word in words 
            if word not in stop_words and len(word) > 2
        ]
        
        return keywords

    def _categorize_query_intent(self, query: str) -> str:
        """Categorize the intent of the query.
        
        Args:
            query: The investigation query
            
        Returns:
            Intent category
        """
        query_lower = query.lower()
        
        intent_patterns = {
            "code_analysis": ["code", "function", "class", "method", "algorithm", "implementation"],
            "file_exploration": ["file", "directory", "folder", "structure", "organization"],
            "documentation": ["documentation", "readme", "guide", "manual", "help", "docs"],
            "configuration": ["config", "configuration", "settings", "environment", "setup"],
            "testing": ["test", "testing", "unit", "integration", "coverage", "spec"],
            "debugging": ["debug", "error", "bug", "issue", "problem", "fix", "troubleshoot"],
            "performance": ["performance", "speed", "optimization", "benchmark", "profile"],
            "security": ["security", "vulnerability", "auth", "permission", "encrypt"],
            "dependency": ["dependency", "import", "require", "package", "library", "module"],
            "general": []  # Default category
        }
        
        # Count matches for each intent
        intent_scores = {}
        for intent, patterns in intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Return the intent with the highest score, or "general" if no matches
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        else:
            return "general"

    def _prioritize_models_for_intent(self, intent: str) -> List[str]:
        """Get prioritized model list for a specific intent.
        
        Args:
            intent: Query intent category
            
        Returns:
            List of model names prioritized for this intent
        """
        model_preferences = {
            "code_analysis": ["deepseek-coder-v2", "codellama", "qwen2.5-coder", "starcoder2"],
            "file_exploration": ["llama3.1", "qwen2.5", "gemma2", "mistral"],
            "documentation": ["llama3.1", "gemma2", "qwen2.5", "mistral"],
            "configuration": ["llama3.1", "qwen2.5", "deepseek-coder-v2"],
            "testing": ["deepseek-coder-v2", "codellama", "llama3.1"],
            "debugging": ["deepseek-coder-v2", "codellama", "qwen2.5-coder"],
            "performance": ["deepseek-coder-v2", "llama3.1", "qwen2.5"],
            "security": ["llama3.1", "qwen2.5", "deepseek-coder-v2"],
            "dependency": ["deepseek-coder-v2", "llama3.1", "qwen2.5"],
            "general": ["llama3.1", "qwen2.5", "gemma2", "mistral"]
        }
        
        return model_preferences.get(intent, model_preferences["general"])
