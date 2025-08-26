"""Individual reflection checkpoint for evaluation."""

import time
from typing import Any, List, Optional

from src.core.context_manager import ContextManager, get_context_manager
from src.utils.enhanced_logging import get_logger

from .types import ConfidenceLevel, ReflectionTrigger
from .metrics import PerformanceMetrics
from .result import ReflectionResult


class ReflectionCheckpoint:
    """Individual reflection checkpoint for evaluation."""

    def __init__(self, checkpoint_id: str, trigger: ReflectionTrigger):
        """Initialize reflection checkpoint.

        Args:
            checkpoint_id: Unique identifier for this checkpoint
            trigger: What triggered this reflection
        """
        self.checkpoint_id = checkpoint_id
        self.trigger = trigger
        self.timestamp = time.time()
        self.logger = get_logger()

    def evaluate_step_performance(self, session_id: str, step_id: str,
                                current_model: str, step_result: Any,
                                execution_time: float) -> ReflectionResult:
        """Evaluate performance of a completed step.

        Args:
            session_id: Session identifier
            step_id: Step identifier  
            current_model: Model that executed the step
            step_result: Result of step execution
            execution_time: Time taken to execute step

        Returns:
            ReflectionResult with evaluation and recommendations
        """
        try:
            self.logger.debug(f"Evaluating step performance for {step_id}")

            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(
                step_result, execution_time, current_model
            )

            # Determine confidence level
            confidence = self._determine_confidence_level(step_result, metrics)

            # Check if model swap is recommended
            should_swap, recommended_model = self._should_recommend_swap(
                current_model, metrics, confidence
            )

            # Generate improvement suggestions
            suggestions = self._generate_improvement_suggestions(
                current_model, metrics, step_result
            )

            # Create reasoning
            reasoning = self._generate_reasoning(
                confidence, should_swap, metrics, recommended_model
            )

            return ReflectionResult(
                session_id=session_id,
                step_id=step_id,
                current_model=current_model,
                confidence_level=confidence,
                performance_metrics=metrics,
                should_swap=should_swap,
                recommended_model=recommended_model,
                reasoning=reasoning,
                improvement_suggestions=suggestions,
                timestamp=self.timestamp
            )

        except Exception as e:
            self.logger.error(f"Error evaluating step performance: {e}")
            # Return a conservative evaluation
            return ReflectionResult(
                session_id=session_id,
                step_id=step_id,
                current_model=current_model,
                confidence_level=ConfidenceLevel.LOW,
                performance_metrics=PerformanceMetrics(),
                should_swap=True,
                reasoning=f"Evaluation failed: {str(e)}",
                improvement_suggestions=["Review step execution for errors"]
            )

    def _calculate_performance_metrics(self, step_result: Any, execution_time: float,
                                     current_model: str) -> PerformanceMetrics:
        """Calculate performance metrics for the step."""
        metrics = PerformanceMetrics()
        
        # Basic metrics
        metrics.completion_time = execution_time
        
        # Determine success based on result
        if step_result is None:
            metrics.error_count = 1
            metrics.accuracy_score = 0.0
        elif isinstance(step_result, str):
            if "error" in step_result.lower() or "failed" in step_result.lower():
                metrics.error_count = 1
                metrics.accuracy_score = 0.3
            else:
                metrics.success_count = 1
                metrics.accuracy_score = 0.8
        else:
            metrics.success_count = 1
            metrics.accuracy_score = 0.9

        # Estimate confidence based on execution time and result quality
        if execution_time < 5.0 and metrics.success_count > 0:
            metrics.confidence_avg = 0.9
        elif execution_time < 30.0 and metrics.success_count > 0:
            metrics.confidence_avg = 0.7
        elif metrics.success_count > 0:
            metrics.confidence_avg = 0.5
        else:
            metrics.confidence_avg = 0.2

        return metrics

    def _determine_confidence_level(self, step_result: Any, 
                                  metrics: PerformanceMetrics) -> ConfidenceLevel:
        """Determine confidence level based on step result and metrics."""
        if step_result is None:
            return ConfidenceLevel.VERY_LOW

        if isinstance(step_result, str):
            if "error" in step_result.lower() or "failed" in step_result.lower():
                return ConfidenceLevel.LOW
            elif len(step_result) < 10:
                return ConfidenceLevel.MEDIUM
            else:
                return ConfidenceLevel.HIGH

        # For non-string results, use metrics
        if metrics.accuracy_score > 0.8 and metrics.completion_time < 10.0:
            return ConfidenceLevel.VERY_HIGH
        elif metrics.accuracy_score > 0.6:
            return ConfidenceLevel.HIGH
        elif metrics.accuracy_score > 0.4:
            return ConfidenceLevel.MEDIUM
        elif metrics.accuracy_score > 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _should_recommend_swap(self, current_model: str, metrics: PerformanceMetrics,
                             confidence: ConfidenceLevel) -> tuple[bool, Optional[str]]:
        """Determine if model swap should be recommended."""
        # Don't swap if performing well
        if confidence in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]:
            return False, None

        # Recommend swap for poor performance
        if confidence == ConfidenceLevel.VERY_LOW or metrics.error_count > 0:
            recommended = self._get_alternative_model(current_model)
            return True, recommended

        # Consider swap for mediocre performance with slow execution
        if confidence == ConfidenceLevel.LOW and metrics.completion_time > 60.0:
            recommended = self._get_faster_model(current_model)
            return True, recommended

        return False, None

    def _get_alternative_model(self, current_model: str) -> str:
        """Get an alternative model recommendation."""
        model_alternatives = {
            "llama3.2:1b": "qwen2.5:7b",
            "llama3.2:3b": "qwen2.5-coder:7b-instruct",
            "qwen2.5:7b": "llama3.2:3b",
            "qwen2.5-coder:7b-instruct": "deepseek-coder:6.7b",
            "phi3:mini": "qwen2.5:7b"
        }
        
        return model_alternatives.get(current_model, "qwen2.5:7b")

    def _get_faster_model(self, current_model: str) -> str:
        """Get a faster model recommendation."""
        faster_models = {
            "qwen2.5:7b": "phi3:mini",
            "qwen2.5-coder:7b-instruct": "llama3.2:3b",
            "deepseek-coder:6.7b": "qwen2.5-coder:7b-instruct",
            "llama3.2:3b": "llama3.2:1b"
        }
        
        return faster_models.get(current_model, "phi3:mini")

    def _generate_improvement_suggestions(self, current_model: str, 
                                        metrics: PerformanceMetrics,
                                        step_result: Any) -> List[str]:
        """Generate improvement suggestions based on evaluation."""
        suggestions = []

        # Performance-based suggestions
        if metrics.completion_time > 120.0:  # 2 minutes
            suggestions.append("Consider using a faster model for time-sensitive tasks")
        
        if metrics.accuracy_score < 0.5:
            suggestions.append("Consider using a more powerful model for complex tasks")

        if "coder" in current_model and metrics.error_count > 0:
            suggestions.append("Verify code syntax and dependencies before execution")

        # Result-based suggestions
        if isinstance(step_result, str):
            if "timeout" in step_result.lower():
                suggestions.append("Increase timeout limits or optimize query complexity")
            elif "memory" in step_result.lower():
                suggestions.append("Consider using a smaller model or optimize memory usage")

        # Model-specific suggestions
        if "phi3" in current_model and metrics.completion_time > 60.0:
            suggestions.append("Phi3 models work best with simpler, focused queries")
        
        if "llama" in current_model and metrics.error_count > 0:
            suggestions.append("Llama models benefit from clear, structured prompts")

        return suggestions

    def _generate_reasoning(self, confidence: ConfidenceLevel, should_swap: bool,
                          metrics: PerformanceMetrics, recommended_model: Optional[str]) -> str:
        """Generate reasoning for the reflection result."""
        parts = []
        
        parts.append(f"Confidence assessment: {confidence.value}")
        
        if metrics.success_count > 0:
            parts.append(f"Step completed successfully in {metrics.completion_time:.1f}s")
        elif metrics.error_count > 0:
            parts.append(f"Step failed after {metrics.completion_time:.1f}s")
        
        if should_swap:
            if recommended_model:
                parts.append(f"Recommending switch to {recommended_model} for better performance")
            else:
                parts.append("Model change recommended due to performance issues")
        else:
            parts.append("Current model performing adequately")

        return ". ".join(parts)
