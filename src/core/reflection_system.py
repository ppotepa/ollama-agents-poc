"""Reflection System - Self-evaluation and automatic swap triggers for intelligent model switching."""
from __future__ import annotations

import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

from src.utils.enhanced_logging import get_logger
from src.core.context_manager import ContextManager, get_context_manager


class ConfidenceLevel(Enum):
    """Confidence levels for self-assessment."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ReflectionTrigger(Enum):
    """Triggers for reflection checkpoints."""
    STEP_COMPLETION = "step_completion"
    ERROR_ENCOUNTERED = "error_encountered"
    LOW_CONFIDENCE = "low_confidence"
    TIMEOUT = "timeout"
    USER_REQUEST = "user_request"
    AUTOMATIC = "automatic"


@dataclass
class PerformanceMetrics:
    """Performance metrics for a step or model."""
    accuracy_score: float = 0.0
    completion_time: float = 0.0
    error_count: int = 0
    success_count: int = 0
    confidence_avg: float = 0.0
    user_feedback_score: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    def calculate_overall_score(self) -> float:
        """Calculate an overall performance score."""
        # Weighted combination of metrics
        weights = {
            "accuracy": 0.3,
            "speed": 0.2,
            "reliability": 0.25,
            "confidence": 0.15,
            "feedback": 0.1
        }
        
        # Normalize speed (lower is better)
        speed_score = max(0, 1 - (self.completion_time / 300))  # 5 minutes baseline
        
        # Calculate reliability (success rate)
        total_attempts = self.error_count + self.success_count
        reliability_score = self.success_count / total_attempts if total_attempts > 0 else 0.5
        
        # Normalize other scores
        accuracy_score = min(1.0, max(0.0, self.accuracy_score))
        confidence_score = min(1.0, max(0.0, self.confidence_avg))
        feedback_score = min(1.0, max(0.0, self.user_feedback_score))
        
        overall_score = (
            weights["accuracy"] * accuracy_score +
            weights["speed"] * speed_score +
            weights["reliability"] * reliability_score +
            weights["confidence"] * confidence_score +
            weights["feedback"] * feedback_score
        )
        
        return min(1.0, max(0.0, overall_score))


@dataclass
class ReflectionResult:
    """Result of a reflection checkpoint."""
    session_id: str
    step_id: str
    current_model: str
    confidence_level: ConfidenceLevel
    performance_metrics: PerformanceMetrics
    should_swap: bool
    recommended_model: Optional[str]
    reasoning: str
    improvement_suggestions: List[str]
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "step_id": self.step_id,
            "current_model": self.current_model,
            "confidence_level": self.confidence_level.value,
            "should_swap": self.should_swap,
            "recommended_model": self.recommended_model,
            "reasoning": self.reasoning,
            "improvement_suggestions": self.improvement_suggestions,
            "timestamp": self.timestamp,
            "performance_metrics": {
                "overall_score": self.performance_metrics.calculate_overall_score(),
                "accuracy": self.performance_metrics.accuracy_score,
                "completion_time": self.performance_metrics.completion_time,
                "error_count": self.performance_metrics.error_count,
                "success_count": self.performance_metrics.success_count,
                "confidence_avg": self.performance_metrics.confidence_avg
            }
        }


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
        """Evaluate the performance of a completed step.
        
        Args:
            session_id: Session identifier
            step_id: Step identifier
            current_model: Current model being used
            step_result: Result of the step execution
            execution_time: Time taken to execute the step
            
        Returns:
            ReflectionResult with evaluation and recommendations
        """
        context_manager = get_context_manager()
        context = context_manager.get_context(session_id)
        
        # Analyze step result
        confidence = self._assess_confidence(step_result, context)
        metrics = self._calculate_metrics(step_result, execution_time, context)
        
        # Determine if model swap is needed
        should_swap, recommended_model, reasoning = self._evaluate_swap_need(
            current_model, confidence, metrics, context
        )
        
        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(
            current_model, metrics, context
        )
        
        result = ReflectionResult(
            session_id=session_id,
            step_id=step_id,
            current_model=current_model,
            confidence_level=confidence,
            performance_metrics=metrics,
            should_swap=should_swap,
            recommended_model=recommended_model,
            reasoning=reasoning,
            improvement_suggestions=suggestions
        )
        
        self.logger.info(f"Reflection checkpoint {self.checkpoint_id}: "
                        f"Confidence={confidence.value}, Swap={should_swap}, "
                        f"Model={recommended_model or current_model}")
        
        return result
    
    def _assess_confidence(self, step_result: Any, context: Any) -> ConfidenceLevel:
        """Assess confidence level based on step result."""
        confidence_score = 0.5  # Default medium confidence
        
        # Analyze result content
        if isinstance(step_result, str):
            result_text = step_result.lower()
            
            # High confidence indicators
            if any(phrase in result_text for phrase in [
                "successfully", "completed", "found", "identified", 
                "implemented", "resolved", "confirmed"
            ]):
                confidence_score += 0.2
            
            # Low confidence indicators
            if any(phrase in result_text for phrase in [
                "error", "failed", "unable", "not found", "unclear", 
                "uncertain", "might", "perhaps", "possibly"
            ]):
                confidence_score -= 0.3
            
            # Quality indicators
            if len(result_text) > 100:  # Detailed response
                confidence_score += 0.1
            
            if any(phrase in result_text for phrase in [
                "specifically", "precisely", "exactly", "detailed", "comprehensive"
            ]):
                confidence_score += 0.1
        
        # Analyze context history
        if context and hasattr(context, 'execution_history'):
            recent_errors = sum(1 for record in context.execution_history[-5:] 
                              if 'error' in record.get('result', '').lower())
            if recent_errors > 2:
                confidence_score -= 0.2
        
        # Convert to confidence level
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        if confidence_score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.6:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.4:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _calculate_metrics(self, step_result: Any, execution_time: float, 
                          context: Any) -> PerformanceMetrics:
        """Calculate performance metrics."""
        metrics = PerformanceMetrics()
        
        # Basic metrics
        metrics.completion_time = execution_time
        
        # Analyze result quality
        if isinstance(step_result, str):
            result_text = step_result
            
            # Accuracy indicators (simplified heuristic)
            accuracy_indicators = ["correct", "accurate", "precise", "exact"]
            error_indicators = ["error", "wrong", "incorrect", "failed"]
            
            accuracy_score = 0.5
            for indicator in accuracy_indicators:
                if indicator in result_text.lower():
                    accuracy_score += 0.1
            for indicator in error_indicators:
                if indicator in result_text.lower():
                    accuracy_score -= 0.2
            
            metrics.accuracy_score = max(0.0, min(1.0, accuracy_score))
            
            # Success/error counting
            if any(word in result_text.lower() for word in ["success", "complete", "done"]):
                metrics.success_count = 1
            elif any(word in result_text.lower() for word in ["error", "fail", "exception"]):
                metrics.error_count = 1
        
        # Analyze execution history from context
        if context and hasattr(context, 'execution_history'):
            recent_history = context.execution_history[-10:]  # Last 10 executions
            
            total_time = sum(record.get('duration', 0) for record in recent_history)
            avg_time = total_time / len(recent_history) if recent_history else 0
            
            # Speed score (faster than average is better)
            if avg_time > 0:
                speed_ratio = execution_time / avg_time
                metrics.resource_usage["speed_ratio"] = speed_ratio
        
        return metrics
    
    def _evaluate_swap_need(self, current_model: str, confidence: ConfidenceLevel,
                           metrics: PerformanceMetrics, context: Any) -> Tuple[bool, Optional[str], str]:
        """Evaluate if a model swap is needed.
        
        Returns:
            Tuple of (should_swap, recommended_model, reasoning)
        """
        should_swap = False
        recommended_model = None
        reasoning = "Performance is acceptable, continuing with current model."
        
        # Low confidence trigger
        if confidence in [ConfidenceLevel.VERY_LOW, ConfidenceLevel.LOW]:
            should_swap = True
            reasoning = f"Low confidence level ({confidence.value}) detected."
            recommended_model = self._recommend_alternative_model(current_model, "confidence")
        
        # High error rate trigger
        elif metrics.error_count > 0 and metrics.success_count == 0:
            should_swap = True
            reasoning = "High error rate detected without successful completions."
            recommended_model = self._recommend_alternative_model(current_model, "errors")
        
        # Performance trigger
        elif metrics.completion_time > 300:  # 5 minutes
            should_swap = True
            reasoning = "Execution time exceeds acceptable threshold."
            recommended_model = self._recommend_alternative_model(current_model, "performance")
        
        # Overall performance score trigger
        elif metrics.calculate_overall_score() < 0.3:
            should_swap = True
            reasoning = "Overall performance score is below acceptable threshold."
            recommended_model = self._recommend_alternative_model(current_model, "overall")
        
        # Context-based triggers
        elif context and hasattr(context, 'execution_history'):
            recent_failures = sum(1 for record in context.execution_history[-3:] 
                                if 'error' in record.get('result', '').lower())
            if recent_failures >= 2:
                should_swap = True
                reasoning = "Multiple recent failures detected."
                recommended_model = self._recommend_alternative_model(current_model, "failures")
        
        return should_swap, recommended_model, reasoning
    
    def _recommend_alternative_model(self, current_model: str, reason: str) -> str:
        """Recommend an alternative model based on the current situation.
        
        Args:
            current_model: Current model name
            reason: Reason for the recommendation (confidence, errors, performance, etc.)
            
        Returns:
            Recommended model name
        """
        # Model capability mapping (simplified)
        model_capabilities = {
            "qwen2.5:7b-instruct-q4_K_M": ["general", "coding", "analysis"],
            "gemma:7b-instruct-q4_K_M": ["creative", "reasoning", "explanation"],
            "llama3.2:latest": ["research", "detailed_analysis", "comprehensive"],
            "phi3:latest": ["quick_tasks", "simple_coding", "fast_response"],
            "codestral:latest": ["advanced_coding", "debugging", "code_review"],
            "deepseek-coder:6.7b-instruct-q4_K_M": ["coding", "technical", "implementation"]
        }
        
        # Get available models (excluding current)
        available_models = [model for model in model_capabilities.keys() 
                          if model != current_model]
        
        if not available_models:
            return current_model
        
        # Recommendation logic based on reason
        if reason == "confidence":
            # For confidence issues, try a more capable model
            if "phi3" in current_model:
                return "qwen2.5:7b-instruct-q4_K_M"
            elif "gemma" in current_model:
                return "llama3.2:latest"
            else:
                return "codestral:latest"
        
        elif reason == "errors":
            # For errors, try a different approach
            if "coding" in str(model_capabilities.get(current_model, [])):
                return "gemma:7b-instruct-q4_K_M"  # Different reasoning approach
            else:
                return "qwen2.5:7b-instruct-q4_K_M"  # General capability
        
        elif reason == "performance":
            # For performance issues, try a faster model
            return "phi3:latest"
        
        elif reason == "failures":
            # For repeated failures, try most capable model
            return "codestral:latest"
        
        else:
            # Default: try qwen as general-purpose
            return "qwen2.5:7b-instruct-q4_K_M"
    
    def _generate_improvement_suggestions(self, current_model: str, 
                                        metrics: PerformanceMetrics,
                                        context: Any) -> List[str]:
        """Generate suggestions for improvement."""
        suggestions = []
        
        # Performance-based suggestions
        if metrics.completion_time > 180:  # 3 minutes
            suggestions.append("Consider breaking down complex tasks into smaller steps")
            suggestions.append("Use more specific prompts to reduce processing time")
        
        if metrics.error_count > 0:
            suggestions.append("Add error handling and validation steps")
            suggestions.append("Verify input parameters before execution")
        
        if metrics.accuracy_score < 0.5:
            suggestions.append("Provide more detailed context and examples")
            suggestions.append("Use step-by-step reasoning approach")
        
        # Model-specific suggestions
        if "phi3" in current_model and metrics.completion_time > 120:
            suggestions.append("Consider using a more powerful model for complex tasks")
        
        if "codestral" in current_model and metrics.error_count > 0:
            suggestions.append("Verify code syntax and dependencies before execution")
        
        # Context-based suggestions
        if context and hasattr(context, 'execution_history'):
            if len(context.execution_history) > 5:
                suggestions.append("Consider consolidating context to improve focus")
            
            model_switches = len(set(record.get('model', '') for record in context.execution_history))
            if model_switches > 3:
                suggestions.append("Reduce frequent model switching for better continuity")
        
        return suggestions


class ReflectionSystem:
    """Manages reflection checkpoints and self-evaluation."""
    
    def __init__(self, context_manager: Optional[ContextManager] = None):
        """Initialize the reflection system.
        
        Args:
            context_manager: Context manager instance (optional)
        """
        self.context_manager = context_manager or get_context_manager()
        self.logger = get_logger()
        self.checkpoints: Dict[str, ReflectionCheckpoint] = {}
        self.reflection_history: List[ReflectionResult] = []
        
        # Configuration
        self.config = {
            "auto_reflection_interval": 300,  # 5 minutes
            "confidence_threshold": ConfidenceLevel.LOW,
            "max_checkpoint_history": 100,
            "enable_auto_swap": True
        }
    
    def create_checkpoint(self, session_id: str, step_id: str, 
                         trigger: ReflectionTrigger) -> str:
        """Create a new reflection checkpoint.
        
        Args:
            session_id: Session identifier
            step_id: Step identifier
            trigger: What triggered this checkpoint
            
        Returns:
            Checkpoint ID
        """
        checkpoint_id = f"{session_id}_{step_id}_{int(time.time())}"
        checkpoint = ReflectionCheckpoint(checkpoint_id, trigger)
        self.checkpoints[checkpoint_id] = checkpoint
        
        self.logger.debug(f"Created reflection checkpoint {checkpoint_id} "
                         f"for {trigger.value}")
        
        return checkpoint_id
    
    def evaluate_step(self, session_id: str, step_id: str, current_model: str,
                     step_result: Any, execution_time: float,
                     trigger: ReflectionTrigger = ReflectionTrigger.STEP_COMPLETION) -> ReflectionResult:
        """Evaluate a completed step and generate reflection result.
        
        Args:
            session_id: Session identifier
            step_id: Step identifier
            current_model: Current model being used
            step_result: Result of the step execution
            execution_time: Time taken to execute
            trigger: What triggered this evaluation
            
        Returns:
            ReflectionResult with evaluation and recommendations
        """
        checkpoint_id = self.create_checkpoint(session_id, step_id, trigger)
        checkpoint = self.checkpoints[checkpoint_id]
        
        result = checkpoint.evaluate_step_performance(
            session_id, step_id, current_model, step_result, execution_time
        )
        
        # Store in history
        self.reflection_history.append(result)
        
        # Maintain history size
        if len(self.reflection_history) > self.config["max_checkpoint_history"]:
            self.reflection_history = self.reflection_history[-self.config["max_checkpoint_history"]:]
        
        # Store in context
        self.context_manager.add_context_data(
            session_id, 
            f"reflection_{checkpoint_id}", 
            result.to_dict(),
            f"reflection_system_{checkpoint_id}"
        )
        
        self.logger.info(f"Reflection evaluation complete for step {step_id}: "
                        f"Confidence={result.confidence_level.value}, "
                        f"Swap recommended={result.should_swap}")
        
        return result
    
    def should_trigger_automatic_reflection(self, session_id: str) -> bool:
        """Check if automatic reflection should be triggered.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if automatic reflection should be triggered
        """
        context = self.context_manager.get_context(session_id)
        if not context:
            return False
        
        # Check time since last reflection
        last_reflection_time = 0
        for entry in context.entries.values():
            if entry.key.startswith("reflection_"):
                last_reflection_time = max(last_reflection_time, entry.timestamp)
        
        time_since_last = time.time() - last_reflection_time
        
        # Trigger if enough time has passed
        if time_since_last > self.config["auto_reflection_interval"]:
            return True
        
        # Trigger if recent errors
        recent_errors = sum(1 for record in context.execution_history[-3:] 
                          if 'error' in record.get('result', '').lower())
        if recent_errors >= 2:
            return True
        
        return False
    
    def get_model_performance_summary(self, session_id: str) -> Dict[str, Any]:
        """Get performance summary for all models used in a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Performance summary by model
        """
        session_reflections = [r for r in self.reflection_history 
                             if r.session_id == session_id]
        
        model_stats = {}
        
        for reflection in session_reflections:
            model = reflection.current_model
            if model not in model_stats:
                model_stats[model] = {
                    "total_steps": 0,
                    "success_rate": 0,
                    "avg_confidence": 0,
                    "avg_performance": 0,
                    "swap_recommendations": 0,
                    "confidence_distribution": {level.value: 0 for level in ConfidenceLevel}
                }
            
            stats = model_stats[model]
            stats["total_steps"] += 1
            stats["confidence_distribution"][reflection.confidence_level.value] += 1
            
            if reflection.should_swap:
                stats["swap_recommendations"] += 1
        
        # Calculate averages
        for model, stats in model_stats.items():
            if stats["total_steps"] > 0:
                stats["success_rate"] = 1 - (stats["swap_recommendations"] / stats["total_steps"])
                
                # Calculate weighted average confidence
                confidence_weights = {
                    ConfidenceLevel.VERY_LOW.value: 0.1,
                    ConfidenceLevel.LOW.value: 0.3,
                    ConfidenceLevel.MEDIUM.value: 0.5,
                    ConfidenceLevel.HIGH.value: 0.7,
                    ConfidenceLevel.VERY_HIGH.value: 0.9
                }
                
                weighted_confidence = sum(
                    count * confidence_weights[level] 
                    for level, count in stats["confidence_distribution"].items()
                )
                stats["avg_confidence"] = weighted_confidence / stats["total_steps"]
        
        return model_stats
    
    def get_reflection_insights(self, session_id: str) -> Dict[str, Any]:
        """Get insights from reflection analysis.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Insights and recommendations
        """
        model_performance = self.get_model_performance_summary(session_id)
        session_reflections = [r for r in self.reflection_history 
                             if r.session_id == session_id]
        
        insights = {
            "session_summary": {
                "total_reflections": len(session_reflections),
                "models_evaluated": len(model_performance),
                "swap_rate": sum(1 for r in session_reflections if r.should_swap) / len(session_reflections) if session_reflections else 0
            },
            "best_performing_model": None,
            "most_swapped_from": None,
            "common_issues": [],
            "recommendations": []
        }
        
        if model_performance:
            # Find best performing model
            best_model = max(model_performance.items(), 
                           key=lambda x: x[1]["success_rate"])
            insights["best_performing_model"] = {
                "model": best_model[0],
                "success_rate": best_model[1]["success_rate"],
                "avg_confidence": best_model[1]["avg_confidence"]
            }
            
            # Find most problematic model
            worst_model = min(model_performance.items(), 
                            key=lambda x: x[1]["success_rate"])
            insights["most_swapped_from"] = {
                "model": worst_model[0],
                "swap_rate": worst_model[1]["swap_recommendations"] / worst_model[1]["total_steps"]
            }
        
        # Analyze common issues
        all_suggestions = []
        for reflection in session_reflections:
            all_suggestions.extend(reflection.improvement_suggestions)
        
        # Count suggestion frequency
        suggestion_counts = {}
        for suggestion in all_suggestions:
            suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1
        
        # Get most common issues
        insights["common_issues"] = sorted(suggestion_counts.items(), 
                                         key=lambda x: x[1], reverse=True)[:5]
        
        # Generate recommendations
        if insights["session_summary"]["swap_rate"] > 0.5:
            insights["recommendations"].append("High swap rate detected - consider task decomposition")
        
        if len(model_performance) > 3:
            insights["recommendations"].append("Multiple models used - consider more consistent model selection")
        
        return insights


__all__ = [
    "ReflectionSystem", "ReflectionCheckpoint", "ReflectionResult", 
    "PerformanceMetrics", "ConfidenceLevel", "ReflectionTrigger"
]
