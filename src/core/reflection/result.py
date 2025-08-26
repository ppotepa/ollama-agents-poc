"""Reflection result and data structures."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .types import ConfidenceLevel
from .metrics import PerformanceMetrics


@dataclass
class ReflectionResult:
    """Result of a reflection checkpoint."""
    session_id: str
    step_id: str
    current_model: str
    confidence_level: ConfidenceLevel
    performance_metrics: PerformanceMetrics
    should_swap: bool
    recommended_model: str = None
    reasoning: str = ""
    improvement_suggestions: List[str] = field(default_factory=list)
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReflectionResult':
        """Create ReflectionResult from dictionary."""
        # Extract performance metrics
        perf_data = data.get("performance_metrics", {})
        metrics = PerformanceMetrics(
            accuracy_score=perf_data.get("accuracy", 0.0),
            completion_time=perf_data.get("completion_time", 0.0),
            error_count=perf_data.get("error_count", 0),
            success_count=perf_data.get("success_count", 0),
            confidence_avg=perf_data.get("confidence_avg", 0.0)
        )

        # Parse confidence level
        confidence_str = data.get("confidence_level", "medium")
        confidence = ConfidenceLevel(confidence_str)

        return cls(
            session_id=data["session_id"],
            step_id=data["step_id"],
            current_model=data["current_model"],
            confidence_level=confidence,
            performance_metrics=metrics,
            should_swap=data.get("should_swap", False),
            recommended_model=data.get("recommended_model"),
            reasoning=data.get("reasoning", ""),
            improvement_suggestions=data.get("improvement_suggestions", []),
            timestamp=data.get("timestamp", time.time())
        )

    def is_successful(self) -> bool:
        """Check if this reflection indicates a successful step."""
        return (
            self.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH] and
            not self.should_swap and
            self.performance_metrics.accuracy_score > 0.7
        )

    def get_severity_level(self) -> str:
        """Get severity level based on confidence and swap recommendation."""
        if self.should_swap and self.confidence_level == ConfidenceLevel.VERY_LOW:
            return "critical"
        elif self.should_swap and self.confidence_level == ConfidenceLevel.LOW:
            return "high"
        elif self.confidence_level == ConfidenceLevel.MEDIUM:
            return "medium"
        elif self.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]:
            return "low"
        else:
            return "medium"

    def get_action_recommendations(self) -> List[str]:
        """Get specific action recommendations based on reflection result."""
        actions = []
        
        if self.should_swap:
            if self.recommended_model:
                actions.append(f"Switch to {self.recommended_model}")
            else:
                actions.append("Consider switching to a different model")

        if self.confidence_level == ConfidenceLevel.VERY_LOW:
            actions.append("Review and revise approach")
            actions.append("Consider breaking down the task into smaller steps")

        if self.performance_metrics.error_count > 0:
            actions.append("Debug and fix errors before proceeding")

        if self.performance_metrics.completion_time > 300:  # 5 minutes
            actions.append("Optimize for faster execution")

        # Add improvement suggestions
        actions.extend(self.improvement_suggestions)

        return list(set(actions))  # Remove duplicates
