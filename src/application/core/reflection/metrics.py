"""Performance metrics and calculation for reflection system."""

from dataclasses import dataclass, field
from typing import Dict


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
            "user_satisfaction": 0.1
        }

        # Normalize metrics to 0-1 scale
        normalized_accuracy = min(1.0, self.accuracy_score)
        
        # Speed score (inverse of time, normalized)
        normalized_speed = 1.0 / (1.0 + self.completion_time / 60.0)  # 1 minute baseline
        
        # Reliability score (success rate)
        total_operations = self.success_count + self.error_count
        normalized_reliability = (self.success_count / total_operations) if total_operations > 0 else 0.0
        
        # Confidence score
        normalized_confidence = min(1.0, self.confidence_avg)
        
        # User satisfaction score
        normalized_user_satisfaction = min(1.0, self.user_feedback_score)

        # Calculate weighted score
        overall_score = (
            weights["accuracy"] * normalized_accuracy +
            weights["speed"] * normalized_speed +
            weights["reliability"] * normalized_reliability +
            weights["confidence"] * normalized_confidence +
            weights["user_satisfaction"] * normalized_user_satisfaction
        )

        return min(1.0, max(0.0, overall_score))

    def update_from_execution(self, execution_time: float, success: bool, 
                            confidence: float = None, user_rating: float = None):
        """Update metrics from a single execution result.
        
        Args:
            execution_time: Time taken for execution
            success: Whether the execution was successful
            confidence: Confidence level (0.0-1.0)
            user_rating: User satisfaction rating (0.0-1.0)
        """
        # Update execution counts
        if success:
            self.success_count += 1
        else:
            self.error_count += 1

        # Update completion time (running average)
        total_operations = self.success_count + self.error_count
        if total_operations == 1:
            self.completion_time = execution_time
        else:
            self.completion_time = (
                (self.completion_time * (total_operations - 1) + execution_time) / total_operations
            )

        # Update confidence (running average)
        if confidence is not None:
            if total_operations == 1:
                self.confidence_avg = confidence
            else:
                self.confidence_avg = (
                    (self.confidence_avg * (total_operations - 1) + confidence) / total_operations
                )

        # Update user feedback (running average)
        if user_rating is not None:
            if total_operations == 1:
                self.user_feedback_score = user_rating
            else:
                self.user_feedback_score = (
                    (self.user_feedback_score * (total_operations - 1) + user_rating) / total_operations
                )

        # Update accuracy based on success rate
        self.accuracy_score = self.success_count / total_operations if total_operations > 0 else 0.0

    def get_summary(self) -> Dict[str, float]:
        """Get a summary of all metrics."""
        return {
            "overall_score": self.calculate_overall_score(),
            "accuracy_score": self.accuracy_score,
            "completion_time": self.completion_time,
            "error_count": self.error_count,
            "success_count": self.success_count,
            "confidence_avg": self.confidence_avg,
            "user_feedback_score": self.user_feedback_score,
            "total_operations": self.success_count + self.error_count
        }

    def compare_with(self, other: 'PerformanceMetrics') -> Dict[str, float]:
        """Compare this metrics with another PerformanceMetrics instance.
        
        Args:
            other: Another PerformanceMetrics instance
            
        Returns:
            Dictionary with comparison results (positive means this is better)
        """
        return {
            "overall_score_diff": self.calculate_overall_score() - other.calculate_overall_score(),
            "accuracy_diff": self.accuracy_score - other.accuracy_score,
            "speed_improvement": other.completion_time - self.completion_time,  # Lower is better
            "reliability_diff": (self.success_count / max(1, self.success_count + self.error_count)) - 
                              (other.success_count / max(1, other.success_count + other.error_count)),
            "confidence_diff": self.confidence_avg - other.confidence_avg,
            "satisfaction_diff": self.user_feedback_score - other.user_feedback_score
        }
