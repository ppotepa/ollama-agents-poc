"""Analytics and insights for the reflection system."""

import time
from typing import Any, Dict, List

from .types import ConfidenceLevel
from .result import ReflectionResult


class ReflectionAnalytics:
    """Provides analytics and insights from reflection data."""

    def __init__(self, reflection_history: List[ReflectionResult]):
        """Initialize analytics with reflection history.
        
        Args:
            reflection_history: List of reflection results to analyze
        """
        self.reflection_history = reflection_history

    def get_model_performance_summary(self, session_id: str = None) -> Dict[str, Any]:
        """Get performance summary for all models used in session(s).

        Args:
            session_id: Optional session identifier to filter by

        Returns:
            Performance summary by model
        """
        reflections = self.reflection_history
        if session_id:
            reflections = [r for r in reflections if r.session_id == session_id]

        model_stats = {}

        for reflection in reflections:
            model = reflection.current_model
            if model not in model_stats:
                model_stats[model] = {
                    "total_steps": 0,
                    "success_rate": 0,
                    "avg_confidence": 0,
                    "avg_performance": 0,
                    "swap_recommendations": 0,
                    "confidence_distribution": {level.value: 0 for level in ConfidenceLevel},
                    "avg_execution_time": 0,
                    "total_execution_time": 0
                }

            stats = model_stats[model]
            stats["total_steps"] += 1
            stats["confidence_distribution"][reflection.confidence_level.value] += 1
            stats["total_execution_time"] += reflection.performance_metrics.completion_time

            if reflection.should_swap:
                stats["swap_recommendations"] += 1

        # Calculate averages
        for model, stats in model_stats.items():
            if stats["total_steps"] > 0:
                stats["success_rate"] = 1 - (stats["swap_recommendations"] / stats["total_steps"])
                stats["avg_execution_time"] = stats["total_execution_time"] / stats["total_steps"]

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

    def get_reflection_insights(self, session_id: str = None) -> Dict[str, Any]:
        """Get insights from reflection analysis.

        Args:
            session_id: Optional session identifier to filter by

        Returns:
            Insights and recommendations
        """
        reflections = self.reflection_history
        if session_id:
            reflections = [r for r in reflections if r.session_id == session_id]

        if not reflections:
            return {"error": "No reflection data available"}

        model_performance = self.get_model_performance_summary(session_id)

        insights = {
            "session_summary": {
                "total_reflections": len(reflections),
                "models_evaluated": len(model_performance),
                "swap_rate": sum(1 for r in reflections if r.should_swap) / len(reflections),
                "avg_confidence": self._calculate_average_confidence(reflections),
                "time_period": {
                    "start": min(r.timestamp for r in reflections),
                    "end": max(r.timestamp for r in reflections),
                    "duration_hours": (max(r.timestamp for r in reflections) - 
                                     min(r.timestamp for r in reflections)) / 3600
                }
            },
            "best_performing_model": None,
            "most_problematic_model": None,
            "common_issues": [],
            "recommendations": [],
            "performance_trends": self._analyze_performance_trends(reflections)
        }

        if model_performance:
            # Find best performing model
            best_model = max(model_performance.items(),
                           key=lambda x: x[1]["success_rate"])
            insights["best_performing_model"] = {
                "model": best_model[0],
                "success_rate": best_model[1]["success_rate"],
                "avg_confidence": best_model[1]["avg_confidence"],
                "avg_execution_time": best_model[1]["avg_execution_time"]
            }

            # Find most problematic model
            worst_model = min(model_performance.items(),
                            key=lambda x: x[1]["success_rate"])
            insights["most_problematic_model"] = {
                "model": worst_model[0],
                "swap_rate": worst_model[1]["swap_recommendations"] / worst_model[1]["total_steps"],
                "avg_confidence": worst_model[1]["avg_confidence"]
            }

        # Analyze common issues
        insights["common_issues"] = self._analyze_common_issues(reflections)

        # Generate recommendations
        insights["recommendations"] = self._generate_recommendations(insights, model_performance)

        return insights

    def _calculate_average_confidence(self, reflections: List[ReflectionResult]) -> float:
        """Calculate average confidence across reflections."""
        confidence_weights = {
            ConfidenceLevel.VERY_LOW: 0.1,
            ConfidenceLevel.LOW: 0.3,
            ConfidenceLevel.MEDIUM: 0.5,
            ConfidenceLevel.HIGH: 0.7,
            ConfidenceLevel.VERY_HIGH: 0.9
        }
        
        total_weight = sum(confidence_weights[r.confidence_level] for r in reflections)
        return total_weight / len(reflections) if reflections else 0.0

    def _analyze_performance_trends(self, reflections: List[ReflectionResult]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if len(reflections) < 2:
            return {"insufficient_data": True}

        # Sort by timestamp
        sorted_reflections = sorted(reflections, key=lambda r: r.timestamp)
        
        # Split into first and second half
        mid_point = len(sorted_reflections) // 2
        first_half = sorted_reflections[:mid_point]
        second_half = sorted_reflections[mid_point:]

        first_avg_confidence = self._calculate_average_confidence(first_half)
        second_avg_confidence = self._calculate_average_confidence(second_half)

        first_swap_rate = sum(1 for r in first_half if r.should_swap) / len(first_half)
        second_swap_rate = sum(1 for r in second_half if r.should_swap) / len(second_half)

        return {
            "confidence_trend": {
                "early_average": first_avg_confidence,
                "recent_average": second_avg_confidence,
                "improvement": second_avg_confidence - first_avg_confidence
            },
            "swap_rate_trend": {
                "early_rate": first_swap_rate,
                "recent_rate": second_swap_rate,
                "change": second_swap_rate - first_swap_rate
            },
            "overall_direction": "improving" if second_avg_confidence > first_avg_confidence else "declining"
        }

    def _analyze_common_issues(self, reflections: List[ReflectionResult]) -> List[Dict[str, Any]]:
        """Analyze most common issues from improvement suggestions."""
        all_suggestions = []
        for reflection in reflections:
            all_suggestions.extend(reflection.improvement_suggestions)

        # Count suggestion frequency
        suggestion_counts = {}
        for suggestion in all_suggestions:
            suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1

        # Get most common issues
        common_issues = sorted(suggestion_counts.items(),
                             key=lambda x: x[1], reverse=True)[:5]

        return [{"issue": issue, "frequency": count, "percentage": count / len(reflections) * 100}
                for issue, count in common_issues]

    def _generate_recommendations(self, insights: Dict[str, Any], 
                                model_performance: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on insights."""
        recommendations = []
        summary = insights["session_summary"]

        # High swap rate recommendations
        if summary["swap_rate"] > 0.5:
            recommendations.append("High swap rate detected - consider task decomposition or model selection optimization")

        # Model diversity recommendations  
        if len(model_performance) > 3:
            recommendations.append("Multiple models used - consider more consistent model selection strategy")

        # Confidence recommendations
        if summary["avg_confidence"] < 0.4:
            recommendations.append("Low overall confidence - review task complexity and model capabilities")

        # Performance trend recommendations
        trends = insights.get("performance_trends", {})
        if trends.get("overall_direction") == "declining":
            recommendations.append("Performance declining over time - consider resetting context or changing approach")

        # Best model recommendations
        best_model = insights.get("best_performing_model")
        if best_model and best_model["success_rate"] > 0.8:
            recommendations.append(f"Consider using {best_model['model']} more frequently (success rate: {best_model['success_rate']:.1%})")

        # Issue-specific recommendations
        common_issues = insights.get("common_issues", [])
        for issue_data in common_issues[:2]:  # Top 2 issues
            if issue_data["frequency"] > len(insights["session_summary"]["total_reflections"]) * 0.3:
                recommendations.append(f"Address recurring issue: {issue_data['issue']}")

        return recommendations

    def generate_performance_report(self, session_id: str = None) -> str:
        """Generate a human-readable performance report.
        
        Args:
            session_id: Optional session ID to filter by
            
        Returns:
            Formatted performance report
        """
        insights = self.get_reflection_insights(session_id)
        
        if "error" in insights:
            return "No reflection data available for analysis."

        report_lines = []
        summary = insights["session_summary"]
        
        # Header
        report_lines.append("🔍 REFLECTION SYSTEM PERFORMANCE REPORT")
        report_lines.append("=" * 50)
        report_lines.append("")

        # Summary
        report_lines.append("📊 SUMMARY")
        report_lines.append(f"• Total Reflections: {summary['total_reflections']}")
        report_lines.append(f"• Models Evaluated: {summary['models_evaluated']}")
        report_lines.append(f"• Average Confidence: {summary['avg_confidence']:.1%}")
        report_lines.append(f"• Swap Rate: {summary['swap_rate']:.1%}")
        
        if summary.get('time_period', {}).get('duration_hours'):
            duration = summary['time_period']['duration_hours']
            report_lines.append(f"• Analysis Period: {duration:.1f} hours")
        report_lines.append("")

        # Best performing model
        best_model = insights.get("best_performing_model")
        if best_model:
            report_lines.append("🏆 BEST PERFORMING MODEL")
            report_lines.append(f"• Model: {best_model['model']}")
            report_lines.append(f"• Success Rate: {best_model['success_rate']:.1%}")
            report_lines.append(f"• Average Confidence: {best_model['avg_confidence']:.1%}")
            report_lines.append(f"• Average Execution Time: {best_model['avg_execution_time']:.1f}s")
            report_lines.append("")

        # Performance trends
        trends = insights.get("performance_trends", {})
        if not trends.get("insufficient_data"):
            report_lines.append("📈 PERFORMANCE TRENDS")
            conf_trend = trends["confidence_trend"]
            report_lines.append(f"• Confidence Trend: {trends['overall_direction']}")
            report_lines.append(f"• Early vs Recent: {conf_trend['early_average']:.1%} → {conf_trend['recent_average']:.1%}")
            report_lines.append("")

        # Common issues
        common_issues = insights.get("common_issues", [])
        if common_issues:
            report_lines.append("⚠️ COMMON ISSUES")
            for issue in common_issues[:3]:
                report_lines.append(f"• {issue['issue']} ({issue['frequency']} times)")
            report_lines.append("")

        # Recommendations
        recommendations = insights.get("recommendations", [])
        if recommendations:
            report_lines.append("💡 RECOMMENDATIONS")
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"{i}. {rec}")

        return "\n".join(report_lines)
