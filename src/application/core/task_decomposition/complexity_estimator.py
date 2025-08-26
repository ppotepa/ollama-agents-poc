"""Complexity estimation for tasks and subtasks."""

from typing import Any, Dict, List

from src.core.task_decomposition.types import TaskType, Subtask
from src.utils.enhanced_logging import get_logger


class ComplexityEstimator:
    """Estimates complexity of tasks and subtasks."""

    def __init__(self):
        """Initialize the complexity estimator."""
        self.logger = get_logger()
        
        # Base complexity scores for different task types
        self.base_complexity = {
            TaskType.CODING: 0.7,
            TaskType.RESEARCH: 0.5,
            TaskType.FILE_ANALYSIS: 0.4,
            TaskType.SYSTEM_OPERATION: 0.6,
            TaskType.DATA_PROCESSING: 0.5,
            TaskType.CREATIVE: 0.6,
            TaskType.GENERAL_QA: 0.3
        }

    def estimate_query_complexity(self, query: str, task_indicators: Dict[str, float]) -> float:
        """Estimate the complexity of a user query.

        Args:
            query: The user query to analyze
            task_indicators: Task type confidence scores

        Returns:
            Complexity score between 0.0 and 1.0
        """
        base_score = 0.3  # Minimum complexity
        
        # Length factor
        word_count = len(query.split())
        length_factor = min(word_count / 50, 0.3)  # Max 0.3 contribution from length
        
        # Complexity keywords
        complexity_keywords = [
            'multiple', 'several', 'various', 'complex', 'advanced',
            'comprehensive', 'detailed', 'thorough', 'complete',
            'integrate', 'optimize', 'refactor', 'redesign'
        ]
        
        keyword_score = 0.0
        query_lower = query.lower()
        for keyword in complexity_keywords:
            if keyword in query_lower:
                keyword_score += 0.1
        
        keyword_score = min(keyword_score, 0.4)  # Max 0.4 from keywords
        
        # Task type complexity
        if task_indicators:
            primary_task_type = max(task_indicators, key=task_indicators.get)
            # Convert string back to enum for lookup
            for task_type in TaskType:
                if task_type.value == primary_task_type:
                    task_complexity = self.base_complexity.get(task_type, 0.5)
                    break
            else:
                task_complexity = 0.5
        else:
            task_complexity = 0.5
        
        # Sequential operation indicators
        sequence_indicators = ['then', 'after', 'before', 'next', 'finally']
        sequence_score = 0.0
        for indicator in sequence_indicators:
            if indicator in query_lower:
                sequence_score += 0.05
        
        sequence_score = min(sequence_score, 0.2)  # Max 0.2 from sequences
        
        # Calculate final complexity
        total_complexity = (
            base_score +
            length_factor +
            keyword_score +
            (task_complexity * 0.3) +  # Weight task complexity
            sequence_score
        )
        
        return min(total_complexity, 1.0)

    def estimate_subtask_complexity(self, subtask: Subtask) -> float:
        """Estimate the complexity of a subtask.

        Args:
            subtask: The subtask to analyze

        Returns:
            Complexity score between 0.0 and 1.0
        """
        # Base complexity from task type
        base_score = self.base_complexity.get(subtask.task_type, 0.5)
        
        # Description length factor
        desc_length = len(subtask.description.split())
        length_factor = min(desc_length / 30, 0.2)
        
        # Required capabilities factor
        capability_factor = min(len(subtask.required_capabilities) * 0.1, 0.3)
        
        # Dependencies factor
        dependency_factor = min(len(subtask.dependencies) * 0.05, 0.2)
        
        # Context requirements factor
        context_factor = min(len(subtask.context_needed) * 0.05, 0.15)
        
        total_complexity = (
            base_score +
            length_factor +
            capability_factor +
            dependency_factor +
            context_factor
        )
        
        return min(total_complexity, 1.0)

    def estimate_total_complexity(self, subtasks: List[Subtask]) -> float:
        """Estimate total complexity for a list of subtasks.

        Args:
            subtasks: List of subtasks

        Returns:
            Total complexity score
        """
        if not subtasks:
            return 0.0
        
        # Calculate individual complexities
        individual_complexities = [
            self.estimate_subtask_complexity(subtask) for subtask in subtasks
        ]
        
        # Base total is sum of individual complexities
        base_total = sum(individual_complexities)
        
        # Apply scaling factors
        
        # Parallelization factor - some tasks can run in parallel
        parallel_reduction = self._calculate_parallel_reduction(subtasks)
        
        # Coordination overhead - managing multiple tasks adds complexity
        coordination_overhead = self._calculate_coordination_overhead(subtasks)
        
        # Dependency complexity - sequential dependencies add overhead
        dependency_overhead = self._calculate_dependency_overhead(subtasks)
        
        # Calculate final total
        total_complexity = (
            base_total * (1 - parallel_reduction) +
            coordination_overhead +
            dependency_overhead
        )
        
        return min(total_complexity, len(subtasks))  # Cap at number of subtasks

    def _calculate_parallel_reduction(self, subtasks: List[Subtask]) -> float:
        """Calculate complexity reduction from parallel execution.

        Args:
            subtasks: List of subtasks

        Returns:
            Reduction factor (0.0 to 0.5)
        """
        if len(subtasks) <= 1:
            return 0.0
        
        # Count tasks that can potentially run in parallel
        independent_tasks = [task for task in subtasks if not task.dependencies]
        
        if len(independent_tasks) <= 1:
            return 0.0
        
        # Calculate potential parallelization
        parallel_ratio = len(independent_tasks) / len(subtasks)
        
        # Maximum 30% reduction for high parallelization
        return min(parallel_ratio * 0.3, 0.3)

    def _calculate_coordination_overhead(self, subtasks: List[Subtask]) -> float:
        """Calculate overhead from coordinating multiple tasks.

        Args:
            subtasks: List of subtasks

        Returns:
            Coordination overhead (0.0 to 0.5)
        """
        if len(subtasks) <= 1:
            return 0.0
        
        # Base overhead increases with number of tasks
        base_overhead = min(len(subtasks) * 0.05, 0.3)
        
        # Additional overhead for tasks with different types
        unique_types = len(set(task.task_type for task in subtasks))
        type_overhead = min(unique_types * 0.03, 0.2)
        
        return base_overhead + type_overhead

    def _calculate_dependency_overhead(self, subtasks: List[Subtask]) -> float:
        """Calculate overhead from task dependencies.

        Args:
            subtasks: List of subtasks

        Returns:
            Dependency overhead (0.0 to 0.4)
        """
        total_dependencies = sum(len(task.dependencies) for task in subtasks)
        
        if total_dependencies == 0:
            return 0.0
        
        # Calculate dependency complexity
        dependency_ratio = total_dependencies / len(subtasks)
        
        # Higher dependency ratios create more overhead
        return min(dependency_ratio * 0.1, 0.4)

    def get_complexity_breakdown(self, subtasks: List[Subtask]) -> Dict[str, Any]:
        """Get detailed complexity breakdown.

        Args:
            subtasks: List of subtasks

        Returns:
            Dictionary with complexity breakdown
        """
        individual_complexities = [
            self.estimate_subtask_complexity(subtask) for subtask in subtasks
        ]
        
        return {
            "individual_complexities": individual_complexities,
            "base_total": sum(individual_complexities),
            "parallel_reduction": self._calculate_parallel_reduction(subtasks),
            "coordination_overhead": self._calculate_coordination_overhead(subtasks),
            "dependency_overhead": self._calculate_dependency_overhead(subtasks),
            "final_total": self.estimate_total_complexity(subtasks),
            "average_per_task": sum(individual_complexities) / len(subtasks) if subtasks else 0.0,
            "most_complex_task": max(individual_complexities) if individual_complexities else 0.0,
            "complexity_distribution": self._get_complexity_distribution(individual_complexities)
        }

    def _get_complexity_distribution(self, complexities: List[float]) -> Dict[str, int]:
        """Get distribution of complexity levels.

        Args:
            complexities: List of complexity scores

        Returns:
            Distribution by complexity level
        """
        distribution = {
            "simple": 0,      # 0.0 - 0.3
            "moderate": 0,    # 0.3 - 0.6
            "complex": 0,     # 0.6 - 0.8
            "very_complex": 0 # 0.8 - 1.0
        }
        
        for complexity in complexities:
            if complexity < 0.3:
                distribution["simple"] += 1
            elif complexity < 0.6:
                distribution["moderate"] += 1
            elif complexity < 0.8:
                distribution["complex"] += 1
            else:
                distribution["very_complex"] += 1
        
        return distribution
