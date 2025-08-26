"""Execution strategy determination for task decomposition."""

from typing import Any, Dict, List

from src.core.task_decomposition.types import TaskType, Subtask, TaskPriority
from src.utils.enhanced_logging import get_logger


class ExecutionStrategy:
    """Determines optimal execution strategies for task sets."""

    def __init__(self):
        """Initialize the execution strategy engine."""
        self.logger = get_logger()

    def determine_strategy(self, subtasks: List[Subtask], context: Dict[str, Any] = None) -> str:
        """Determine the best execution strategy for a set of subtasks.

        Args:
            subtasks: List of subtasks to execute
            context: Optional context information

        Returns:
            Execution strategy name
        """
        if context is None:
            context = {}

        if not subtasks:
            return "direct"

        # Analyze task characteristics
        analysis = self._analyze_task_set(subtasks)

        # Determine strategy based on analysis
        if analysis["has_dependencies"] and analysis["task_count"] > 3:
            return "pipeline"
        elif analysis["parallel_potential"] > 0.5 and analysis["task_count"] > 2:
            return "parallel"
        elif analysis["complexity_variance"] > 0.3:
            return "adaptive"
        elif analysis["has_critical_tasks"]:
            return "priority_based"
        else:
            return "sequential"

    def _analyze_task_set(self, subtasks: List[Subtask]) -> Dict[str, Any]:
        """Analyze characteristics of a task set.

        Args:
            subtasks: List of subtasks to analyze

        Returns:
            Dictionary with analysis results
        """
        analysis = {
            "task_count": len(subtasks),
            "has_dependencies": False,
            "parallel_potential": 0.0,
            "complexity_variance": 0.0,
            "has_critical_tasks": False,
            "task_types": set(),
            "priority_distribution": {}
        }

        if not subtasks:
            return analysis

        # Check for dependencies
        total_deps = sum(len(task.dependencies) for task in subtasks)
        analysis["has_dependencies"] = total_deps > 0

        # Calculate parallel potential
        independent_tasks = [task for task in subtasks if not task.dependencies]
        analysis["parallel_potential"] = len(independent_tasks) / len(subtasks)

        # Calculate complexity variance
        complexities = [task.estimated_complexity for task in subtasks]
        if complexities:
            mean_complexity = sum(complexities) / len(complexities)
            variance = sum((c - mean_complexity) ** 2 for c in complexities) / len(complexities)
            analysis["complexity_variance"] = variance ** 0.5

        # Check for critical tasks
        analysis["has_critical_tasks"] = any(
            task.priority == TaskPriority.CRITICAL for task in subtasks
        )

        # Collect task types
        analysis["task_types"] = set(task.task_type for task in subtasks)

        # Priority distribution
        priority_counts = {}
        for task in subtasks:
            priority = task.priority.value
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        analysis["priority_distribution"] = priority_counts

        return analysis

    def get_execution_plan(self, subtasks: List[Subtask], strategy: str) -> Dict[str, Any]:
        """Get detailed execution plan for a strategy.

        Args:
            subtasks: List of subtasks
            strategy: Execution strategy name

        Returns:
            Detailed execution plan
        """
        if strategy == "sequential":
            return self._create_sequential_plan(subtasks)
        elif strategy == "parallel":
            return self._create_parallel_plan(subtasks)
        elif strategy == "pipeline":
            return self._create_pipeline_plan(subtasks)
        elif strategy == "priority_based":
            return self._create_priority_plan(subtasks)
        elif strategy == "adaptive":
            return self._create_adaptive_plan(subtasks)
        else:
            return self._create_direct_plan(subtasks)

    def _create_sequential_plan(self, subtasks: List[Subtask]) -> Dict[str, Any]:
        """Create a sequential execution plan.

        Args:
            subtasks: List of subtasks

        Returns:
            Sequential execution plan
        """
        # Sort by dependencies and priority
        ordered_tasks = self._topological_sort(subtasks)

        phases = []
        for i, task in enumerate(ordered_tasks):
            phases.append({
                "phase": i + 1,
                "tasks": [task.id],
                "description": f"Execute {task.description}",
                "estimated_duration": task.estimated_complexity,
                "dependencies": task.dependencies
            })

        return {
            "strategy": "sequential",
            "phases": phases,
            "total_phases": len(phases),
            "estimated_total_time": sum(task.estimated_complexity for task in subtasks),
            "parallelization": "none"
        }

    def _create_parallel_plan(self, subtasks: List[Subtask]) -> Dict[str, Any]:
        """Create a parallel execution plan.

        Args:
            subtasks: List of subtasks

        Returns:
            Parallel execution plan
        """
        # Group tasks by dependency level
        dependency_levels = self._get_dependency_levels(subtasks)

        phases = []
        for level, tasks in dependency_levels.items():
            if tasks:
                phases.append({
                    "phase": level + 1,
                    "tasks": [task.id for task in tasks],
                    "description": f"Execute {len(tasks)} tasks in parallel",
                    "estimated_duration": max(task.estimated_complexity for task in tasks),
                    "parallelization": "full" if len(tasks) > 1 else "none"
                })

        return {
            "strategy": "parallel",
            "phases": phases,
            "total_phases": len(phases),
            "estimated_total_time": sum(phase["estimated_duration"] for phase in phases),
            "parallelization": "high"
        }

    def _create_pipeline_plan(self, subtasks: List[Subtask]) -> Dict[str, Any]:
        """Create a pipeline execution plan.

        Args:
            subtasks: List of subtasks

        Returns:
            Pipeline execution plan
        """
        # Sort tasks by dependencies
        ordered_tasks = self._topological_sort(subtasks)

        phases = []
        overlap_factor = 0.3  # 30% overlap between phases

        for i, task in enumerate(ordered_tasks):
            start_offset = i * overlap_factor
            phases.append({
                "phase": i + 1,
                "tasks": [task.id],
                "description": f"Pipeline stage: {task.description}",
                "estimated_duration": task.estimated_complexity,
                "start_offset": start_offset,
                "dependencies": task.dependencies
            })

        # Calculate total time with overlap
        if phases:
            last_phase = phases[-1]
            total_time = last_phase["start_offset"] + last_phase["estimated_duration"]
        else:
            total_time = 0.0

        return {
            "strategy": "pipeline",
            "phases": phases,
            "total_phases": len(phases),
            "estimated_total_time": total_time,
            "parallelization": "pipeline",
            "overlap_factor": overlap_factor
        }

    def _create_priority_plan(self, subtasks: List[Subtask]) -> Dict[str, Any]:
        """Create a priority-based execution plan.

        Args:
            subtasks: List of subtasks

        Returns:
            Priority-based execution plan
        """
        # Sort by priority and dependencies
        priority_order = [TaskPriority.CRITICAL, TaskPriority.HIGH, TaskPriority.MEDIUM, TaskPriority.LOW]
        
        phases = []
        processed_tasks = set()

        for priority in priority_order:
            priority_tasks = [
                task for task in subtasks 
                if task.priority == priority and task.id not in processed_tasks
            ]

            if priority_tasks:
                # Further sort by dependencies
                ordered_priority_tasks = self._topological_sort(priority_tasks)
                
                phases.append({
                    "phase": len(phases) + 1,
                    "tasks": [task.id for task in ordered_priority_tasks],
                    "description": f"Execute {priority.value} priority tasks",
                    "estimated_duration": sum(task.estimated_complexity for task in ordered_priority_tasks),
                    "priority": priority.value,
                    "parallelization": "limited" if len(ordered_priority_tasks) > 1 else "none"
                })

                processed_tasks.update(task.id for task in ordered_priority_tasks)

        return {
            "strategy": "priority_based",
            "phases": phases,
            "total_phases": len(phases),
            "estimated_total_time": sum(phase["estimated_duration"] for phase in phases),
            "parallelization": "priority_grouped"
        }

    def _create_adaptive_plan(self, subtasks: List[Subtask]) -> Dict[str, Any]:
        """Create an adaptive execution plan.

        Args:
            subtasks: List of subtasks

        Returns:
            Adaptive execution plan
        """
        # Group tasks by complexity and type
        simple_tasks = [task for task in subtasks if task.estimated_complexity < 0.4]
        complex_tasks = [task for task in subtasks if task.estimated_complexity >= 0.4]

        phases = []

        # Phase 1: Simple tasks in parallel
        if simple_tasks:
            phases.append({
                "phase": 1,
                "tasks": [task.id for task in simple_tasks],
                "description": "Execute simple tasks in parallel",
                "estimated_duration": max(task.estimated_complexity for task in simple_tasks),
                "complexity_level": "simple",
                "parallelization": "full"
            })

        # Phase 2+: Complex tasks with adaptive approach
        for i, task in enumerate(complex_tasks):
            phases.append({
                "phase": len(phases) + 1,
                "tasks": [task.id],
                "description": f"Execute complex task: {task.description}",
                "estimated_duration": task.estimated_complexity,
                "complexity_level": "complex",
                "parallelization": "none"
            })

        return {
            "strategy": "adaptive",
            "phases": phases,
            "total_phases": len(phases),
            "estimated_total_time": sum(phase["estimated_duration"] for phase in phases),
            "parallelization": "adaptive"
        }

    def _create_direct_plan(self, subtasks: List[Subtask]) -> Dict[str, Any]:
        """Create a direct execution plan (single task).

        Args:
            subtasks: List of subtasks

        Returns:
            Direct execution plan
        """
        if not subtasks:
            return {
                "strategy": "direct",
                "phases": [],
                "total_phases": 0,
                "estimated_total_time": 0.0,
                "parallelization": "none"
            }

        task = subtasks[0]
        return {
            "strategy": "direct",
            "phases": [{
                "phase": 1,
                "tasks": [task.id],
                "description": task.description,
                "estimated_duration": task.estimated_complexity,
                "parallelization": "none"
            }],
            "total_phases": 1,
            "estimated_total_time": task.estimated_complexity,
            "parallelization": "none"
        }

    def _topological_sort(self, subtasks: List[Subtask]) -> List[Subtask]:
        """Sort tasks topologically based on dependencies.

        Args:
            subtasks: List of subtasks

        Returns:
            Topologically sorted list of tasks
        """
        # Create a mapping of task IDs to tasks
        task_map = {task.id: task for task in subtasks}
        
        # Calculate in-degree for each task
        in_degree = {task.id: 0 for task in subtasks}
        
        for task in subtasks:
            for dep_id in task.dependencies:
                if dep_id in in_degree:
                    in_degree[task.id] += 1

        # Topological sort using Kahn's algorithm
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            # Sort by priority to break ties
            queue.sort(key=lambda tid: (
                task_map[tid].priority.value,
                -task_map[tid].estimated_complexity
            ))
            
            current_id = queue.pop(0)
            result.append(task_map[current_id])

            # Update in-degrees
            for task in subtasks:
                if current_id in task.dependencies:
                    in_degree[task.id] -= 1
                    if in_degree[task.id] == 0:
                        queue.append(task.id)

        return result

    def _get_dependency_levels(self, subtasks: List[Subtask]) -> Dict[int, List[Subtask]]:
        """Group tasks by dependency level.

        Args:
            subtasks: List of subtasks

        Returns:
            Dictionary mapping levels to lists of tasks
        """
        levels = {}
        task_map = {task.id: task for task in subtasks}
        task_levels = {}

        # Calculate level for each task
        for task in subtasks:
            level = self._calculate_task_level(task, task_map, task_levels)
            if level not in levels:
                levels[level] = []
            levels[level].append(task)

        return levels

    def _calculate_task_level(self, task: Subtask, task_map: Dict[str, Subtask], 
                             memo: Dict[str, int]) -> int:
        """Calculate the dependency level of a task.

        Args:
            task: Task to calculate level for
            task_map: Mapping of task IDs to tasks
            memo: Memoization dictionary

        Returns:
            Dependency level (0 = no dependencies)
        """
        if task.id in memo:
            return memo[task.id]

        if not task.dependencies:
            memo[task.id] = 0
            return 0

        max_dep_level = 0
        for dep_id in task.dependencies:
            if dep_id in task_map:
                dep_level = self._calculate_task_level(task_map[dep_id], task_map, memo)
                max_dep_level = max(max_dep_level, dep_level)

        level = max_dep_level + 1
        memo[task.id] = level
        return level
