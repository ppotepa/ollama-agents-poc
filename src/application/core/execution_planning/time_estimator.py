"""Time Estimator - Estimates execution times for plans and steps."""

from typing import Any

from src.utils.enhanced_logging import get_logger

from .data_models import ExecutionStep


class TimeEstimator:
    """Estimates execution times for steps, transitions, and complete plans."""

    def __init__(self):
        """Initialize the time estimator."""
        self.logger = get_logger()

        # Base time estimates for different types of tasks (in seconds)
        self.base_times = {
            "code_analysis": 15.0,
            "code_generation": 25.0,
            "debugging": 20.0,
            "testing": 18.0,
            "documentation": 12.0,
            "file_operations": 8.0,
            "data_processing": 22.0,
            "research": 30.0,
            "general_qa": 10.0
        }

        # Model speed multipliers (relative to baseline)
        self.model_speed_multipliers = {
            "1b": 0.3,   # Very fast but limited
            "3b": 0.5,   # Fast
            "7b": 1.0,   # Baseline
            "8b": 1.1,   # Slightly slower
            "13b": 1.8,  # Slower but more capable
            "14b": 2.0,  # Slower
            "30b": 3.5,  # Much slower
            "34b": 4.0,  # Much slower
            "70b": 7.0   # Very slow
        }

    def estimate_total_execution_time(self, execution_steps: list[ExecutionStep],
                                     model_transitions: list[dict[str, Any]]) -> float:
        """Estimate total execution time for a plan.

        Args:
            execution_steps: List of execution steps
            model_transitions: List of model transitions

        Returns:
            Estimated total execution time in seconds
        """
        if not execution_steps:
            return 0.0

        # Calculate step execution times
        step_time = sum(self.estimate_step_execution_time(step) for step in execution_steps)

        # Calculate transition times
        transition_time = sum(transition.get("estimated_time", 0.0) for transition in model_transitions)

        # Add overhead (startup, context switching, etc.)
        overhead_time = len(execution_steps) * 2.0  # 2 seconds per step overhead

        total_time = step_time + transition_time + overhead_time

        self.logger.debug(f"Estimated execution time: {total_time:.1f}s "
                         f"(steps: {step_time:.1f}s, transitions: {transition_time:.1f}s, "
                         f"overhead: {overhead_time:.1f}s)")

        return total_time

    def estimate_step_execution_time(self, step: ExecutionStep) -> float:
        """Estimate execution time for a single step.

        Args:
            step: Execution step to estimate

        Returns:
            Estimated execution time in seconds
        """
        # Get base time for task type
        task_type = step.subtask.task_type.value
        base_time = self.base_times.get(task_type, 15.0)  # Default 15 seconds

        # Apply complexity multiplier
        complexity_multiplier = 0.5 + (step.subtask.estimated_complexity * 1.5)

        # Apply model speed multiplier
        model_multiplier = self._get_model_speed_multiplier(step.assigned_model)

        # Apply confidence penalty (lower confidence = more time)
        confidence_multiplier = 1.0 + (1.0 - step.model_confidence) * 0.5

        # Calculate final time
        estimated_time = base_time * complexity_multiplier * model_multiplier * confidence_multiplier

        # Apply reasonable bounds
        estimated_time = max(3.0, min(estimated_time, 300.0))  # 3 seconds to 5 minutes

        return estimated_time

    def _get_model_speed_multiplier(self, model_name: str) -> float:
        """Get speed multiplier for a model.

        Args:
            model_name: Name of the model

        Returns:
            Speed multiplier (lower = faster)
        """
        model_name_lower = model_name.lower()

        # Check for size indicators
        for size, multiplier in self.model_speed_multipliers.items():
            if size in model_name_lower:
                return multiplier

        # Model-specific adjustments
        if "mistral" in model_name_lower or "mixtral" in model_name_lower:
            return 1.2  # Slightly slower than llama of same size
        elif "qwen" in model_name_lower:
            return 0.9  # Slightly faster
        elif "deepseek" in model_name_lower:
            return 1.1  # Slightly slower

        return 1.0  # Default baseline

    def estimate_parallel_execution_time(self, execution_steps: list[ExecutionStep],
                                        parallelism_factor: float = 1.0) -> float:
        """Estimate execution time if steps can be run in parallel.

        Args:
            execution_steps: List of execution steps
            parallelism_factor: How many steps can run in parallel (1.0 = sequential)

        Returns:
            Estimated parallel execution time in seconds
        """
        if not execution_steps or parallelism_factor <= 0:
            return 0.0

        # Get individual step times
        step_times = [self.estimate_step_execution_time(step) for step in execution_steps]

        if parallelism_factor >= len(step_times):
            # Can run all steps in parallel
            return max(step_times)

        # Simulate parallel execution
        # Sort steps by execution time (longest first)
        sorted_times = sorted(step_times, reverse=True)
        
        # Distribute steps across parallel slots
        parallel_slots = [0.0] * int(parallelism_factor)
        
        for step_time in sorted_times:
            # Assign to slot with least total time
            min_slot = min(range(len(parallel_slots)), key=lambda i: parallel_slots[i])
            parallel_slots[min_slot] += step_time

        # Total time is the maximum slot time
        return max(parallel_slots)

    def get_time_breakdown(self, execution_steps: list[ExecutionStep],
                          model_transitions: list[dict[str, Any]]) -> dict[str, Any]:
        """Get detailed time breakdown for analysis.

        Args:
            execution_steps: List of execution steps
            model_transitions: List of model transitions

        Returns:
            Dictionary with detailed time breakdown
        """
        if not execution_steps:
            return {}

        # Step times by model
        model_times = {}
        for step in execution_steps:
            model = step.assigned_model
            if model not in model_times:
                model_times[model] = {"count": 0, "total_time": 0.0, "steps": []}
            
            step_time = self.estimate_step_execution_time(step)
            model_times[model]["count"] += 1
            model_times[model]["total_time"] += step_time
            model_times[model]["steps"].append({
                "step_id": step.id,
                "estimated_time": step_time,
                "task_type": step.subtask.task_type.value,
                "complexity": step.subtask.estimated_complexity
            })

        # Task type times
        task_type_times = {}
        for step in execution_steps:
            task_type = step.subtask.task_type.value
            if task_type not in task_type_times:
                task_type_times[task_type] = {"count": 0, "total_time": 0.0}
            
            step_time = self.estimate_step_execution_time(step)
            task_type_times[task_type]["count"] += 1
            task_type_times[task_type]["total_time"] += step_time

        # Transition times
        total_transition_time = sum(t.get("estimated_time", 0.0) for t in model_transitions)
        
        # Total times
        total_step_time = sum(model_times[m]["total_time"] for m in model_times)
        overhead_time = len(execution_steps) * 2.0

        return {
            "total_estimated_time": total_step_time + total_transition_time + overhead_time,
            "step_execution_time": total_step_time,
            "model_transition_time": total_transition_time,
            "overhead_time": overhead_time,
            "model_breakdown": model_times,
            "task_type_breakdown": task_type_times,
            "transition_details": model_transitions,
            "average_step_time": total_step_time / len(execution_steps) if execution_steps else 0.0
        }

    def estimate_resource_usage(self, execution_steps: list[ExecutionStep]) -> dict[str, Any]:
        """Estimate resource usage for execution plan.

        Args:
            execution_steps: List of execution steps

        Returns:
            Dictionary with resource usage estimates
        """
        if not execution_steps:
            return {}

        # Model memory usage estimates (in GB)
        model_memory_usage = {}
        for step in execution_steps:
            model = step.assigned_model
            if model not in model_memory_usage:
                model_memory_usage[model] = self._estimate_model_memory(model)

        # Peak memory (largest model)
        peak_memory = max(model_memory_usage.values()) if model_memory_usage else 0.0

        # Compute estimates based on task complexity
        compute_units = sum(step.subtask.estimated_complexity for step in execution_steps)

        return {
            "peak_memory_gb": peak_memory,
            "model_memory_usage": model_memory_usage,
            "estimated_compute_units": compute_units,
            "models_required": list(model_memory_usage.keys()),
            "total_model_switches": len(set(step.assigned_model for step in execution_steps)) - 1
        }

    def _estimate_model_memory(self, model_name: str) -> float:
        """Estimate memory usage for a model in GB.

        Args:
            model_name: Name of the model

        Returns:
            Estimated memory usage in GB
        """
        model_name_lower = model_name.lower()

        # Memory estimates based on model size
        if "1b" in model_name_lower:
            return 2.0
        elif "3b" in model_name_lower:
            return 4.0
        elif "7b" in model_name_lower:
            return 8.0
        elif "8b" in model_name_lower:
            return 10.0
        elif "13b" in model_name_lower:
            return 16.0
        elif "14b" in model_name_lower:
            return 18.0
        elif "30b" in model_name_lower:
            return 32.0
        elif "34b" in model_name_lower:
            return 36.0
        elif "70b" in model_name_lower:
            return 64.0
        else:
            return 8.0  # Default estimate
