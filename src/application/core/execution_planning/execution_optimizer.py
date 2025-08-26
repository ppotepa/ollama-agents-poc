"""Execution Optimizer - Optimizes execution order and model transitions."""

from typing import Any

from src.utils.enhanced_logging import get_logger

from .data_models import ExecutionStep


class ExecutionOptimizer:
    """Optimizes execution order and minimizes model transitions."""

    def __init__(self):
        """Initialize the execution optimizer."""
        self.logger = get_logger()

    def determine_execution_order(self, execution_steps: list[ExecutionStep]) -> list[str]:
        """Determine optimal execution order for steps.

        Args:
            execution_steps: List of execution steps to order

        Returns:
            List of step IDs in optimal execution order
        """
        if not execution_steps:
            return []

        # For now, use a simple strategy that groups by model to minimize transitions
        # Future improvements could consider dependencies, priorities, etc.
        
        # Group steps by assigned model
        model_groups = {}
        for step in execution_steps:
            model = step.assigned_model
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(step)

        # Sort groups by priority and complexity
        sorted_groups = []
        for model, steps in model_groups.items():
            # Sort steps within group by priority then complexity
            steps.sort(key=lambda s: (
                -s.subtask.priority.value if hasattr(s.subtask.priority, 'value') else 0,
                -s.subtask.estimated_complexity
            ))
            sorted_groups.append((model, steps))

        # Sort groups by total complexity (highest first)
        sorted_groups.sort(key=lambda g: sum(s.subtask.estimated_complexity for s in g[1]), reverse=True)

        # Build execution order
        execution_order = []
        for model, steps in sorted_groups:
            for step in steps:
                execution_order.append(step.id)

        self.logger.debug(f"Optimized execution order for {len(execution_steps)} steps "
                         f"across {len(model_groups)} models")

        return execution_order

    def calculate_model_transitions(self, execution_steps: list[ExecutionStep],
                                   execution_order: list[str]) -> list[dict[str, Any]]:
        """Calculate model transitions needed for execution plan.

        Args:
            execution_steps: List of execution steps
            execution_order: Ordered list of step IDs

        Returns:
            List of model transition information
        """
        if not execution_order:
            return []

        # Create step lookup
        step_lookup = {step.id: step for step in execution_steps}

        transitions = []
        current_model = None

        for step_id in execution_order:
            step = step_lookup.get(step_id)
            if not step:
                continue

            if current_model != step.assigned_model:
                # Model transition needed
                transition = {
                    "from_model": current_model,
                    "to_model": step.assigned_model,
                    "step_id": step_id,
                    "transition_type": "load" if current_model is None else "switch",
                    "estimated_time": self._estimate_transition_time(current_model, step.assigned_model)
                }
                transitions.append(transition)
                current_model = step.assigned_model

        self.logger.debug(f"Calculated {len(transitions)} model transitions")
        return transitions

    def _estimate_transition_time(self, from_model: str | None, to_model: str) -> float:
        """Estimate time for model transition.

        Args:
            from_model: Current model (None if first load)
            to_model: Target model

        Returns:
            Estimated transition time in seconds
        """
        if from_model is None:
            # Initial model load
            return self._estimate_model_load_time(to_model)
        
        if from_model == to_model:
            return 0.0  # No transition needed

        # Model switch - unload + load
        unload_time = self._estimate_model_unload_time(from_model)
        load_time = self._estimate_model_load_time(to_model)
        
        return unload_time + load_time

    def _estimate_model_load_time(self, model_name: str) -> float:
        """Estimate model loading time based on model characteristics.

        Args:
            model_name: Name of model to load

        Returns:
            Estimated load time in seconds
        """
        model_name_lower = model_name.lower()
        
        # Estimate based on model size
        if "1b" in model_name_lower:
            return 2.0
        elif "3b" in model_name_lower:
            return 5.0
        elif "7b" in model_name_lower or "8b" in model_name_lower:
            return 10.0
        elif "13b" in model_name_lower or "14b" in model_name_lower:
            return 15.0
        elif "30b" in model_name_lower or "34b" in model_name_lower:
            return 25.0
        elif "70b" in model_name_lower:
            return 45.0
        else:
            return 8.0  # Default estimate

    def _estimate_model_unload_time(self, model_name: str) -> float:
        """Estimate model unloading time.

        Args:
            model_name: Name of model to unload

        Returns:
            Estimated unload time in seconds
        """
        # Unloading is typically much faster than loading
        return 1.0

    def optimize_for_minimal_transitions(self, execution_steps: list[ExecutionStep]) -> list[str]:
        """Optimize execution order specifically to minimize model transitions.

        Args:
            execution_steps: List of execution steps

        Returns:
            Optimized execution order with minimal transitions
        """
        if not execution_steps:
            return []

        # Group by model
        model_groups = {}
        for step in execution_steps:
            model = step.assigned_model
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(step)

        # Execute all steps for each model together
        execution_order = []
        
        # Sort models by estimated load time (lighter models first for faster startup)
        sorted_models = sorted(model_groups.keys(), 
                              key=lambda m: self._estimate_model_load_time(m))

        for model in sorted_models:
            steps = model_groups[model]
            # Sort steps within model by priority
            steps.sort(key=lambda s: (
                -s.subtask.priority.value if hasattr(s.subtask.priority, 'value') else 0,
                -s.subtask.estimated_complexity
            ))
            
            for step in steps:
                execution_order.append(step.id)

        return execution_order

    def get_execution_statistics(self, execution_steps: list[ExecutionStep],
                                execution_order: list[str]) -> dict[str, Any]:
        """Get statistics about the execution plan.

        Args:
            execution_steps: List of execution steps
            execution_order: Execution order

        Returns:
            Dictionary with execution statistics
        """
        if not execution_steps:
            return {}

        # Model usage statistics
        model_usage = {}
        for step in execution_steps:
            model = step.assigned_model
            if model not in model_usage:
                model_usage[model] = {"count": 0, "total_complexity": 0.0}
            model_usage[model]["count"] += 1
            model_usage[model]["total_complexity"] += step.subtask.estimated_complexity

        # Transition count
        transitions = self.calculate_model_transitions(execution_steps, execution_order)
        
        # Complexity distribution
        complexities = [step.subtask.estimated_complexity for step in execution_steps]
        
        return {
            "total_steps": len(execution_steps),
            "unique_models": len(model_usage),
            "model_usage": model_usage,
            "total_transitions": len(transitions),
            "avg_complexity": sum(complexities) / len(complexities) if complexities else 0.0,
            "max_complexity": max(complexities) if complexities else 0.0,
            "min_complexity": min(complexities) if complexities else 0.0
        }
