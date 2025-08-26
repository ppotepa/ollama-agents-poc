"""Data Models for Execution Planning."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.core.task_decomposer import Subtask, TaskDecomposition


class ExecutionStatus(Enum):
    """Status of execution steps."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ExecutionStep:
    """Represents a single execution step with model assignment."""
    id: str
    subtask: Subtask
    assigned_model: str
    model_confidence: float
    status: ExecutionStatus = ExecutionStatus.PENDING
    start_time: float | None = None
    end_time: float | None = None
    result: str | None = None
    error: str | None = None
    context_input: dict[str, Any] = field(default_factory=dict)
    context_output: dict[str, Any] = field(default_factory=dict)
    performance_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float | None:
        """Calculate execution duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def is_completed(self) -> bool:
        """Check if step is completed successfully."""
        return self.status == ExecutionStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        """Check if step has failed."""
        return self.status == ExecutionStatus.FAILED

    @property
    def is_pending(self) -> bool:
        """Check if step is pending execution."""
        return self.status == ExecutionStatus.PENDING

    def start_execution(self):
        """Mark step as started."""
        self.status = ExecutionStatus.RUNNING
        self.start_time = time.time()

    def complete_execution(self, result: str):
        """Mark step as completed with result."""
        self.status = ExecutionStatus.COMPLETED
        self.end_time = time.time()
        self.result = result

    def fail_execution(self, error: str):
        """Mark step as failed with error."""
        self.status = ExecutionStatus.FAILED
        self.end_time = time.time()
        self.error = error


@dataclass
class ExecutionPlan:
    """Complete execution plan with steps and optimization data."""
    task_decomposition: TaskDecomposition
    execution_steps: list[ExecutionStep]
    execution_order: list[str]  # Step IDs in execution order
    total_estimated_time: float
    model_transitions: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        """Get total number of steps."""
        return len(self.execution_steps)

    @property
    def unique_models(self) -> set[str]:
        """Get unique models used in plan."""
        return {step.assigned_model for step in self.execution_steps}

    @property
    def model_count(self) -> int:
        """Get number of unique models."""
        return len(self.unique_models)

    def get_step_by_id(self, step_id: str) -> ExecutionStep | None:
        """Get execution step by ID."""
        for step in self.execution_steps:
            if step.id == step_id:
                return step
        return None

    def get_steps_for_model(self, model_name: str) -> list[ExecutionStep]:
        """Get all steps assigned to a specific model."""
        return [step for step in self.execution_steps if step.assigned_model == model_name]

    def _are_dependencies_satisfied(self, step: ExecutionStep) -> bool:
        """Check if all dependencies for a step are satisfied."""
        # For now, simple sequential dependency
        # In future, could check specific task dependencies
        step_index = next(i for i, s in enumerate(self.execution_steps) if s.id == step.id)
        
        # All previous steps should be completed
        for i in range(step_index):
            if not self.execution_steps[i].is_completed:
                return False
        
        return True

    def get_next_executable_steps(self) -> list[ExecutionStep]:
        """Get steps that are ready for execution."""
        executable_steps = []
        
        for step in self.execution_steps:
            if step.is_pending and self._are_dependencies_satisfied(step):
                executable_steps.append(step)
        
        return executable_steps

    def is_complete(self) -> bool:
        """Check if all steps are completed."""
        return all(step.is_completed for step in self.execution_steps)

    def has_failures(self) -> bool:
        """Check if any steps have failed."""
        return any(step.is_failed for step in self.execution_steps)

    def get_completion_percentage(self) -> float:
        """Get completion percentage."""
        if not self.execution_steps:
            return 100.0
        
        completed_count = sum(1 for step in self.execution_steps if step.is_completed)
        return (completed_count / len(self.execution_steps)) * 100.0
