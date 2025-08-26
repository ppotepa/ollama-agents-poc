"""Core investigation types and data structures."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List


class InvestigationStrategy(Enum):
    """Different investigation strategies."""
    DEPTH_FIRST = "depth_first"
    BREADTH_FIRST = "breadth_first"
    TARGETED = "targeted"
    EXPLORATORY = "exploratory"
    HYPOTHESIS_DRIVEN = "hypothesis_driven"
    INCREMENTAL = "incremental"


class InvestigationPriority(Enum):
    """Priority levels for investigation steps."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    OPTIONAL = "optional"


@dataclass
class InvestigationStep:
    """Individual step in an investigation."""
    step_id: str
    description: str
    strategy: InvestigationStrategy
    priority: InvestigationPriority
    estimated_duration: float
    required_models: List[str]
    dependencies: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    validation_criteria: List[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "description": self.description,
            "strategy": self.strategy.value,
            "priority": self.priority.value,
            "estimated_duration": self.estimated_duration,
            "required_models": self.required_models,
            "dependencies": self.dependencies,
            "expected_outputs": self.expected_outputs,
            "validation_criteria": self.validation_criteria,
            "metadata": self.metadata
        }

    def is_executable(self, completed_steps: List[str]) -> bool:
        """Check if this step can be executed given completed dependencies."""
        return all(dep in completed_steps for dep in self.dependencies)

    def get_complexity_score(self) -> float:
        """Calculate a complexity score for this step."""
        base_score = self.estimated_duration
        
        # Add complexity for dependencies
        dependency_factor = len(self.dependencies) * 0.1
        
        # Add complexity for required models
        model_factor = len(self.required_models) * 0.05
        
        # Priority affects complexity
        priority_factors = {
            InvestigationPriority.CRITICAL: 1.2,
            InvestigationPriority.HIGH: 1.0,
            InvestigationPriority.MEDIUM: 0.8,
            InvestigationPriority.LOW: 0.6,
            InvestigationPriority.OPTIONAL: 0.4
        }
        
        priority_factor = priority_factors.get(self.priority, 1.0)
        
        return (base_score + dependency_factor + model_factor) * priority_factor


@dataclass
class InvestigationPlan:
    """Complete investigation plan with ordered steps."""
    investigation_id: str
    query: str
    strategy: InvestigationStrategy
    steps: List[InvestigationStep]
    total_estimated_duration: float
    success_criteria: List[str]
    fallback_strategies: List[InvestigationStrategy] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def get_next_executable_steps(self, completed_steps: List[str]) -> List[InvestigationStep]:
        """Get steps that can be executed next based on completed dependencies."""
        executable = []

        for step in self.steps:
            if step.step_id in completed_steps:
                continue

            # Check if this step can be executed
            if step.is_executable(completed_steps):
                executable.append(step)

        # Sort by priority
        priority_order = {
            InvestigationPriority.CRITICAL: 0,
            InvestigationPriority.HIGH: 1,
            InvestigationPriority.MEDIUM: 2,
            InvestigationPriority.LOW: 3,
            InvestigationPriority.OPTIONAL: 4
        }

        executable.sort(key=lambda x: priority_order[x.priority])
        return executable

    def get_critical_path(self) -> List[InvestigationStep]:
        """Get the critical path through the investigation plan."""
        # Find critical and high priority steps
        critical_steps = [
            step for step in self.steps 
            if step.priority in [InvestigationPriority.CRITICAL, InvestigationPriority.HIGH]
        ]
        
        # Sort by dependencies and priority
        return sorted(critical_steps, key=lambda x: (len(x.dependencies), x.priority.value))

    def estimate_parallel_execution_time(self) -> float:
        """Estimate execution time if steps can be run in parallel."""
        # Group steps by dependency level
        dependency_levels = {}
        
        for step in self.steps:
            level = len(step.dependencies)
            if level not in dependency_levels:
                dependency_levels[level] = []
            dependency_levels[level].append(step)
        
        # Calculate time for each level (parallel execution within level)
        total_time = 0.0
        for level in sorted(dependency_levels.keys()):
            level_steps = dependency_levels[level]
            # Max time in this level (parallel execution)
            level_time = max(step.estimated_duration for step in level_steps) if level_steps else 0
            total_time += level_time
        
        return total_time

    def get_resource_requirements(self) -> dict[str, Any]:
        """Get resource requirements for the entire plan."""
        all_models = set()
        max_parallel_models = 0
        
        # Analyze by dependency level for parallel requirements
        dependency_levels = {}
        for step in self.steps:
            level = len(step.dependencies)
            if level not in dependency_levels:
                dependency_levels[level] = []
            dependency_levels[level].append(step)
            all_models.update(step.required_models)
        
        # Find max models needed at any level
        for level_steps in dependency_levels.values():
            level_models = set()
            for step in level_steps:
                level_models.update(step.required_models)
            max_parallel_models = max(max_parallel_models, len(level_models))
        
        return {
            "total_unique_models": len(all_models),
            "all_models": list(all_models),
            "max_parallel_models": max_parallel_models,
            "total_steps": len(self.steps),
            "critical_steps": len([s for s in self.steps if s.priority == InvestigationPriority.CRITICAL]),
            "estimated_serial_time": self.total_estimated_duration,
            "estimated_parallel_time": self.estimate_parallel_execution_time()
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "investigation_id": self.investigation_id,
            "query": self.query,
            "strategy": self.strategy.value,
            "steps": [step.to_dict() for step in self.steps],
            "total_estimated_duration": self.total_estimated_duration,
            "success_criteria": self.success_criteria,
            "fallback_strategies": [s.value for s in self.fallback_strategies],
            "created_at": self.created_at,
            "resource_requirements": self.get_resource_requirements()
        }

    def validate_plan(self) -> List[str]:
        """Validate the investigation plan and return any issues."""
        issues = []
        
        # Check for circular dependencies
        step_ids = {step.step_id for step in self.steps}
        
        for step in self.steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    issues.append(f"Step '{step.step_id}' depends on unknown step '{dep}'")
        
        # Check for orphaned steps (no path to execution)
        executable_steps = set()
        remaining_steps = {step.step_id: step for step in self.steps}
        
        # Find steps with no dependencies
        while remaining_steps:
            found_executable = False
            
            for step_id, step in list(remaining_steps.items()):
                if all(dep in executable_steps for dep in step.dependencies):
                    executable_steps.add(step_id)
                    del remaining_steps[step_id]
                    found_executable = True
            
            if not found_executable and remaining_steps:
                # Circular dependency or orphaned steps
                orphaned = list(remaining_steps.keys())
                issues.append(f"Orphaned steps with unresolvable dependencies: {orphaned}")
                break
        
        # Check for reasonable duration estimates
        for step in self.steps:
            if step.estimated_duration <= 0:
                issues.append(f"Step '{step.step_id}' has non-positive duration")
            elif step.estimated_duration > 3600:  # 1 hour
                issues.append(f"Step '{step.step_id}' has unusually long duration: {step.estimated_duration}s")
        
        return issues
