"""Execution Planner - Creates step-by-step execution plans with optimal model selection."""
from __future__ import annotations

import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from src.core.task_decomposer import TaskDecomposition, Subtask, TaskType
from src.core.agent_resolver import create_agent_resolver, ModelCandidate
from src.core.model_capability_checker import get_capability_checker
from src.utils.enhanced_logging import get_logger


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
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None
    context_input: Dict[str, Any] = field(default_factory=dict)
    context_output: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
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
        """Check if step failed."""
        return self.status == ExecutionStatus.FAILED


@dataclass
class ExecutionPlan:
    """Complete execution plan with steps and dependencies."""
    task_decomposition: TaskDecomposition
    execution_steps: List[ExecutionStep]
    execution_order: List[str]  # Step IDs in execution order
    total_estimated_time: float
    model_transitions: List[Tuple[str, str]]  # (from_model, to_model) pairs
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_step_by_id(self, step_id: str) -> Optional[ExecutionStep]:
        """Get execution step by ID."""
        for step in self.execution_steps:
            if step.id == step_id:
                return step
        return None
    
    def get_next_pending_step(self) -> Optional[ExecutionStep]:
        """Get the next step that can be executed."""
        for step_id in self.execution_order:
            step = self.get_step_by_id(step_id)
            if step and step.status == ExecutionStatus.PENDING:
                # Check if dependencies are satisfied
                if self._are_dependencies_satisfied(step):
                    return step
        return None
    
    def _are_dependencies_satisfied(self, step: ExecutionStep) -> bool:
        """Check if all dependencies for a step are satisfied."""
        for dep_id in step.subtask.dependencies:
            dep_step = self.get_step_by_id(dep_id)
            if not dep_step or not dep_step.is_completed:
                return False
        return True
    
    def get_completion_percentage(self) -> float:
        """Get execution completion percentage."""
        if not self.execution_steps:
            return 0.0
        completed = sum(1 for step in self.execution_steps if step.is_completed)
        return (completed / len(self.execution_steps)) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert execution plan to dictionary."""
        return {
            "task_decomposition": self.task_decomposition.to_dict() if hasattr(self.task_decomposition, 'to_dict') else str(self.task_decomposition),
            "execution_steps": [
                {
                    "id": step.id,
                    "subtask": step.subtask.to_dict() if hasattr(step.subtask, 'to_dict') else str(step.subtask),
                    "assigned_model": step.assigned_model,
                    "model_confidence": step.model_confidence,
                    "status": step.status.value if hasattr(step.status, 'value') else str(step.status),
                    "start_time": step.start_time,
                    "end_time": step.end_time,
                    "result": step.result,
                    "error": step.error
                } for step in self.execution_steps
            ],
            "execution_order": self.execution_order,
            "total_estimated_time": self.total_estimated_time,
            "model_transitions": self.model_transitions,
            "metadata": self.metadata
        }


class ExecutionPlanner:
    """Creates optimized execution plans for task decompositions."""
    
    def __init__(self, max_model_size_b: float = 14.0):
        """Initialize the execution planner.
        
        Args:
            max_model_size_b: Maximum model size in billions of parameters
        """
        self.logger = get_logger()
        self.agent_resolver = create_agent_resolver(max_size_b=max_model_size_b)
        self.max_model_size_b = max_model_size_b
    
    def create_execution_plan(self, task_decomposition: TaskDecomposition, 
                            context: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        """Create an optimized execution plan from task decomposition.
        
        Args:
            task_decomposition: The task decomposition to plan
            context: Optional context information
            
        Returns:
            ExecutionPlan with optimized model assignments and execution order
        """
        self.logger.info(f"Creating execution plan for {len(task_decomposition.subtasks)} subtasks")
        
        # Create execution steps with model assignments
        execution_steps = self._create_execution_steps(task_decomposition.subtasks, context)
        
        # Determine optimal execution order
        execution_order = self._determine_execution_order(execution_steps)
        
        # Calculate model transitions
        model_transitions = self._calculate_model_transitions(execution_steps, execution_order)
        
        # Estimate total execution time
        total_time = self._estimate_total_execution_time(execution_steps, model_transitions)
        
        plan = ExecutionPlan(
            task_decomposition=task_decomposition,
            execution_steps=execution_steps,
            execution_order=execution_order,
            total_estimated_time=total_time,
            model_transitions=model_transitions,
            metadata={
                "plan_created_at": time.time(),
                "planner_version": "1.0",
                "context_provided": context is not None,
                "optimization_strategy": task_decomposition.execution_strategy
            }
        )
        
        self.logger.info(f"Created execution plan with {len(execution_steps)} steps, "
                        f"{len(model_transitions)} model transitions, "
                        f"estimated time: {total_time:.1f}s")
        
        return plan
    
    def _create_execution_steps(self, subtasks: List[Subtask], 
                              context: Optional[Dict[str, Any]]) -> List[ExecutionStep]:
        """Create execution steps with optimal model assignments."""
        execution_steps = []
        
        for subtask in subtasks:
            # Create a query for model selection based on subtask
            model_selection_query = self._create_model_selection_query(subtask)
            
            # Get model recommendations from agent resolver
            model_candidates = self.agent_resolver.get_model_recommendations(
                model_selection_query, top_n=3
            )
            
            # Select the best model considering subtask preferences
            assigned_model, confidence = self._select_optimal_model(subtask, model_candidates)
            
            # Create execution step
            step = ExecutionStep(
                id=subtask.id,
                subtask=subtask,
                assigned_model=assigned_model,
                model_confidence=confidence,
                context_input=self._prepare_step_context(subtask, context)
            )
            
            execution_steps.append(step)
            
            self.logger.debug(f"Step {subtask.id}: assigned {assigned_model} "
                            f"(confidence: {confidence:.2f})")
        
        return execution_steps
    
    def _create_model_selection_query(self, subtask: Subtask) -> str:
        """Create a query for model selection based on subtask characteristics."""
        query_parts = [subtask.description]
        
        # Add task type context
        if subtask.task_type == TaskType.CODING:
            query_parts.append("This requires advanced coding and programming capabilities.")
        elif subtask.task_type == TaskType.RESEARCH:
            query_parts.append("This requires research and analysis capabilities.")
        elif subtask.task_type == TaskType.FILE_ANALYSIS:
            query_parts.append("This requires file analysis and content understanding.")
        
        # Add complexity context
        if subtask.estimated_complexity > 0.7:
            query_parts.append("This is a complex task requiring advanced capabilities.")
        
        # Add capability requirements
        if subtask.required_capabilities:
            caps = ", ".join(subtask.required_capabilities)
            query_parts.append(f"Required capabilities: {caps}.")
        
        return " ".join(query_parts)
    
    def _select_optimal_model(self, subtask: Subtask, 
                            model_candidates: List[ModelCandidate]) -> Tuple[str, float]:
        """Select the optimal model for a subtask using capability checking."""
        capability_checker = get_capability_checker()
        
        # Determine if this subtask requires tools
        requires_tools = self._subtask_requires_tools(subtask)
        
        # If we have model candidates, filter them by tool capability first
        if model_candidates:
            suitable_candidates = []
            
            for candidate in model_candidates:
                if requires_tools:
                    # Only use models that support tools
                    if capability_checker.supports_tools(candidate.model_id):
                        suitable_candidates.append(candidate)
                else:
                    # Can use any model
                    suitable_candidates.append(candidate)
            
            # If no suitable candidates after filtering, use capability checker
            if not suitable_candidates:
                self.logger.warning(f"No tool-capable models found for {subtask.id}, using capability checker")
                task_type_str = subtask.task_type.value if hasattr(subtask.task_type, 'value') else str(subtask.task_type)
                best_model = capability_checker.get_best_model_for_task(task_type_str, requires_tools)
                if best_model:
                    return best_model, 0.7
            else:
                # Check if any preferred models are in the suitable candidates
                for preferred in subtask.preferred_models:
                    for candidate in suitable_candidates:
                        if candidate.model_id == preferred:
                            # Verify preferred model supports tools if needed
                            if not requires_tools or capability_checker.supports_tools(preferred):
                                boosted_confidence = min(candidate.score + 0.1, 1.0)
                                self.logger.debug(f"Using preferred model {preferred} for {subtask.id}")
                                return candidate.model_id, boosted_confidence
                
                # Use the highest scored suitable candidate
                best_candidate = suitable_candidates[0]
                return best_candidate.model_id, best_candidate.score
        
        # No candidates available, use capability checker
        task_type_str = subtask.task_type.value if hasattr(subtask.task_type, 'value') else str(subtask.task_type)
        best_model = capability_checker.get_best_model_for_task(task_type_str, requires_tools)
        
        if best_model:
            self.logger.debug(f"Using capability checker recommendation: {best_model} for {subtask.id}")
            return best_model, 0.6
        
        # Final fallback - ensure we don't use models without tool support for tool tasks
        if requires_tools:
            tool_models = capability_checker.get_tool_supporting_models()
            if tool_models:
                fallback_model = tool_models[0]
                self.logger.warning(f"Using fallback tool-supporting model {fallback_model} for {subtask.id}")
                return fallback_model, 0.4
        
        # Ultimate fallback for non-tool tasks
        fallback_model = "qwen2.5:7b-instruct-q4_K_M"
        if requires_tools and not capability_checker.supports_tools(fallback_model):
            # Find any model that supports tools
            tool_models = capability_checker.get_tool_supporting_models()
            if tool_models:
                fallback_model = tool_models[0]
        
        return fallback_model, 0.3
    
    def _subtask_requires_tools(self, subtask: Subtask) -> bool:
        """Determine if a subtask requires tool support."""
        # Check required capabilities for tool-requiring operations
        tool_requiring_capabilities = [
            "file_operations", "system_operations", "code_execution", 
            "web_operations", "project_operations"
        ]
        
        for capability in subtask.required_capabilities:
            if any(tool_cap in capability.lower() for tool_cap in tool_requiring_capabilities):
                return True
        
        # Check task type for tool requirements
        if subtask.task_type in [TaskType.CODING, TaskType.SYSTEM_OPERATION]:
            return True
        
        # Check task description for tool-indicating keywords
        tool_keywords = ["file", "code", "execute", "run", "install", "create", "modify", "analyze"]
        task_text = subtask.description.lower()
        if any(keyword in task_text for keyword in tool_keywords):
            return True
        
        return False
    
    def _prepare_step_context(self, subtask: Subtask, 
                            global_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare context for an execution step."""
        step_context = {}
        
        if global_context:
            step_context.update(global_context)
        
        # Add subtask-specific context
        step_context.update({
            "subtask_id": subtask.id,
            "task_type": subtask.task_type.value,
            "priority": subtask.priority.value,
            "complexity": subtask.estimated_complexity,
            "required_capabilities": subtask.required_capabilities,
            "context_needed": subtask.context_needed
        })
        
        return step_context
    
    def _determine_execution_order(self, execution_steps: List[ExecutionStep]) -> List[str]:
        """Determine optimal execution order considering dependencies and model efficiency."""
        # Create dependency graph
        step_map = {step.id: step for step in execution_steps}
        
        # Topological sort with model optimization
        ordered_steps = []
        remaining_steps = set(step.id for step in execution_steps)
        
        while remaining_steps:
            # Find steps with satisfied dependencies
            ready_steps = []
            for step_id in remaining_steps:
                step = step_map[step_id]
                if all(dep not in remaining_steps for dep in step.subtask.dependencies):
                    ready_steps.append(step)
            
            if not ready_steps:
                # Circular dependency or error - add remaining steps
                ordered_steps.extend(remaining_steps)
                break
            
            # Sort ready steps by model efficiency and priority
            ready_steps.sort(key=lambda s: (
                s.assigned_model,  # Group by model
                -s.subtask.priority.value.count('h'),  # Higher priority first
                -s.model_confidence  # Higher confidence first
            ))
            
            # Add the next step
            next_step = ready_steps[0]
            ordered_steps.append(next_step.id)
            remaining_steps.remove(next_step.id)
        
        return ordered_steps
    
    def _calculate_model_transitions(self, execution_steps: List[ExecutionStep], 
                                   execution_order: List[str]) -> List[Tuple[str, str]]:
        """Calculate model transitions in the execution plan."""
        transitions = []
        current_model = None
        
        step_map = {step.id: step for step in execution_steps}
        
        for step_id in execution_order:
            step = step_map[step_id]
            if current_model and current_model != step.assigned_model:
                transitions.append((current_model, step.assigned_model))
            current_model = step.assigned_model
        
        return transitions
    
    def _estimate_total_execution_time(self, execution_steps: List[ExecutionStep],
                                     model_transitions: List[Tuple[str, str]]) -> float:
        """Estimate total execution time including model loading overhead."""
        # Base time estimates per task type (in seconds)
        task_time_estimates = {
            TaskType.CODING: 30.0,
            TaskType.RESEARCH: 15.0,
            TaskType.FILE_ANALYSIS: 10.0,
            TaskType.SYSTEM_OPERATION: 5.0,
            TaskType.DATA_PROCESSING: 20.0,
            TaskType.CREATIVE: 25.0,
            TaskType.GENERAL_QA: 8.0
        }
        
        total_time = 0.0
        
        # Add time for each step
        for step in execution_steps:
            base_time = task_time_estimates.get(step.subtask.task_type, 10.0)
            complexity_multiplier = 1.0 + step.subtask.estimated_complexity
            total_time += base_time * complexity_multiplier
        
        # Add model transition overhead (10 seconds per transition)
        total_time += len(model_transitions) * 10.0
        
        return total_time
    
    def optimize_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Optimize an existing execution plan."""
        # TODO: Implement plan optimization
        # - Reduce model transitions
        # - Parallelize independent steps
        # - Reorder for efficiency
        self.logger.info("Plan optimization not yet implemented")
        return plan
    
    def update_step_status(self, plan: ExecutionPlan, step_id: str, 
                          status: ExecutionStatus, result: Optional[str] = None,
                          error: Optional[str] = None, 
                          context_output: Optional[Dict[str, Any]] = None) -> None:
        """Update the status of an execution step."""
        step = plan.get_step_by_id(step_id)
        if not step:
            self.logger.warning(f"Step {step_id} not found in plan")
            return
        
        step.status = status
        if status == ExecutionStatus.RUNNING:
            step.start_time = time.time()
        elif status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.SKIPPED]:
            step.end_time = time.time()
        
        if result:
            step.result = result
        if error:
            step.error = error
        if context_output:
            step.context_output.update(context_output)
        
        self.logger.debug(f"Updated step {step_id} status to {status.value}")


def create_execution_planner(max_model_size_b: float = 14.0) -> ExecutionPlanner:
    """Create an ExecutionPlanner instance."""
    return ExecutionPlanner(max_model_size_b=max_model_size_b)


__all__ = [
    "ExecutionPlanner", "ExecutionPlan", "ExecutionStep", "ExecutionStatus",
    "create_execution_planner"
]
