"""Step Creator - Creates execution steps with model assignments."""

from typing import Any

from src.core.agent_resolver import create_agent_resolver
from src.core.task_decomposer import Subtask
from src.utils.enhanced_logging import get_logger

from .data_models import ExecutionStep
from .model_selector import ModelSelector


class StepCreator:
    """Creates execution steps with optimal model assignments."""

    def __init__(self, max_model_size_b: float = 14.0):
        """Initialize the step creator.

        Args:
            max_model_size_b: Maximum model size in billions of parameters
        """
        self.logger = get_logger()
        self.agent_resolver = create_agent_resolver(max_size_b=max_model_size_b)
        self.model_selector = ModelSelector(self.agent_resolver)
        self.max_model_size_b = max_model_size_b

    def create_execution_steps(self, subtasks: list[Subtask],
                              context: dict[str, Any] | None = None) -> list[ExecutionStep]:
        """Create execution steps with optimal model assignments.

        Args:
            subtasks: List of subtasks to create steps for
            context: Optional context information

        Returns:
            List of execution steps with model assignments
        """
        execution_steps = []

        for subtask in subtasks:
            # Create a query for model selection based on subtask
            model_selection_query = self._create_model_selection_query(subtask)

            # Get model recommendations from agent resolver
            model_candidates = self.agent_resolver.get_model_recommendations(
                model_selection_query, top_n=3
            )

            # Select the best model considering subtask preferences
            assigned_model, confidence = self.model_selector.select_optimal_model(
                subtask, model_candidates
            )

            # Create execution step
            step = ExecutionStep(
                id=f"step_{subtask.id}",
                subtask=subtask,
                assigned_model=assigned_model,
                model_confidence=confidence,
                context_input=self._prepare_step_context(subtask, context),
                metadata={
                    "model_candidates": [c.model_name for c in model_candidates],
                    "selection_query": model_selection_query,
                    "requires_tools": self._subtask_requires_tools(subtask)
                }
            )

            execution_steps.append(step)
            
            self.logger.debug(f"Created step {step.id} with model {assigned_model} "
                             f"(confidence: {confidence:.2f})")

        return execution_steps

    def _create_model_selection_query(self, subtask: Subtask) -> str:
        """Create a query for model selection based on subtask characteristics.

        Args:
            subtask: The subtask to create query for

        Returns:
            Query string optimized for model selection
        """
        # Base query from subtask description
        query_parts = [subtask.description]

        # Add task type specific requirements
        task_type_hints = {
            "code_analysis": "analyze code structure and patterns",
            "code_generation": "generate working code solutions",
            "debugging": "identify and fix code issues",
            "testing": "create comprehensive test cases",
            "documentation": "write clear technical documentation",
            "file_operations": "manage files and directories",
            "data_processing": "process and transform data",
            "research": "find and analyze information",
            "general_qa": "answer questions accurately"
        }

        if subtask.task_type.value in task_type_hints:
            query_parts.append(task_type_hints[subtask.task_type.value])

        # Add capability requirements
        if subtask.required_capabilities:
            capabilities_text = "Required capabilities: " + ", ".join(subtask.required_capabilities)
            query_parts.append(capabilities_text)

        # Add complexity indicator
        if subtask.estimated_complexity > 0.7:
            query_parts.append("complex task requiring advanced reasoning")
        elif subtask.estimated_complexity < 0.3:
            query_parts.append("simple task requiring basic capabilities")

        return " | ".join(query_parts)

    def _subtask_requires_tools(self, subtask: Subtask) -> bool:
        """Check if subtask requires tool/function calling capabilities.

        Args:
            subtask: Subtask to check

        Returns:
            True if tools are required
        """
        tool_indicators = [
            "file", "directory", "create", "write", "read", "execute",
            "run", "install", "build", "compile", "test", "debug",
            "search", "find", "analyze", "process"
        ]

        description_lower = subtask.description.lower()
        return any(indicator in description_lower for indicator in tool_indicators)

    def _prepare_step_context(self, subtask: Subtask,
                             global_context: dict[str, Any] | None) -> dict[str, Any]:
        """Prepare context information for execution step.

        Args:
            subtask: The subtask being prepared
            global_context: Global context from plan creation

        Returns:
            Context dictionary for the execution step
        """
        step_context = {
            "subtask_id": subtask.id,
            "task_type": subtask.task_type.value,
            "priority": subtask.priority.value,
            "complexity": subtask.estimated_complexity,
            "required_capabilities": subtask.required_capabilities,
            "context_needed": subtask.context_needed
        }

        # Add global context if provided
        if global_context:
            step_context["global_context"] = global_context

        # Add task-specific context
        if subtask.context_needed:
            for context_key in subtask.context_needed:
                if global_context and context_key in global_context:
                    step_context[context_key] = global_context[context_key]

        return step_context
