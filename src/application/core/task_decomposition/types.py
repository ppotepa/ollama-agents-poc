"""Task types and data structures for task decomposition."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskType(Enum):
    """Types of tasks that can be identified."""
    CODING = "coding"
    RESEARCH = "research"
    FILE_ANALYSIS = "file_analysis"
    SYSTEM_OPERATION = "system_operation"
    DATA_PROCESSING = "data_processing"
    CREATIVE = "creative"
    GENERAL_QA = "general_qa"


class TaskPriority(Enum):
    """Priority levels for tasks."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Subtask:
    """Represents a decomposed subtask."""
    id: str
    description: str
    task_type: TaskType
    priority: TaskPriority
    estimated_complexity: float  # 0.0 to 1.0
    required_capabilities: list[str]
    dependencies: list[str] = field(default_factory=list)
    preferred_models: list[str] = field(default_factory=list)
    context_needed: list[str] = field(default_factory=list)
    expected_output: str = ""

    def __post_init__(self):
        """Set preferred models based on task type."""
        if not self.preferred_models:
            self.preferred_models = self._get_default_preferred_models()

    def _get_default_preferred_models(self) -> list[str]:
        """Get default preferred models based on task type."""
        model_preferences = {
            TaskType.CODING: ["codellama:7b", "qwen2.5-coder:7b", "deepseek-coder:6.7b"],
            TaskType.RESEARCH: ["qwen2.5:7b", "llama3.2:3b", "mistral:7b"],
            TaskType.FILE_ANALYSIS: ["codellama:7b", "qwen2.5:7b", "llama3.2:3b"],
            TaskType.SYSTEM_OPERATION: ["qwen2.5:7b", "llama3.2:3b"],
            TaskType.DATA_PROCESSING: ["qwen2.5:7b", "llama3.2:3b"],
            TaskType.CREATIVE: ["llama3.2:3b", "mistral:7b"],
            TaskType.GENERAL_QA: ["qwen2.5:7b", "llama3.2:3b", "mistral:7b"]
        }
        return model_preferences.get(self.task_type, ["llama3.2:3b"])

    def _task_requires_tools(self) -> bool:
        """Determine if this task requires tool support."""
        # Tasks that typically require tools
        tool_requiring_capabilities = [
            "file_operations", "system_operations", "code_execution",
            "web_operations", "project_operations"
        ]

        # Check if any required capabilities need tools
        for capability in self.required_capabilities:
            if any(tool_cap in capability.lower() for tool_cap in tool_requiring_capabilities):
                return True

        # Check task type for tool requirements
        if self.task_type in [TaskType.CODING, TaskType.SYSTEM_OPERATION]:
            # These often need file operations or code execution
            return True

        return False

    def _get_required_capability(self) -> str:
        """Get the primary capability required for this task type."""
        capability_mapping = {
            TaskType.CODING: "coding",
            TaskType.RESEARCH: "general_qa",
            TaskType.FILE_ANALYSIS: "file_operations",
            TaskType.SYSTEM_OPERATION: "general_qa",
            TaskType.DATA_PROCESSING: "general_qa",
            TaskType.CREATIVE: "general_qa",
            TaskType.GENERAL_QA: "general_qa"
        }
        return capability_mapping.get(self.task_type, "general_qa")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "task_type": self.task_type.value if hasattr(self.task_type, 'value') else str(self.task_type),
            "priority": self.priority.value if hasattr(self.priority, 'value') else str(self.priority),
            "estimated_complexity": self.estimated_complexity,
            "required_capabilities": self.required_capabilities,
            "dependencies": self.dependencies,
            "preferred_models": self.preferred_models,
            "context_needed": self.context_needed,
            "expected_output": self.expected_output
        }


@dataclass
class TaskDecomposition:
    """Result of task decomposition."""
    original_query: str
    subtasks: list[Subtask]
    execution_strategy: str
    estimated_total_complexity: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_query": self.original_query,
            "subtasks": [subtask.to_dict() for subtask in self.subtasks],
            "execution_strategy": self.execution_strategy,
            "estimated_total_complexity": self.estimated_total_complexity,
            "metadata": self.metadata
        }
