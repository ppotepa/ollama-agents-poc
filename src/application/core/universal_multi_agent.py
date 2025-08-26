"""Universal Multi-Model Agent - A dynamic agent that can switch between multiple models."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.agents.base.base_agent import AbstractAgent
from src.core.agent_factory import get_agent_factory
from src.utils.enhanced_logging import get_logger


class TaskType(Enum):
    """Types of tasks that can influence model selection."""
    CODING = "coding"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    GENERAL = "general"
    DEBUGGING = "debugging"
    DOCUMENTATION = "documentation"
    RESEARCH = "research"


@dataclass
class ModelCapability:
    """Model capability definition."""
    model_id: str
    strengths: list[str]
    task_types: list[TaskType]
    size_gb: float
    performance_score: float


class UniversalMultiAgent(AbstractAgent):
    """Universal agent that can dynamically switch between multiple models based on task requirements."""

    # Available models with their capabilities
    AVAILABLE_MODELS = {
        "qwen2.5-coder:7b": ModelCapability(
            model_id="qwen2.5-coder:7b",
            strengths=["programming", "code_review", "debugging", "software_architecture"],
            task_types=[TaskType.CODING, TaskType.DEBUGGING, TaskType.DOCUMENTATION],
            size_gb=4.5,
            performance_score=8.5
        ),
        "deepcoder:14b": ModelCapability(
            model_id="deepcoder:14b",
            strengths=["advanced_coding", "complex_algorithms", "system_design", "refactoring"],
            task_types=[TaskType.CODING, TaskType.DEBUGGING, TaskType.ANALYSIS],
            size_gb=8.5,
            performance_score=9.2
        ),
        "qwen2.5:7b-instruct-q4_K_M": ModelCapability(
            model_id="qwen2.5:7b-instruct-q4_K_M",
            strengths=["analysis", "instruction_following", "reasoning", "problem_solving"],
            task_types=[TaskType.ANALYSIS, TaskType.GENERAL, TaskType.RESEARCH],
            size_gb=4.2,
            performance_score=8.7
        ),
        "gemma:7b-instruct-q4_K_M": ModelCapability(
            model_id="gemma:7b-instruct-q4_K_M",
            strengths=["creativity", "writing", "conversation", "general_tasks"],
            task_types=[TaskType.CREATIVE, TaskType.GENERAL, TaskType.DOCUMENTATION],
            size_gb=4.1,
            performance_score=8.0
        ),
        "codellama:13b-instruct": ModelCapability(
            model_id="codellama:13b-instruct",
            strengths=["code_generation", "programming_languages", "technical_documentation"],
            task_types=[TaskType.CODING, TaskType.DOCUMENTATION, TaskType.DEBUGGING],
            size_gb=7.3,
            performance_score=8.8
        ),
        "mistral:7b-instruct": ModelCapability(
            model_id="mistral:7b-instruct",
            strengths=["balanced_performance", "versatility", "reasoning", "conversation"],
            task_types=[TaskType.GENERAL, TaskType.ANALYSIS, TaskType.CREATIVE],
            size_gb=4.1,
            performance_score=8.3
        )
    }

    def __init__(self, agent_id: str = "universal-multi", config: dict[str, Any] | None = None):
        """Initialize the Universal Multi-Model Agent."""
        super().__init__(agent_id, config or {})
        self.logger = get_logger()
        self._agent_factory = get_agent_factory()
        self._current_agent: AbstractAgent | None = None
        self._current_model_id: str | None = None
        self._model_switch_history: list[dict[str, Any]] = []
        self._task_context: dict[str, Any] = {}

        # Initialize with default model
        self._initialize_default_model()

    def _initialize_default_model(self):
        """Initialize with a default model for general tasks."""
        default_model = "qwen2.5:7b-instruct-q4_K_M"  # Good general-purpose model
        self.logger.info(f"🔄 Initializing Universal Multi-Agent with default model: {default_model}")
        self._switch_to_model(default_model, "Initial setup")

    def _analyze_task_requirements(self, prompt: str) -> TaskType:
        """Analyze the prompt to determine the required task type."""
        prompt_lower = prompt.lower()

        # Coding indicators
        coding_keywords = [
            "code", "function", "class", "debug", "error", "programming",
            "python", "javascript", "java", "c++", "algorithm", "implement",
            "refactor", "optimize", "bug", "syntax", "compile", "execute"
        ]

        # Analysis indicators
        analysis_keywords = [
            "analyze", "explain", "compare", "evaluate", "assess", "review",
            "breakdown", "examine", "investigate", "understand", "interpret"
        ]

        # Creative indicators
        creative_keywords = [
            "write", "create", "story", "poem", "creative", "imagine",
            "brainstorm", "design", "compose", "generate", "artistic"
        ]

        # Research indicators
        research_keywords = [
            "research", "find", "search", "learn", "study", "discover",
            "information", "facts", "knowledge", "explore", "investigate"
        ]

        # Documentation indicators
        doc_keywords = [
            "document", "documentation", "readme", "guide", "manual",
            "instructions", "tutorial", "help", "explain", "describe"
        ]

        # Count keyword matches
        coding_score = sum(1 for kw in coding_keywords if kw in prompt_lower)
        analysis_score = sum(1 for kw in analysis_keywords if kw in prompt_lower)
        creative_score = sum(1 for kw in creative_keywords if kw in prompt_lower)
        research_score = sum(1 for kw in research_keywords if kw in prompt_lower)
        doc_score = sum(1 for kw in doc_keywords if kw in prompt_lower)

        # Determine task type based on highest score
        scores = {
            TaskType.CODING: coding_score,
            TaskType.ANALYSIS: analysis_score,
            TaskType.CREATIVE: creative_score,
            TaskType.RESEARCH: research_score,
            TaskType.DOCUMENTATION: doc_score
        }

        max_score = max(scores.values())
        if max_score == 0:
            return TaskType.GENERAL

        # Return the task type with highest score
        for task_type, score in scores.items():
            if score == max_score:
                return task_type

        return TaskType.GENERAL

    def _select_optimal_model(self, task_type: TaskType, context: dict[str, Any] = None) -> str:
        """Select the optimal model for the given task type."""
        context = context or {}

        # Filter models that can handle this task type
        suitable_models = [
            model for model in self.AVAILABLE_MODELS.values()
            if task_type in model.task_types
        ]

        if not suitable_models:
            # Fallback to general-purpose models
            self.logger.warning(f"No specific model found for {task_type}, using general-purpose model")
            return "qwen2.5:7b-instruct-q4_K_M"

        # Sort by performance score for the task type
        suitable_models.sort(key=lambda m: m.performance_score, reverse=True)

        # Consider resource constraints if specified
        max_size = context.get("max_model_size_gb", 10.0)
        for model in suitable_models:
            if model.size_gb <= max_size:
                return model.model_id

        # If all models are too large, pick the smallest suitable one
        suitable_models.sort(key=lambda m: m.size_gb)
        return suitable_models[0].model_id

    def _switch_to_model(self, model_id: str, reason: str) -> bool:
        """Switch to a specific model."""
        if self._current_model_id == model_id:
            self.logger.debug(f"Already using model {model_id}, no switch needed")
            return True

        try:
            new_agent = self._agent_factory.create_agent(model_id)
            if new_agent:
                # Record the switch
                switch_record = {
                    "from_model": self._current_model_id,
                    "to_model": model_id,
                    "reason": reason,
                    "timestamp": self._get_timestamp()
                }
                self._model_switch_history.append(switch_record)

                # Update current agent
                old_model = self._current_model_id
                self._current_agent = new_agent
                self._current_model_id = model_id

                self.logger.info(f"🔄 Model switch: {old_model} → {model_id} (Reason: {reason})")
                return True
            else:
                self.logger.error(f"❌ Failed to create agent for model {model_id}")
                return False

        except Exception as e:
            self.logger.error(f"❌ Error switching to model {model_id}: {e}")
            return False

    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        import datetime
        return datetime.datetime.now().isoformat()

    def process_request(self, prompt: str, context: dict[str, Any] = None) -> str:
        """Process a request with automatic model selection."""
        context = context or {}

        # Analyze task requirements
        task_type = self._analyze_task_requirements(prompt)
        self.logger.info(f"📊 Task analysis: {task_type.value}")

        # Select optimal model
        optimal_model = self._select_optimal_model(task_type, context)

        # Switch model if needed
        switch_reason = f"Optimal for {task_type.value} task"
        if not self._switch_to_model(optimal_model, switch_reason):
            self.logger.warning("Failed to switch model, using current agent")

        # Update task context
        self._task_context = {
            "task_type": task_type.value,
            "selected_model": self._current_model_id,
            "prompt_length": len(prompt),
            **context
        }

        # Execute with current agent
        if self._current_agent:
            try:
                return self._current_agent.process(prompt)
            except Exception as e:
                self.logger.error(f"❌ Error processing with {self._current_model_id}: {e}")
                return f"Error: Failed to process request with {self._current_model_id}: {e}"
        else:
            return "Error: No active agent available"

    def stream(self, prompt: str, on_token, context: dict[str, Any] = None):
        """Stream response with automatic model selection."""
        context = context or {}

        # Analyze and switch model if needed
        task_type = self._analyze_task_requirements(prompt)
        optimal_model = self._select_optimal_model(task_type, context)
        switch_reason = f"Optimal for {task_type.value} task"

        if not self._switch_to_model(optimal_model, switch_reason):
            on_token(f"⚠️ Failed to switch to optimal model {optimal_model}, using current model\n")

        # Stream with current agent
        if self._current_agent:
            try:
                return self._current_agent.stream(prompt, on_token)
            except Exception as e:
                error_msg = f"❌ Error streaming with {self._current_model_id}: {e}"
                self.logger.error(error_msg)
                on_token(error_msg)
                return ""
        else:
            error_msg = "❌ No active agent available"
            on_token(error_msg)
            return ""

    def get_status(self) -> dict[str, Any]:
        """Get current status of the multi-agent system."""
        return {
            "current_model": self._current_model_id,
            "available_models": list(self.AVAILABLE_MODELS.keys()),
            "switch_history": self._model_switch_history[-5:],  # Last 5 switches
            "task_context": self._task_context,
            "total_switches": len(self._model_switch_history)
        }

    def force_model_switch(self, model_id: str, reason: str = "Manual override") -> bool:
        """Force a switch to a specific model."""
        if model_id not in self.AVAILABLE_MODELS:
            self.logger.error(f"❌ Unknown model: {model_id}")
            return False

        return self._switch_to_model(model_id, reason)

    def get_model_recommendations(self, task_description: str) -> list[dict[str, Any]]:
        """Get model recommendations for a task."""
        task_type = self._analyze_task_requirements(task_description)

        # Get suitable models
        recommendations = []
        for model in self.AVAILABLE_MODELS.values():
            if task_type in model.task_types:
                score = model.performance_score
                # Boost score if it's a primary strength
                if any(strength in task_description.lower() for strength in model.strengths):
                    score += 0.5

                recommendations.append({
                    "model_id": model.model_id,
                    "score": score,
                    "strengths": model.strengths,
                    "size_gb": model.size_gb,
                    "suitable_for": [t.value for t in model.task_types]
                })

        # Sort by score
        recommendations.sort(key=lambda r: r["score"], reverse=True)
        return recommendations

    # AbstractAgent interface methods
    def load(self):
        """Load the agent."""
        if self._current_agent:
            self._current_agent.load()

    def process(self, prompt: str) -> str:
        """Process a prompt."""
        return self.process_request(prompt)

    def _build_llm(self):
        """Build LLM - delegated to current agent."""
        if self._current_agent and hasattr(self._current_agent, '_build_llm'):
            return self._current_agent._build_llm()
        return None

    def _build_tools(self):
        """Build tools - delegated to current agent."""
        if self._current_agent and hasattr(self._current_agent, '_build_tools'):
            return self._current_agent._build_tools()
        return []

    @property
    def loaded(self) -> bool:
        """Check if agent is loaded."""
        return self._current_agent is not None and self._current_agent.loaded


def create_universal_multi_agent(config: dict[str, Any] = None) -> UniversalMultiAgent:
    """Create a new Universal Multi-Agent."""
    return UniversalMultiAgent("universal-multi", config)


__all__ = ["UniversalMultiAgent", "create_universal_multi_agent", "TaskType", "ModelCapability"]
