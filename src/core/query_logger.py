"""Enhanced Query Logging System - Comprehensive execution tracking and analysis."""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.enhanced_logging import get_logger


@dataclass
class ContextInfo:
    """Information about context used in query execution."""
    source: str  # 'user_input', 'file_content', 'tool_output', 'agent_switch', etc.
    content: str
    size_chars: int
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptDecoration:
    """Details about how prompts were decorated/enhanced."""
    original_prompt: str
    decorated_prompt: str
    decorations_applied: list[str]  # List of decoration types applied
    context_added: int  # Number of characters of context added
    system_message: str | None = None
    prompt_hash: str = ""

    def __post_init__(self):
        """Generate hash for prompt tracking."""
        if not self.prompt_hash:
            self.prompt_hash = hashlib.md5(
                self.decorated_prompt.encode('utf-8')
            ).hexdigest()[:12]


@dataclass
class AgentSwitch:
    """Information about agent/model switching during execution."""
    from_agent: str | None
    to_agent: str
    reason: str
    timestamp: str
    context_preserved: str
    context_size: int
    success: bool


@dataclass
class ToolExecution:
    """Details about tool execution."""
    tool_name: str
    args: dict[str, Any]
    execution_time: float
    success: bool
    output_size: int
    output_preview: str  # First 200 chars of output
    follow_up_tools: list[str] = field(default_factory=list)
    timestamp: str = ""


@dataclass
class ExecutionStep:
    """Single step in the execution tree."""
    step_number: int
    step_type: str  # 'tool_execution', 'agent_analysis', 'model_switch', 'final_answer'
    agent_used: str
    description: str
    tools_executed: list[ToolExecution] = field(default_factory=list)
    agent_switches: list[AgentSwitch] = field(default_factory=list)
    context_used: list[ContextInfo] = field(default_factory=list)
    prompt_decorations: list[PromptDecoration] = field(default_factory=list)
    output_generated: str | None = None
    execution_time: float = 0.0
    timestamp: str = ""


@dataclass
class QueryExecutionLog:
    """Complete log of a query execution session."""
    query_id: str
    original_query: str
    execution_mode: str  # 'intelligent', 'collaborative', 'single'
    start_time: str
    end_time: str | None = None
    total_execution_time: float = 0.0

    # Execution flow
    execution_steps: list[ExecutionStep] = field(default_factory=list)

    # Context tracking
    total_context_used: list[ContextInfo] = field(default_factory=list)
    context_evolution: list[dict[str, Any]] = field(default_factory=list)

    # Agent/Model information
    models_used: list[str] = field(default_factory=list)
    agent_switches: list[AgentSwitch] = field(default_factory=list)

    # Tool usage
    tools_executed: list[ToolExecution] = field(default_factory=list)
    unique_tools: list[str] = field(default_factory=list)

    # Prompt evolution
    prompt_decorations: list[PromptDecoration] = field(default_factory=list)

    # Results
    final_answer: str | None = None
    success: bool = False
    error_messages: list[str] = field(default_factory=list)

    # Metrics
    metrics: dict[str, Any] = field(default_factory=dict)


class QueryLogger:
    """Logger for tracking query execution with comprehensive details."""

    def __init__(self, log_directory: str = "logs/query_execution"):
        """Initialize the query logger."""
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()

        # Current session tracking
        self.current_log: QueryExecutionLog | None = None
        self.current_step: ExecutionStep | None = None
        self.step_counter = 0

        # Accumulated logs
        self.session_logs: list[QueryExecutionLog] = []

    def start_query_session(self, query: str, execution_mode: str = "unknown") -> str:
        """Start a new query logging session."""
        query_id = str(uuid.uuid4())[:8]

        self.current_log = QueryExecutionLog(
            query_id=query_id,
            original_query=query,
            execution_mode=execution_mode,
            start_time=datetime.now().isoformat(),
            metrics={}
        )

        self.step_counter = 0
        self.current_step = None

        self.logger.info(f"📊 Started query logging session: {query_id}")
        return query_id

    def start_execution_step(self, step_type: str, agent_used: str, description: str) -> int:
        """Start a new execution step."""
        self.step_counter += 1

        self.current_step = ExecutionStep(
            step_number=self.step_counter,
            step_type=step_type,
            agent_used=agent_used,
            description=description,
            timestamp=datetime.now().isoformat()
        )

        if self.current_log:
            self.current_log.execution_steps.append(self.current_step)

        return self.step_counter

    def log_context_usage(self, source: str, content: str, metadata: dict[str, Any] = None):
        """Log context information used in execution."""
        context_info = ContextInfo(
            source=source,
            content=content,
            size_chars=len(content),
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )

        # Add to current step
        if self.current_step:
            self.current_step.context_used.append(context_info)

        # Add to overall log
        if self.current_log:
            self.current_log.total_context_used.append(context_info)

    def log_prompt_decoration(self, original: str, decorated: str, decorations: list[str],
                            system_message: str = None):
        """Log prompt decoration details."""
        decoration = PromptDecoration(
            original_prompt=original,
            decorated_prompt=decorated,
            decorations_applied=decorations,
            context_added=len(decorated) - len(original),
            system_message=system_message
        )

        # Add to current step
        if self.current_step:
            self.current_step.prompt_decorations.append(decoration)

        # Add to overall log
        if self.current_log:
            self.current_log.prompt_decorations.append(decoration)

    def log_tool_execution(self, tool_name: str, args: dict[str, Any],
                          execution_time: float, success: bool, output: str,
                          follow_up_tools: list[str] = None):
        """Log tool execution details."""
        tool_exec = ToolExecution(
            tool_name=tool_name,
            args=args,
            execution_time=execution_time,
            success=success,
            output_size=len(output),
            output_preview=output[:200] if output else "",
            follow_up_tools=follow_up_tools or [],
            timestamp=datetime.now().isoformat()
        )

        # Add to current step
        if self.current_step:
            self.current_step.tools_executed.append(tool_exec)

        # Add to overall log
        if self.current_log:
            self.current_log.tools_executed.append(tool_exec)
            if tool_name not in self.current_log.unique_tools:
                self.current_log.unique_tools.append(tool_name)

    def log_agent_switch(self, from_agent: str | None, to_agent: str, reason: str,
                        context_preserved: str, success: bool):
        """Log agent/model switching details."""
        switch = AgentSwitch(
            from_agent=from_agent,
            to_agent=to_agent,
            reason=reason,
            timestamp=datetime.now().isoformat(),
            context_preserved=context_preserved,
            context_size=len(context_preserved),
            success=success
        )

        # Add to current step
        if self.current_step:
            self.current_step.agent_switches.append(switch)

        # Add to overall log
        if self.current_log:
            self.current_log.agent_switches.append(switch)
            if to_agent not in self.current_log.models_used:
                self.current_log.models_used.append(to_agent)

    def log_context_evolution(self, context_state: dict[str, Any]):
        """Log how context evolves during execution."""
        if self.current_log:
            evolution_entry = {
                "timestamp": datetime.now().isoformat(),
                "step": self.step_counter,
                "context_state": context_state
            }
            self.current_log.context_evolution.append(evolution_entry)

    def end_execution_step(self, output_generated: str = None, execution_time: float = 0.0):
        """End the current execution step."""
        if self.current_step:
            self.current_step.output_generated = output_generated
            self.current_step.execution_time = execution_time

    def end_query_session(self, final_answer: str = None, success: bool = True,
                         error_messages: list[str] = None):
        """End the current query logging session."""
        if not self.current_log:
            return

        self.current_log.end_time = datetime.now().isoformat()
        self.current_log.final_answer = final_answer
        self.current_log.success = success
        self.current_log.error_messages = error_messages or []

        # Calculate total execution time
        if self.current_log.start_time and self.current_log.end_time:
            start = datetime.fromisoformat(self.current_log.start_time)
            end = datetime.fromisoformat(self.current_log.end_time)
            self.current_log.total_execution_time = (end - start).total_seconds()

        # Calculate metrics
        self._calculate_metrics()

        # Save log
        self._save_log()

        # Add to session logs
        self.session_logs.append(self.current_log)

        self.logger.info(f"📊 Ended query logging session: {self.current_log.query_id}")

        # Reset current tracking
        self.current_log = None
        self.current_step = None
        self.step_counter = 0

    def _calculate_metrics(self):
        """Calculate execution metrics."""
        if not self.current_log:
            return

        log = self.current_log

        metrics = {
            "total_steps": len(log.execution_steps),
            "total_tools_executed": len(log.tools_executed),
            "unique_tools_count": len(log.unique_tools),
            "total_agent_switches": len(log.agent_switches),
            "models_used_count": len(log.models_used),
            "total_context_chars": sum(ctx.size_chars for ctx in log.total_context_used),
            "context_sources": list({ctx.source for ctx in log.total_context_used}),
            "average_step_time": log.total_execution_time / max(len(log.execution_steps), 1),
            "total_prompt_decorations": len(log.prompt_decorations),
            "context_evolution_points": len(log.context_evolution),
            "successful_tool_executions": sum(1 for tool in log.tools_executed if tool.success),
            "failed_tool_executions": sum(1 for tool in log.tools_executed if not tool.success),
            "successful_agent_switches": sum(1 for switch in log.agent_switches if switch.success),
            "failed_agent_switches": sum(1 for switch in log.agent_switches if not switch.success)
        }

        log.metrics = metrics

    def _save_log(self):
        """Save the current log to file."""
        if not self.current_log:
            return

        # Create filename with timestamp and query ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"query_log_{timestamp}_{self.current_log.query_id}.json"
        filepath = self.log_directory / filename

        try:
            # Convert to dictionary and save
            log_dict = asdict(self.current_log)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_dict, f, indent=2, ensure_ascii=False)

            self.logger.info(f"📝 Saved query log to: {filepath}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save query log: {e}")

    def get_session_summary(self) -> dict[str, Any]:
        """Get summary of all sessions in current instance."""
        if not self.session_logs:
            return {"message": "No completed sessions"}

        total_queries = len(self.session_logs)
        successful_queries = sum(1 for log in self.session_logs if log.success)

        # Aggregate metrics
        total_execution_time = sum(log.total_execution_time for log in self.session_logs)
        total_steps = sum(len(log.execution_steps) for log in self.session_logs)
        total_tools = sum(len(log.tools_executed) for log in self.session_logs)

        # Most used tools and models
        all_tools = []
        all_models = []
        for log in self.session_logs:
            all_tools.extend(log.unique_tools)
            all_models.extend(log.models_used)

        tool_usage = {}
        for tool in all_tools:
            tool_usage[tool] = tool_usage.get(tool, 0) + 1

        model_usage = {}
        for model in all_models:
            model_usage[model] = model_usage.get(model, 0) + 1

        return {
            "session_summary": {
                "total_queries": total_queries,
                "successful_queries": successful_queries,
                "success_rate": successful_queries / total_queries if total_queries > 0 else 0,
                "total_execution_time": total_execution_time,
                "average_query_time": total_execution_time / total_queries if total_queries > 0 else 0,
                "total_steps": total_steps,
                "average_steps_per_query": total_steps / total_queries if total_queries > 0 else 0,
                "total_tools_executed": total_tools,
                "most_used_tools": sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:5],
                "most_used_models": sorted(model_usage.items(), key=lambda x: x[1], reverse=True)[:5]
            }
        }

    def create_execution_tree_visualization(self, query_id: str = None) -> str:
        """Create a text visualization of the execution tree."""
        log = self.current_log
        if query_id:
            # Find log by ID
            log = next((log_entry for log_entry in self.session_logs if log_entry.query_id == query_id), None)

        if not log:
            return "No log found for visualization"

        tree = []
        tree.append(f"🌳 Execution Tree for Query: {log.query_id}")
        tree.append(f"📝 Original Query: {log.original_query}")
        tree.append(f"🕐 Duration: {log.total_execution_time:.2f}s")
        tree.append("")

        for step in log.execution_steps:
            tree.append(f"📍 Step {step.step_number}: {step.step_type}")
            tree.append(f"   🤖 Agent: {step.agent_used}")
            tree.append(f"   📄 Description: {step.description}")

            if step.tools_executed:
                tree.append("   🔧 Tools executed:")
                for tool in step.tools_executed:
                    status = "✅" if tool.success else "❌"
                    tree.append(f"      {status} {tool.tool_name} ({tool.execution_time:.3f}s)")

            if step.agent_switches:
                tree.append("   🔄 Agent switches:")
                for switch in step.agent_switches:
                    status = "✅" if switch.success else "❌"
                    tree.append(f"      {status} {switch.from_agent} → {switch.to_agent}")

            if step.context_used:
                total_context = sum(ctx.size_chars for ctx in step.context_used)
                tree.append(f"   📊 Context used: {total_context} chars from {len(step.context_used)} sources")

            tree.append("")

        return "\n".join(tree)


# Global logger instance
_global_query_logger: QueryLogger | None = None


def get_query_logger() -> QueryLogger:
    """Get the global query logger instance."""
    global _global_query_logger
    if _global_query_logger is None:
        _global_query_logger = QueryLogger()
    return _global_query_logger


def log_query_start(query: str, execution_mode: str = "unknown") -> str:
    """Convenience function to start query logging."""
    logger = get_query_logger()
    return logger.start_query_session(query, execution_mode)


def log_query_end(final_answer: str = None, success: bool = True, error_messages: list[str] = None):
    """Convenience function to end query logging."""
    logger = get_query_logger()
    logger.end_query_session(final_answer, success, error_messages)


__all__ = [
    "QueryLogger", "QueryExecutionLog", "ExecutionStep", "ContextInfo",
    "PromptDecoration", "AgentSwitch", "ToolExecution",
    "get_query_logger", "log_query_start", "log_query_end"
]
