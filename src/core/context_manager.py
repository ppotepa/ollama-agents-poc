"""Context Manager - Manages persistent state and context transfer between model swaps."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.enhanced_logging import get_logger


@dataclass
class ContextEntry:
    """Represents a single context entry."""
    key: str
    value: Any
    timestamp: float
    source: str  # Which step/model created this context
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp,
            "source": self.source,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextEntry:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ExecutionContext:
    """Persistent context for execution across model swaps."""
    session_id: str
    original_query: str
    current_step: str
    entries: dict[str, ContextEntry] = field(default_factory=dict)
    execution_history: list[dict[str, Any]] = field(default_factory=list)
    model_history: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def add_entry(self, key: str, value: Any, source: str, metadata: dict[str, Any] | None = None) -> None:
        """Add or update a context entry."""
        entry = ContextEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            source=source,
            metadata=metadata or {}
        )
        self.entries[key] = entry
        self.updated_at = time.time()

    def get_entry(self, key: str) -> ContextEntry | None:
        """Get a context entry by key."""
        return self.entries.get(key)

    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a context value by key."""
        entry = self.entries.get(key)
        return entry.value if entry else default

    def remove_entry(self, key: str) -> bool:
        """Remove a context entry."""
        if key in self.entries:
            del self.entries[key]
            self.updated_at = time.time()
            return True
        return False

    def get_entries_by_source(self, source: str) -> list[ContextEntry]:
        """Get all entries created by a specific source."""
        return [entry for entry in self.entries.values() if entry.source == source]

    def add_execution_record(self, step_id: str, model: str, result: str,
                           duration: float, metadata: dict[str, Any] | None = None) -> None:
        """Add an execution record to history."""
        record = {
            "step_id": step_id,
            "model": model,
            "result": result,
            "duration": duration,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self.execution_history.append(record)

        # Track model usage
        if model not in self.model_history:
            self.model_history.append(model)

        self.updated_at = time.time()

    def get_relevant_context(self, max_entries: int = 10) -> dict[str, Any]:
        """Get the most relevant context for current execution."""
        # Sort entries by timestamp (most recent first)
        sorted_entries = sorted(self.entries.values(), key=lambda x: x.timestamp, reverse=True)

        # Take the most recent entries
        relevant_entries = sorted_entries[:max_entries]

        return {
            "session_id": self.session_id,
            "original_query": self.original_query,
            "current_step": self.current_step,
            "context_entries": {entry.key: entry.value for entry in relevant_entries},
            "recent_execution_history": self.execution_history[-5:],  # Last 5 executions
            "models_used": self.model_history
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "original_query": self.original_query,
            "current_step": self.current_step,
            "entries": {k: v.to_dict() for k, v in self.entries.items()},
            "execution_history": self.execution_history,
            "model_history": self.model_history,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionContext:
        """Create from dictionary."""
        entries = {k: ContextEntry.from_dict(v) for k, v in data.get("entries", {}).items()}

        return cls(
            session_id=data["session_id"],
            original_query=data["original_query"],
            current_step=data["current_step"],
            entries=entries,
            execution_history=data.get("execution_history", []),
            model_history=data.get("model_history", []),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time())
        )


class ContextManager:
    """Manages persistent context and state across model swaps."""

    def __init__(self, persistence_dir: str | None = None):
        """Initialize the context manager.

        Args:
            persistence_dir: Directory to persist context data (optional)
        """
        self.logger = get_logger()
        self.persistence_dir = Path(persistence_dir) if persistence_dir else None
        self.active_contexts: dict[str, ExecutionContext] = {}

        if self.persistence_dir:
            self.persistence_dir.mkdir(parents=True, exist_ok=True)

    def create_context(self, session_id: str, original_query: str,
                      initial_step: str = "init") -> ExecutionContext:
        """Create a new execution context.

        Args:
            session_id: Unique identifier for the session
            original_query: The original user query
            initial_step: Initial step identifier

        Returns:
            New ExecutionContext instance
        """
        context = ExecutionContext(
            session_id=session_id,
            original_query=original_query,
            current_step=initial_step
        )

        self.active_contexts[session_id] = context
        self.logger.info(f"Created new context for session {session_id}")

        return context

    def get_context(self, session_id: str) -> ExecutionContext | None:
        """Get an existing context by session ID."""
        context = self.active_contexts.get(session_id)
        if not context and self.persistence_dir:
            # Try to load from persistence
            context = self._load_context(session_id)
            if context:
                self.active_contexts[session_id] = context

        return context

    def update_context_step(self, session_id: str, new_step: str) -> bool:
        """Update the current step in a context."""
        context = self.get_context(session_id)
        if context:
            context.current_step = new_step
            context.updated_at = time.time()
            self.logger.debug(f"Updated context {session_id} to step {new_step}")
            return True
        return False

    def add_context_data(self, session_id: str, key: str, value: Any,
                        source: str, metadata: dict[str, Any] | None = None) -> bool:
        """Add data to a context.

        Args:
            session_id: Session identifier
            key: Context key
            value: Context value
            source: Source that created this context (step_id, model_name, etc.)
            metadata: Optional metadata

        Returns:
            True if successful, False otherwise
        """
        context = self.get_context(session_id)
        if context:
            context.add_entry(key, value, source, metadata)
            self.logger.debug(f"Added context data {key} to session {session_id}")
            return True
        return False

    def get_context_data(self, session_id: str, key: str, default: Any = None) -> Any:
        """Get context data by key."""
        context = self.get_context(session_id)
        return context.get_value(key, default) if context else default

    def record_execution(self, session_id: str, step_id: str, model: str,
                        result: str, duration: float,
                        metadata: dict[str, Any] | None = None) -> bool:
        """Record an execution in the context history."""
        context = self.get_context(session_id)
        if context:
            context.add_execution_record(step_id, model, result, duration, metadata)
            self.logger.debug(f"Recorded execution {step_id} in session {session_id}")
            return True
        return False

    def prepare_model_transition_context(self, session_id: str,
                                       from_model: str, to_model: str) -> dict[str, Any] | None:
        """Prepare context for transitioning between models.

        Args:
            session_id: Session identifier
            from_model: Model being transitioned from
            to_model: Model being transitioned to

        Returns:
            Context dictionary for the new model
        """
        context = self.get_context(session_id)
        if not context:
            return None

        # Get relevant context for the transition
        transition_context = context.get_relevant_context()

        # Add transition metadata
        transition_context.update({
            "model_transition": {
                "from_model": from_model,
                "to_model": to_model,
                "transition_time": time.time()
            },
            "context_summary": self._create_context_summary(context)
        })

        self.logger.info(f"Prepared transition context from {from_model} to {to_model} "
                        f"for session {session_id}")

        return transition_context

    def _create_context_summary(self, context: ExecutionContext) -> str:
        """Create a human-readable context summary."""
        summary_parts = [
            f"Session: {context.session_id}",
            f"Original Query: {context.original_query}",
            f"Current Step: {context.current_step}",
            f"Models Used: {', '.join(context.model_history)}",
            f"Execution Steps: {len(context.execution_history)}"
        ]

        # Add key context entries
        if context.entries:
            important_keys = ["repository_structure", "file_analysis", "implementation_plan",
                            "discovered_files", "error_states", "progress_status"]
            for key in important_keys:
                if key in context.entries:
                    entry = context.entries[key]
                    summary_parts.append(f"{key}: {str(entry.value)[:100]}...")

        return "\n".join(summary_parts)

    def save_context(self, session_id: str) -> bool:
        """Save context to persistence storage."""
        if not self.persistence_dir:
            return False

        context = self.get_context(session_id)
        if not context:
            return False

        try:
            file_path = self.persistence_dir / f"context_{session_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(context.to_dict(), f, indent=2, default=str)

            self.logger.debug(f"Saved context {session_id} to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save context {session_id}: {e}")
            return False

    def _load_context(self, session_id: str) -> ExecutionContext | None:
        """Load context from persistence storage."""
        if not self.persistence_dir:
            return None

        try:
            file_path = self.persistence_dir / f"context_{session_id}.json"
            if not file_path.exists():
                return None

            with open(file_path, encoding='utf-8') as f:
                data = json.load(f)

            context = ExecutionContext.from_dict(data)
            self.logger.debug(f"Loaded context {session_id} from {file_path}")
            return context
        except Exception as e:
            self.logger.error(f"Failed to load context {session_id}: {e}")
            return None

    def cleanup_old_contexts(self, max_age_hours: int = 24) -> int:
        """Clean up old contexts from memory and persistence.

        Args:
            max_age_hours: Maximum age in hours before cleanup

        Returns:
            Number of contexts cleaned up
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0

        # Clean up from memory
        sessions_to_remove = []
        for session_id, context in self.active_contexts.items():
            if current_time - context.updated_at > max_age_seconds:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            del self.active_contexts[session_id]
            cleaned_count += 1

        # Clean up from persistence
        if self.persistence_dir and self.persistence_dir.exists():
            for file_path in self.persistence_dir.glob("context_*.json"):
                try:
                    if current_time - file_path.stat().st_mtime > max_age_seconds:
                        file_path.unlink()
                        cleaned_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to clean up {file_path}: {e}")

        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old contexts")

        return cleaned_count

    def get_session_statistics(self, session_id: str) -> dict[str, Any] | None:
        """Get statistics for a session."""
        context = self.get_context(session_id)
        if not context:
            return None

        total_duration = sum(record.get("duration", 0) for record in context.execution_history)

        return {
            "session_id": session_id,
            "original_query": context.original_query,
            "total_steps": len(context.execution_history),
            "total_duration": total_duration,
            "models_used": len(context.model_history),
            "context_entries": len(context.entries),
            "created_at": context.created_at,
            "updated_at": context.updated_at,
            "age_hours": (time.time() - context.created_at) / 3600
        }


# Global context manager instance
_global_context_manager: ContextManager | None = None


def get_context_manager(persistence_dir: str | None = None) -> ContextManager:
    """Get the global context manager instance."""
    global _global_context_manager
    if _global_context_manager is None:
        _global_context_manager = ContextManager(persistence_dir)
    return _global_context_manager


__all__ = [
    "ContextManager", "ExecutionContext", "ContextEntry",
    "get_context_manager"
]
