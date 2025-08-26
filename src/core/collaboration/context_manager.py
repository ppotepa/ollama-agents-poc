"""Context management for collaborative execution sessions."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .execution_tree import ExecutionNode


@dataclass
class CollaborationContext:
    """Context shared between agents during collaboration."""
    original_query: str
    current_step: int
    max_steps: int
    execution_tree: ExecutionNode
    discovered_files: List[str] = field(default_factory=list)
    executed_commands: List[str] = field(default_factory=list)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    working_directory: str = "."
    
    def add_discovered_file(self, file_path: str) -> None:
        """Add a discovered file to the context."""
        if file_path not in self.discovered_files:
            self.discovered_files.append(file_path)
    
    def add_executed_command(self, command: str) -> None:
        """Add an executed command to the context."""
        if command not in self.executed_commands:
            self.executed_commands.append(command)
    
    def set_intermediate_result(self, key: str, value: Any) -> None:
        """Set an intermediate result."""
        self.intermediate_results[key] = value
    
    def get_intermediate_result(self, key: str, default: Any = None) -> Any:
        """Get an intermediate result."""
        return self.intermediate_results.get(key, default)


class ContextManager:
    """Manager for collaboration context operations."""

    def __init__(self, context: CollaborationContext):
        """Initialize with a collaboration context."""
        self.context = context

    def update_step(self, step: int) -> None:
        """Update the current step."""
        self.context.current_step = step

    def is_max_steps_reached(self) -> bool:
        """Check if maximum steps have been reached."""
        return self.context.current_step >= self.context.max_steps

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of the current progress."""
        return {
            "current_step": self.context.current_step,
            "max_steps": self.context.max_steps,
            "progress_percentage": (self.context.current_step / self.context.max_steps) * 100,
            "discovered_files_count": len(self.context.discovered_files),
            "executed_commands_count": len(self.context.executed_commands),
            "intermediate_results_count": len(self.context.intermediate_results),
            "working_directory": self.context.working_directory
        }

    def get_file_discovery_summary(self) -> Dict[str, Any]:
        """Get a summary of discovered files."""
        if not self.context.discovered_files:
            return {"total_files": 0, "file_types": {}, "sample_files": []}

        # Analyze file types
        file_types = {}
        for file_path in self.context.discovered_files:
            if '.' in file_path:
                ext = file_path.split('.')[-1].lower()
                file_types[ext] = file_types.get(ext, 0) + 1
            else:
                file_types['no_extension'] = file_types.get('no_extension', 0) + 1

        return {
            "total_files": len(self.context.discovered_files),
            "file_types": file_types,
            "sample_files": self.context.discovered_files[:10]  # First 10 files
        }

    def get_command_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of executed commands."""
        if not self.context.executed_commands:
            return {"total_commands": 0, "command_types": {}, "sample_commands": []}

        # Analyze command types
        command_types = {}
        for command in self.context.executed_commands:
            cmd_type = command.split()[0] if command.split() else "unknown"
            command_types[cmd_type] = command_types.get(cmd_type, 0) + 1

        return {
            "total_commands": len(self.context.executed_commands),
            "command_types": command_types,
            "sample_commands": self.context.executed_commands[:5]  # First 5 commands
        }

    def should_continue_execution(self) -> bool:
        """Determine if execution should continue based on context."""
        # Stop if max steps reached
        if self.is_max_steps_reached():
            return False

        # Continue if we haven't discovered any files yet (early stage)
        if not self.context.discovered_files and self.context.current_step < 3:
            return True

        # Continue if we haven't executed any commands yet
        if not self.context.executed_commands and self.context.current_step < 4:
            return True

        # Stop if we have good results and are past step 3
        if (len(self.context.discovered_files) > 5 and 
            len(self.context.executed_commands) > 2 and 
            self.context.current_step >= 3):
            return False

        return True

    def get_context_state_summary(self) -> str:
        """Get a human-readable summary of the current context state."""
        summary_parts = [
            f"Step {self.context.current_step}/{self.context.max_steps}",
            f"Query: {self.context.original_query[:50]}{'...' if len(self.context.original_query) > 50 else ''}",
            f"Working directory: {self.context.working_directory}",
            f"Discovered files: {len(self.context.discovered_files)}",
            f"Executed commands: {len(self.context.executed_commands)}",
            f"Intermediate results: {len(self.context.intermediate_results)}"
        ]
        
        if self.context.discovered_files:
            summary_parts.append(f"Recent files: {', '.join(self.context.discovered_files[:3])}")
        
        if self.context.executed_commands:
            summary_parts.append(f"Recent commands: {', '.join(self.context.executed_commands[-2:])}")

        return "\n".join(summary_parts)

    def export_context_state(self) -> Dict[str, Any]:
        """Export the complete context state."""
        return {
            "original_query": self.context.original_query,
            "current_step": self.context.current_step,
            "max_steps": self.context.max_steps,
            "working_directory": self.context.working_directory,
            "discovered_files": self.context.discovered_files.copy(),
            "executed_commands": self.context.executed_commands.copy(),
            "intermediate_results": self.context.intermediate_results.copy(),
            "progress_summary": self.get_progress_summary(),
            "file_discovery_summary": self.get_file_discovery_summary(),
            "command_execution_summary": self.get_command_execution_summary()
        }

    def merge_context_results(self, other_results: Dict[str, Any]) -> None:
        """Merge results from another context or execution."""
        if "discovered_files" in other_results:
            for file_path in other_results["discovered_files"]:
                self.context.add_discovered_file(file_path)
        
        if "executed_commands" in other_results:
            for command in other_results["executed_commands"]:
                self.context.add_executed_command(command)
        
        if "intermediate_results" in other_results:
            for key, value in other_results["intermediate_results"].items():
                self.context.set_intermediate_result(key, value)
