"""Command execution and management for collaborative system."""

import json
import subprocess
import time
from typing import Any, Dict, List, Tuple

from src.agents.interceptor.agent import CommandRecommendation
from src.tools.file_ops import get_file_info, list_files, list_files_recurse, read_file
from .context_manager import CollaborationContext


class CommandExecutor:
    """Handles safe execution of commands during collaborative sessions."""

    def __init__(self):
        """Initialize the command executor."""
        self.execution_history = []
        self.safety_limits = {
            "max_execution_time": 60.0,  # seconds
            "max_file_size": 10 * 1024 * 1024,  # 10MB
            "max_recursive_depth": 5,
            "max_files_per_operation": 100
        }

    def execute_command_safely(self, command: str, args: Dict[str, Any]) -> Tuple[bool, str, float]:
        """Execute a command safely with proper error handling and timeouts.
        
        Args:
            command: The command to execute
            args: Command arguments
            
        Returns:
            Tuple of (success, result, execution_time)
        """
        start_time = time.time()
        
        try:
            # Route to appropriate handler based on command
            if command == "list_files":
                success, result = self._execute_list_files(args)
            elif command == "list_files_recurse":
                success, result = self._execute_list_files_recurse(args)
            elif command == "read_file":
                success, result = self._execute_read_file(args)
            elif command == "get_file_info":
                success, result = self._execute_get_file_info(args)
            elif command == "analyze_repository_context":
                success, result = self._execute_analyze_repository_context(args)
            elif command == "execute_shell_command":
                success, result = self._execute_shell_command(args)
            else:
                success = False
                result = f"Unknown command: {command}"

            execution_time = time.time() - start_time
            
            # Log execution
            self.execution_history.append({
                "command": command,
                "args": args,
                "success": success,
                "execution_time": execution_time,
                "timestamp": time.time(),
                "result_length": len(str(result))
            })

            return success, result, execution_time

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Error executing {command}: {str(e)}"
            
            self.execution_history.append({
                "command": command,
                "args": args,
                "success": False,
                "execution_time": execution_time,
                "timestamp": time.time(),
                "error": error_msg
            })
            
            return False, error_msg, execution_time

    def _execute_list_files(self, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Execute list_files command."""
        try:
            path = args.get("path", ".")
            pattern = args.get("pattern", "*")
            
            files = list_files(path, pattern)
            
            # Apply safety limits
            if len(files) > self.safety_limits["max_files_per_operation"]:
                files = files[:self.safety_limits["max_files_per_operation"]]
                files.append(f"... (truncated to {self.safety_limits['max_files_per_operation']} files)")
            
            result = json.dumps(files, indent=2)
            return True, result
            
        except Exception as e:
            return False, f"Error listing files: {str(e)}"

    def _execute_list_files_recurse(self, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Execute list_files_recurse command."""
        try:
            path = args.get("path", ".")
            max_depth = min(args.get("max_depth", 3), self.safety_limits["max_recursive_depth"])
            
            files = list_files_recurse(path, max_depth=max_depth)
            
            # Apply safety limits
            if len(files) > self.safety_limits["max_files_per_operation"]:
                files = files[:self.safety_limits["max_files_per_operation"]]
                files.append(f"... (truncated to {self.safety_limits['max_files_per_operation']} files)")
            
            result = json.dumps(files, indent=2)
            return True, result
            
        except Exception as e:
            return False, f"Error listing files recursively: {str(e)}"

    def _execute_read_file(self, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Execute read_file command."""
        try:
            filepath = args.get("filepath")
            if not filepath:
                return False, "Missing filepath argument"
            
            max_lines = args.get("max_lines", 100)
            start_line = args.get("start_line", 1)
            
            content = read_file(filepath, max_lines=max_lines, start_line=start_line)
            
            # Check file size safety limit
            if len(content) > self.safety_limits["max_file_size"]:
                content = content[:self.safety_limits["max_file_size"]] + "\n... (truncated due to size limit)"
            
            return True, content
            
        except Exception as e:
            return False, f"Error reading file: {str(e)}"

    def _execute_get_file_info(self, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Execute get_file_info command."""
        try:
            filepath = args.get("filepath")
            if not filepath:
                return False, "Missing filepath argument"
            
            info = get_file_info(filepath)
            result = json.dumps(info, indent=2, default=str)
            return True, result
            
        except Exception as e:
            return False, f"Error getting file info: {str(e)}"

    def _execute_analyze_repository_context(self, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Execute repository context analysis."""
        try:
            path = args.get("path", ".")
            
            # Get basic repository structure
            context = {
                "files": list_files(path),
                "recursive_structure": list_files_recurse(path, max_depth=2),
                "directory_info": {
                    "path": path,
                    "analysis_timestamp": time.time()
                }
            }
            
            # Look for common project files
            project_indicators = [
                "package.json", "requirements.txt", "Cargo.toml", "go.mod",
                "pom.xml", "build.gradle", "setup.py", "pyproject.toml",
                "Gemfile", "composer.json", ".gitignore", "README.md"
            ]
            
            found_indicators = []
            for indicator in project_indicators:
                try:
                    if indicator in context["files"]:
                        found_indicators.append(indicator)
                except Exception:
                    pass
            
            context["project_indicators"] = found_indicators
            
            result = json.dumps(context, indent=2)
            return True, result
            
        except Exception as e:
            return False, f"Error analyzing repository context: {str(e)}"

    def _execute_shell_command(self, args: Dict[str, Any]) -> Tuple[bool, str]:
        """Execute a shell command safely."""
        try:
            command = args.get("command")
            if not command:
                return False, "Missing command argument"
            
            cwd = args.get("cwd", ".")
            timeout = min(args.get("timeout", 30), self.safety_limits["max_execution_time"])
            
            # Basic safety checks
            dangerous_commands = ["rm", "del", "format", "shutdown", "reboot", "kill"]
            command_lower = command.lower()
            
            if any(dangerous_cmd in command_lower for dangerous_cmd in dangerous_commands):
                return False, f"Potentially dangerous command blocked: {command}"
            
            # Execute the command
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                output = result.stdout.strip()
                if len(output) > self.safety_limits["max_file_size"]:
                    output = output[:self.safety_limits["max_file_size"]] + "\n... (truncated due to size limit)"
                return True, output
            else:
                error_output = result.stderr.strip()
                return False, f"Command failed (exit code {result.returncode}): {error_output}"
                
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout} seconds"
        except Exception as e:
            return False, f"Error executing shell command: {str(e)}"

    def determine_command_args(self, recommendation: CommandRecommendation, context: CollaborationContext) -> Dict[str, Any]:
        """Determine appropriate arguments for a command based on context.
        
        Args:
            recommendation: The command recommendation
            context: Current collaboration context
            
        Returns:
            Dictionary of command arguments
        """
        args = {}
        command = recommendation.command

        if command == "list_files":
            args["path"] = context.working_directory
            args["pattern"] = "*"

        elif command == "list_files_recurse":
            args["path"] = context.working_directory
            args["max_depth"] = 3

        elif command == "read_file":
            # Try to determine which file to read based on context
            if context.discovered_files:
                # Read the most recently discovered file
                args["filepath"] = context.discovered_files[-1]
            else:
                # Default to common project files
                common_files = ["README.md", "package.json", "requirements.txt", "main.py"]
                for file in common_files:
                    try:
                        # Check if file exists in working directory
                        import os
                        full_path = os.path.join(context.working_directory, file)
                        if os.path.exists(full_path):
                            args["filepath"] = full_path
                            break
                    except Exception:
                        continue
            
            args["max_lines"] = 50

        elif command == "get_file_info":
            if context.discovered_files:
                args["filepath"] = context.discovered_files[-1]

        elif command == "analyze_repository_context":
            args["path"] = context.working_directory

        elif command == "execute_shell_command":
            # This would need more sophisticated logic based on the recommendation
            args["command"] = "echo 'No specific command provided'"
            args["cwd"] = context.working_directory
            args["timeout"] = 30

        return args

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get statistics about command execution."""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0,
                "command_distribution": {}
            }

        successful = [e for e in self.execution_history if e["success"]]
        total = len(self.execution_history)
        
        # Calculate command distribution
        command_counts = {}
        for execution in self.execution_history:
            cmd = execution["command"]
            command_counts[cmd] = command_counts.get(cmd, 0) + 1

        # Calculate average execution time
        execution_times = [e["execution_time"] for e in self.execution_history]
        avg_time = sum(execution_times) / len(execution_times) if execution_times else 0.0

        return {
            "total_executions": total,
            "successful_executions": len(successful),
            "success_rate": len(successful) / total if total > 0 else 0.0,
            "average_execution_time": avg_time,
            "total_execution_time": sum(execution_times),
            "command_distribution": command_counts,
            "recent_executions": self.execution_history[-5:]  # Last 5 executions
        }

    def clear_execution_history(self) -> None:
        """Clear the execution history."""
        self.execution_history.clear()
