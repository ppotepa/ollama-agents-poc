"""Command generation for enhancing prompts with context information."""

import os
from typing import List, Dict, Optional, Tuple

from .context_analyzer import PromptContext
from .pattern_matcher import PromptType


class CommandGenerator:
    """Generates commands to gather context information for prompts."""

    def __init__(self):
        """Initialize the command generator."""
        self.command_templates = self._initialize_command_templates()

    def _initialize_command_templates(self) -> Dict[str, Dict]:
        """Initialize command templates for different context types."""
        return {
            "list_files": {
                "command": "find {directory} -type f -name '*.{extensions}' | head -20",
                "description": "List files in directory",
                "extensions": ["py", "js", "ts", "java", "cpp", "h", "md", "txt", "json", "yaml", "yml"],
                "timeout": 10
            },
            "analyze_repository_context": {
                "command": "ls -la {directory} && find {directory} -maxdepth 2 -type d",
                "description": "Analyze repository structure",
                "timeout": 15
            },
            "get_file_content": {
                "command": "head -50 {filepath}",
                "description": "Get file content preview",
                "timeout": 5
            },
            "analyze_repo_languages": {
                "command": "find {directory} -name '*.py' -o -name '*.js' -o -name '*.ts' -o -name '*.java' -o -name '*.cpp' -o -name '*.h' | xargs wc -l | sort -nr | head -10",
                "description": "Analyze programming languages",
                "timeout": 20
            },
            "analyze_technologies": {
                "command": "find {directory} -name 'package.json' -o -name 'requirements.txt' -o -name 'pom.xml' -o -name 'Cargo.toml' -o -name 'go.mod' | xargs cat",
                "description": "Analyze technologies and dependencies",
                "timeout": 15
            },
            "analyze_dependencies": {
                "command": "find {directory} -name 'requirements.txt' -o -name 'package.json' -o -name 'Pipfile' | head -5 | xargs cat",
                "description": "Analyze project dependencies",
                "timeout": 10
            },
            "analyze_code_structure": {
                "command": "find {directory} -name '*.py' | xargs grep -l 'class \\|def ' | head -10",
                "description": "Analyze code structure",
                "timeout": 15
            },
            "analyze_repo_directories": {
                "command": "find {directory} -maxdepth 3 -type d | sort",
                "description": "Analyze directory structure",
                "timeout": 10
            },
            "find_config_files": {
                "command": "find {directory} -name '*.config' -o -name '*.conf' -o -name '*.ini' -o -name '*.yaml' -o -name '*.yml' -o -name '*.json' | grep -v node_modules | head -10",
                "description": "Find configuration files",
                "timeout": 10
            },
            "get_recent_logs": {
                "command": "find {directory} -name '*.log' -mtime -7 | head -5 | xargs tail -20",
                "description": "Get recent log entries",
                "timeout": 15
            },
            "get_git_status": {
                "command": "cd {directory} && git status --porcelain && git log --oneline -5",
                "description": "Get git status and recent commits",
                "timeout": 10
            },
            "get_recent_commits": {
                "command": "cd {directory} && git log --oneline --since='7 days ago'",
                "description": "Get recent commits",
                "timeout": 10
            },
            "get_repository_state": {
                "command": "ls -la {directory} && pwd && echo 'Files:' && find {directory} -maxdepth 1 -type f | wc -l && echo 'Directories:' && find {directory} -maxdepth 1 -type d | wc -l",
                "description": "Get basic repository state",
                "timeout": 10
            }
        }

    def generate_context_commands(self, context: PromptContext) -> List[Dict[str, str]]:
        """Generate commands to gather context information.
        
        Args:
            context: The prompt context requiring information
            
        Returns:
            List of command dictionaries with command, description, and metadata
        """
        commands = []
        
        for command_name in context.supplemental_commands:
            if command_name in self.command_templates:
                template = self.command_templates[command_name]
                
                # Generate the actual command
                command = self._format_command(template, context.working_directory)
                
                if command:
                    commands.append({
                        "name": command_name,
                        "command": command,
                        "description": template["description"],
                        "timeout": template.get("timeout", 15),
                        "directory": context.working_directory
                    })
        
        return commands

    def _format_command(self, template: Dict, directory: str) -> Optional[str]:
        """Format a command template with the given directory.
        
        Args:
            template: Command template dictionary
            directory: Working directory
            
        Returns:
            Formatted command string or None if formatting fails
        """
        try:
            command = template["command"]
            
            # Replace directory placeholder
            command = command.replace("{directory}", directory)
            
            # Replace extensions placeholder if present
            if "{extensions}" in command:
                extensions = "|".join(template.get("extensions", ["*"]))
                command = command.replace("{extensions}", extensions)
            
            # Handle filepath placeholder (requires specific file)
            if "{filepath}" in command:
                # For file content commands, we'll need to modify this based on context
                # For now, skip these commands
                return None
            
            return command
            
        except Exception:
            return None

    def optimize_command_sequence(self, commands: List[Dict[str, str]], prompt_type: PromptType) -> List[Dict[str, str]]:
        """Optimize the sequence of commands for efficiency.
        
        Args:
            commands: List of commands to optimize
            prompt_type: Type of prompt being processed
            
        Returns:
            Optimized command sequence
        """
        # Define command priorities for different prompt types
        priority_orders = {
            PromptType.REPOSITORY_ANALYSIS: [
                "get_repository_state",
                "analyze_repository_context", 
                "analyze_repo_directories",
                "analyze_repo_languages",
                "analyze_technologies"
            ],
            PromptType.CODE_ANALYSIS: [
                "analyze_code_structure",
                "get_repository_state",
                "analyze_repo_languages",
                "list_files"
            ],
            PromptType.FILE_OPERATIONS: [
                "get_repository_state",
                "list_files",
                "analyze_repo_directories"
            ],
            PromptType.DEBUGGING: [
                "get_recent_logs",
                "get_git_status",
                "get_repository_state",
                "analyze_code_structure"
            ],
            PromptType.TECHNOLOGY_INQUIRY: [
                "analyze_technologies",
                "analyze_dependencies",
                "find_config_files",
                "get_repository_state"
            ]
        }
        
        # Get priority order for this prompt type
        priority_order = priority_orders.get(prompt_type, [])
        
        # Sort commands based on priority
        prioritized = []
        remaining = commands.copy()
        
        # Add commands in priority order
        for priority_cmd in priority_order:
            for cmd in remaining:
                if cmd["name"] == priority_cmd:
                    prioritized.append(cmd)
                    remaining.remove(cmd)
                    break
        
        # Add any remaining commands
        prioritized.extend(remaining)
        
        return prioritized

    def estimate_execution_time(self, commands: List[Dict[str, str]]) -> int:
        """Estimate total execution time for commands.
        
        Args:
            commands: List of commands to execute
            
        Returns:
            Estimated execution time in seconds
        """
        total_time = 0
        for cmd in commands:
            total_time += cmd.get("timeout", 15)
        
        # Add overhead for command switching and processing
        overhead = len(commands) * 2
        
        return total_time + overhead

    def generate_parallel_groups(self, commands: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
        """Group commands that can be executed in parallel.
        
        Args:
            commands: List of commands to group
            
        Returns:
            List of command groups that can be executed in parallel
        """
        # Commands that can run in parallel (don't interfere with each other)
        parallel_safe = {
            "get_repository_state",
            "analyze_repo_directories", 
            "list_files",
            "find_config_files",
            "analyze_repo_languages"
        }
        
        # Commands that should run sequentially
        sequential = {
            "get_git_status",
            "get_recent_commits",
            "get_recent_logs",
            "analyze_code_structure"
        }
        
        groups = []
        current_parallel_group = []
        
        for cmd in commands:
            cmd_name = cmd["name"]
            
            if cmd_name in parallel_safe and len(current_parallel_group) < 3:
                # Add to current parallel group (max 3 parallel commands)
                current_parallel_group.append(cmd)
            else:
                # Finish current parallel group if it exists
                if current_parallel_group:
                    groups.append(current_parallel_group)
                    current_parallel_group = []
                
                # Add sequential command as its own group
                groups.append([cmd])
        
        # Add final parallel group if it exists
        if current_parallel_group:
            groups.append(current_parallel_group)
        
        return groups

    def validate_commands(self, commands: List[Dict[str, str]], directory: str) -> Tuple[List[Dict[str, str]], List[str]]:
        """Validate commands and return valid commands and error messages.
        
        Args:
            commands: List of commands to validate
            directory: Working directory
            
        Returns:
            Tuple of (valid_commands, error_messages)
        """
        valid_commands = []
        errors = []
        
        # Check if directory exists
        if not os.path.exists(directory):
            errors.append(f"Directory does not exist: {directory}")
            return [], errors
        
        # Check if directory is readable
        if not os.access(directory, os.R_OK):
            errors.append(f"Directory is not readable: {directory}")
            return [], errors
        
        for cmd in commands:
            try:
                # Basic validation of command structure
                if not cmd.get("command"):
                    errors.append(f"Command '{cmd.get('name', 'unknown')}' has no command string")
                    continue
                
                if not cmd.get("name"):
                    errors.append("Command has no name")
                    continue
                
                # Check for potentially dangerous commands
                dangerous_patterns = ["rm ", "del ", "format", "shutdown", "reboot"]
                command_lower = cmd["command"].lower()
                
                if any(pattern in command_lower for pattern in dangerous_patterns):
                    errors.append(f"Command '{cmd['name']}' contains potentially dangerous operations")
                    continue
                
                valid_commands.append(cmd)
                
            except Exception as e:
                errors.append(f"Error validating command '{cmd.get('name', 'unknown')}': {str(e)}")
        
        return valid_commands, errors
