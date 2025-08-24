"""Command Resolver - Cognitive interpretation of user input to map to local commands."""

import re
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import importlib.util
from dataclasses import dataclass


@dataclass
class ResolvedCommand:
    """Represents a resolved command with its parameters."""
    command_name: str
    command_module: str
    command_class: str
    parameters: Dict[str, Any]
    confidence: float
    description: str


class CommandResolver:
    """Resolves user input to appropriate commands from the commands folder."""
    
    def __init__(self, commands_path: str = "src/commands"):
        """Initialize the command resolver.
        
        Args:
            commands_path: Path to the commands folder
        """
        self.commands_path = Path(commands_path)
        self.command_patterns = self._build_command_patterns()
        self.available_commands = self._discover_commands()
    
    def _build_command_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Build patterns for recognizing different types of commands."""
        return {
            "repository_analysis": {
                "patterns": [
                    r"analyz[e|ing]*\s+repositor[y|ies]*",
                    r"repositor[y|ies]*\s+analys[is|e]*",
                    r"analyz[e|ing]*\s+project\s+structure",
                    r"project\s+structure\s+analys[is|e]*",
                    r"code\s+analys[is|e]*",
                    r"codebase\s+analys[is|e]*",
                    r"what.*language.*project",
                    r"what.*technologies.*used",
                    r"project\s+overview",
                    r"repository\s+structure",
                    r"file\s+structure",
                    r"directory\s+structure"
                ],
                "command": "repository_analysis",
                "description": "Analyze repository structure, languages, and technologies used",
                "parameters": {
                    "include_files": True,
                    "include_languages": True,
                    "include_dependencies": True,
                    "include_git_info": True
                }
            },
            "code_cleanup": {
                "patterns": [
                    r"clean\s*up\s+code",
                    r"code\s+cleanup",
                    r"refactor\s+code",
                    r"optimize\s+code",
                    r"improve\s+code\s+quality",
                    r"remove\s+unused\s+imports",
                    r"fix\s+code\s+style",
                    r"format\s+code"
                ],
                "command": "code_cleanup",
                "description": "Clean up and optimize code",
                "parameters": {
                    "remove_unused_imports": True,
                    "fix_formatting": True,
                    "optimize_structure": True
                }
            },
            "reference_removal": {
                "patterns": [
                    r"remove\s+references*",
                    r"clean\s+references*",
                    r"unused\s+references*",
                    r"dead\s+code",
                    r"remove\s+dead\s+code"
                ],
                "command": "reference_removal",
                "description": "Remove unused references and dead code",
                "parameters": {
                    "scan_imports": True,
                    "scan_variables": True,
                    "scan_functions": True
                }
            },
            "file_operations": {
                "patterns": [
                    r"create\s+file",
                    r"write\s+file",
                    r"read\s+file",
                    r"list\s+files",
                    r"find\s+files",
                    r"search\s+files"
                ],
                "command": "file_operations",
                "description": "Perform file operations",
                "parameters": {
                    "operation": "auto_detect"
                }
            }
        }
    
    def _discover_commands(self) -> List[str]:
        """Discover available commands in the commands folder."""
        commands = []
        if self.commands_path.exists():
            for item in self.commands_path.iterdir():
                if item.is_dir() and not item.name.startswith('_'):
                    commands.append(item.name)
        return commands
    
    def resolve(self, user_input: str) -> Optional[ResolvedCommand]:
        """Resolve user input to a command.
        
        Args:
            user_input: The user's input text
            
        Returns:
            ResolvedCommand if a match is found, None otherwise
        """
        user_input_lower = user_input.lower().strip()
        
        best_match = None
        highest_confidence = 0.0
        
        # Check each command pattern
        for command_type, config in self.command_patterns.items():
            confidence = self._calculate_confidence(user_input_lower, config["patterns"])
            
            if confidence > highest_confidence:
                highest_confidence = confidence
                best_match = ResolvedCommand(
                    command_name=config["command"],
                    command_module=f"src.commands.{config['command']}",
                    command_class=self._get_command_class_name(config["command"]),
                    parameters=config["parameters"].copy(),
                    confidence=confidence,
                    description=config["description"]
                )
        
        # Only return if confidence is above threshold
        if highest_confidence >= 0.3:  # 30% confidence threshold
            return best_match
        
        return None
    
    def _calculate_confidence(self, user_input: str, patterns: List[str]) -> float:
        """Calculate confidence score for pattern matching."""
        max_score = 0.0
        
        for pattern in patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                # Base score for pattern match
                score = 0.7
                
                # Boost score if it's a very specific match
                if len(pattern) > 20:  # Longer patterns are more specific
                    score += 0.2
                
                # Boost score if the pattern covers a significant portion of input
                pattern_words = len(pattern.split())
                input_words = len(user_input.split())
                if input_words > 0:
                    coverage = min(pattern_words / input_words, 1.0)
                    score += coverage * 0.1
                
                max_score = max(max_score, score)
        
        return min(max_score, 1.0)  # Cap at 1.0
    
    def _get_command_class_name(self, command_name: str) -> str:
        """Convert command name to class name."""
        # Convert snake_case to PascalCase
        parts = command_name.split('_')
        return ''.join(part.capitalize() for part in parts) + 'Command'
    
    def get_available_commands(self) -> List[str]:
        """Get list of available commands."""
        return self.available_commands.copy()
    
    def explain_resolution(self, user_input: str) -> str:
        """Explain how the input would be resolved (for debugging)."""
        resolved = self.resolve(user_input)
        
        if resolved:
            return (f"Input: '{user_input}'\n"
                   f"Resolved to: {resolved.command_name}\n"
                   f"Confidence: {resolved.confidence:.2f}\n"
                   f"Description: {resolved.description}\n"
                   f"Parameters: {resolved.parameters}")
        else:
            return f"Input: '{user_input}'\nNo command resolution found (confidence too low)"
    
    def add_custom_pattern(self, command_name: str, pattern: str, description: str, 
                          parameters: Dict[str, Any] = None) -> None:
        """Add a custom pattern for command resolution.
        
        Args:
            command_name: Name of the command
            pattern: Regex pattern to match
            description: Description of what the command does
            parameters: Default parameters for the command
        """
        if parameters is None:
            parameters = {}
        
        if command_name not in self.command_patterns:
            self.command_patterns[command_name] = {
                "patterns": [],
                "command": command_name,
                "description": description,
                "parameters": parameters
            }
        
        self.command_patterns[command_name]["patterns"].append(pattern)


# Singleton instance for global access
_resolver_instance = None


def get_command_resolver() -> CommandResolver:
    """Get the global command resolver instance."""
    global _resolver_instance
    if _resolver_instance is None:
        _resolver_instance = CommandResolver()
    return _resolver_instance


def resolve_user_input(user_input: str) -> Optional[ResolvedCommand]:
    """Convenience function to resolve user input."""
    return get_command_resolver().resolve(user_input)


if __name__ == "__main__":
    # Test the command resolver
    resolver = CommandResolver()
    
    test_inputs = [
        "analyze repository structure",
        "what language is this project using",
        "analyze the codebase",
        "clean up the code",
        "remove unused imports",
        "create a new file",
        "show me the project overview",
        "this is random text that should not match"
    ]
    
    print("🧠 Command Resolver Test\n" + "="*50)
    for test_input in test_inputs:
        print(f"\n📝 Testing: '{test_input}'")
        print(resolver.explain_resolution(test_input))
