"""Context analysis for determining what information is needed for prompts."""

import os
from dataclasses import dataclass
from typing import List, Optional

from .pattern_matcher import PromptType


@dataclass
class PromptContext:
    """Context information extracted from a prompt."""
    original_prompt: str
    prompt_type: PromptType
    confidence: float
    detected_intent: str
    required_context: List[str]
    supplemental_commands: List[str]
    working_directory: str
    has_repository: bool


class ContextAnalyzer:
    """Analyzes prompts to determine what context information is needed."""

    def __init__(self):
        """Initialize the context analyzer."""
        self.context_providers = self._initialize_context_providers()

    def _initialize_context_providers(self) -> dict[str, dict]:
        """Initialize available context providers and their capabilities."""
        return {
            "repository_structure": {
                "commands": ["list_files", "analyze_repository_context"],
                "description": "Overall repository structure and file organization",
                "weight": 1.0
            },
            "file_content": {
                "commands": ["get_file_content"],
                "description": "Content of specific files",
                "weight": 0.8
            },
            "languages": {
                "commands": ["analyze_repo_languages"],
                "description": "Programming languages used in the project",
                "weight": 0.7
            },
            "technologies": {
                "commands": ["analyze_technologies"],
                "description": "Frameworks, libraries, and technologies used",
                "weight": 0.7
            },
            "dependencies": {
                "commands": ["analyze_dependencies"],
                "description": "Project dependencies and package information",
                "weight": 0.6
            },
            "code_structure": {
                "commands": ["analyze_code_structure"],
                "description": "Code organization and architecture",
                "weight": 0.8
            },
            "file_structure": {
                "commands": ["list_files", "analyze_repo_directories"],
                "description": "File and directory structure",
                "weight": 0.6
            },
            "current_files": {
                "commands": ["list_files"],
                "description": "Current files in the working directory",
                "weight": 0.5
            },
            "config_files": {
                "commands": ["find_config_files"],
                "description": "Configuration files and settings",
                "weight": 0.6
            },
            "error_logs": {
                "commands": ["get_recent_logs"],
                "description": "Recent error logs and debug information",
                "weight": 0.7
            },
            "recent_changes": {
                "commands": ["get_git_status", "get_recent_commits"],
                "description": "Recent changes and git history",
                "weight": 0.6
            },
            "basic_context": {
                "commands": ["get_repository_state"],
                "description": "Basic repository context",
                "weight": 0.3
            }
        }

    def analyze_prompt_context(self, prompt: str, working_directory: Optional[str] = None) -> PromptContext:
        """Analyze a prompt and determine what context is needed.
        
        Args:
            prompt: The user prompt to analyze
            working_directory: Current working directory
            
        Returns:
            PromptContext with analysis results
        """
        from .pattern_matcher import PromptPatternMatcher
        
        # Use pattern matcher to classify the prompt
        pattern_matcher = PromptPatternMatcher()
        prompt_type, confidence, intent = pattern_matcher.analyze_prompt_type(prompt)
        
        # Get required context types
        required_context = pattern_matcher.get_required_context(prompt_type, prompt)
        
        # Determine supplemental commands based on required context
        supplemental_commands = self._determine_commands(required_context)
        
        # Check repository context
        working_dir = working_directory or os.getcwd()
        has_repository = self._check_repository_context(working_dir)
        
        return PromptContext(
            original_prompt=prompt,
            prompt_type=prompt_type,
            confidence=confidence,
            detected_intent=intent,
            required_context=required_context,
            supplemental_commands=supplemental_commands,
            working_directory=working_dir,
            has_repository=has_repository
        )

    def _determine_commands(self, required_context: List[str]) -> List[str]:
        """Determine which commands to execute based on required context.
        
        Args:
            required_context: List of required context types
            
        Returns:
            List of commands to execute
        """
        commands = []
        
        for context_type in required_context:
            if context_type in self.context_providers:
                provider = self.context_providers[context_type]
                commands.extend(provider["commands"])
        
        # Remove duplicates while preserving order
        unique_commands = []
        seen = set()
        for cmd in commands:
            if cmd not in seen:
                unique_commands.append(cmd)
                seen.add(cmd)
        
        return unique_commands

    def _check_repository_context(self, working_directory: str) -> bool:
        """Check if the working directory is part of a repository.
        
        Args:
            working_directory: Directory to check
            
        Returns:
            True if directory is in a repository
        """
        # Check for common repository indicators
        repo_indicators = [
            '.git',           # Git repository
            '.svn',           # SVN repository
            '.hg',            # Mercurial repository
            'package.json',   # Node.js project
            'requirements.txt', # Python project
            'pom.xml',        # Maven project
            'build.gradle',   # Gradle project
            'Cargo.toml',     # Rust project
            'go.mod',         # Go module
            'composer.json',  # PHP project
            'Gemfile',        # Ruby project
            'pyproject.toml', # Modern Python project
        ]
        
        for indicator in repo_indicators:
            if os.path.exists(os.path.join(working_directory, indicator)):
                return True
        
        # Check parent directories for .git (common case)
        current_dir = working_directory
        for _ in range(5):  # Check up to 5 parent directories
            if os.path.exists(os.path.join(current_dir, '.git')):
                return True
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # Reached root
                break
            current_dir = parent_dir
        
        return False

    def prioritize_context(self, required_context: List[str], prompt_type: PromptType) -> List[str]:
        """Prioritize context types based on prompt type and importance.
        
        Args:
            required_context: List of required context types
            prompt_type: The type of prompt being processed
            
        Returns:
            Prioritized list of context types
        """
        # Define priority weights for different prompt types
        type_priorities = {
            PromptType.REPOSITORY_ANALYSIS: {
                "repository_structure": 1.0,
                "languages": 0.9,
                "technologies": 0.9,
                "file_structure": 0.8,
                "dependencies": 0.7
            },
            PromptType.CODE_ANALYSIS: {
                "code_structure": 1.0,
                "file_content": 0.9,
                "repository_structure": 0.7,
                "dependencies": 0.6
            },
            PromptType.FILE_OPERATIONS: {
                "file_structure": 1.0,
                "current_files": 0.9,
                "repository_structure": 0.6
            },
            PromptType.DEBUGGING: {
                "error_logs": 1.0,
                "recent_changes": 0.9,
                "code_structure": 0.8,
                "file_content": 0.7
            },
            PromptType.TECHNOLOGY_INQUIRY: {
                "technologies": 1.0,
                "dependencies": 0.9,
                "config_files": 0.8,
                "repository_structure": 0.6
            }
        }
        
        # Get priorities for this prompt type
        priorities = type_priorities.get(prompt_type, {})
        
        # Calculate scores for each context type
        scored_context = []
        for context_type in required_context:
            base_weight = self.context_providers.get(context_type, {}).get("weight", 0.5)
            type_priority = priorities.get(context_type, 0.5)
            total_score = base_weight * type_priority
            scored_context.append((context_type, total_score))
        
        # Sort by score (highest first)
        scored_context.sort(key=lambda x: x[1], reverse=True)
        
        return [context_type for context_type, _ in scored_context]

    def estimate_context_complexity(self, context: PromptContext) -> str:
        """Estimate the complexity of gathering the required context.
        
        Args:
            context: The prompt context to analyze
            
        Returns:
            Complexity level: 'low', 'medium', 'high'
        """
        num_commands = len(context.supplemental_commands)
        num_context_types = len(context.required_context)
        
        # High complexity indicators
        high_complexity_commands = ["analyze_code_structure", "analyze_dependencies", "get_recent_logs"]
        has_high_complexity = any(cmd in context.supplemental_commands for cmd in high_complexity_commands)
        
        if has_high_complexity or num_commands > 8 or num_context_types > 6:
            return "high"
        elif num_commands > 4 or num_context_types > 3:
            return "medium"
        else:
            return "low"
