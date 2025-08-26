"""Prompt enhancement module for augmenting prompts with context information."""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple

from .context_analyzer import ContextAnalyzer, PromptContext
from .command_generator import CommandGenerator


class InterceptionMode(Enum):
    """Mode for prompt interception and enhancement."""
    LIGHTWEIGHT = "lightweight"  # Quick context gathering, minimal commands
    SMART = "smart"              # Balanced approach, moderate context gathering  
    FULL = "full"                # Comprehensive context gathering, all relevant commands
    DISABLED = "disabled"        # No interception, pass through


@dataclass
class CommandExecution:
    """Result of executing a context-gathering command."""
    command: str
    success: bool
    output: str
    error: str
    execution_time: float
    context_type: str


@dataclass
class SupplementedPrompt:
    """A prompt that has been enhanced with context information."""
    original_prompt: str
    enhanced_prompt: str
    context_data: Dict[str, str]
    executed_commands: List[CommandExecution]
    processing_time: float
    mode_used: InterceptionMode
    confidence: float


class PromptEnhancer:
    """Main class for enhancing prompts with contextual information."""

    def __init__(self):
        """Initialize the prompt enhancer."""
        self.context_analyzer = ContextAnalyzer()
        self.command_generator = CommandGenerator()
        self.logger = logging.getLogger(__name__)

    def enhance_prompt(self, prompt: str, working_directory: Optional[str] = None, 
                      mode: InterceptionMode = InterceptionMode.SMART) -> SupplementedPrompt:
        """Enhance a prompt with contextual information.
        
        Args:
            prompt: The original user prompt
            working_directory: Current working directory
            mode: Enhancement mode to use
            
        Returns:
            SupplementedPrompt with enhanced content and metadata
        """
        import time
        
        start_time = time.time()
        
        try:
            # Analyze the prompt to determine context needs
            context = self.context_analyzer.analyze_prompt_context(prompt, working_directory)
            
            # Determine the actual execution mode
            execution_mode = self._determine_execution_mode(context, mode)
            
            if execution_mode == InterceptionMode.DISABLED:
                return SupplementedPrompt(
                    original_prompt=prompt,
                    enhanced_prompt=prompt,
                    context_data={},
                    executed_commands=[],
                    processing_time=time.time() - start_time,
                    mode_used=execution_mode,
                    confidence=0.0
                )
            
            # Apply the appropriate enhancement strategy
            if execution_mode == InterceptionMode.LIGHTWEIGHT:
                result = self._lightweight_enhancement(prompt, context)
            elif execution_mode == InterceptionMode.FULL:
                result = self._full_enhancement(prompt, context)
            else:  # SMART mode
                result = self._smart_enhancement(prompt, context)
            
            # Update processing time and mode
            result.processing_time = time.time() - start_time
            result.mode_used = execution_mode
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error enhancing prompt: {str(e)}")
            
            return SupplementedPrompt(
                original_prompt=prompt,
                enhanced_prompt=prompt,
                context_data={"error": str(e)},
                executed_commands=[],
                processing_time=time.time() - start_time,
                mode_used=mode,
                confidence=0.0
            )

    def _determine_execution_mode(self, context: PromptContext, requested_mode: InterceptionMode) -> InterceptionMode:
        """Determine the actual execution mode based on context and request.
        
        Args:
            context: Analyzed prompt context
            requested_mode: Mode requested by user
            
        Returns:
            Actual execution mode to use
        """
        # If user explicitly requested disabled, honor it
        if requested_mode == InterceptionMode.DISABLED:
            return InterceptionMode.DISABLED
        
        # If confidence is very low, disable enhancement
        if context.confidence < 0.3:
            return InterceptionMode.DISABLED
        
        # If no repository context detected and low confidence, use lightweight
        if not context.has_repository and context.confidence < 0.6:
            return InterceptionMode.LIGHTWEIGHT
        
        # Check complexity estimate
        complexity = self.context_analyzer.estimate_context_complexity(context)
        
        # Adjust mode based on complexity
        if requested_mode == InterceptionMode.FULL:
            return InterceptionMode.FULL
        elif requested_mode == InterceptionMode.LIGHTWEIGHT:
            return InterceptionMode.LIGHTWEIGHT
        else:  # SMART mode - adapt based on complexity
            if complexity == "high":
                return InterceptionMode.LIGHTWEIGHT
            elif complexity == "low":
                return InterceptionMode.FULL
            else:
                return InterceptionMode.SMART

    def _lightweight_enhancement(self, prompt: str, context: PromptContext) -> SupplementedPrompt:
        """Apply lightweight enhancement with minimal context gathering.
        
        Args:
            prompt: Original prompt
            context: Prompt context analysis
            
        Returns:
            Enhanced prompt with minimal context
        """
        # Generate minimal set of commands
        commands = self.command_generator.generate_context_commands(context)
        
        # Limit to maximum 3 fastest commands
        limited_commands = []
        for cmd in commands:
            if cmd.get("timeout", 15) <= 10 and len(limited_commands) < 3:
                limited_commands.append(cmd)
        
        # Execute commands and gather context
        context_data, executions = self._execute_commands(limited_commands, context.working_directory)
        
        # Create enhanced prompt
        enhanced_prompt = self._build_enhanced_prompt(prompt, context_data, "lightweight")
        
        return SupplementedPrompt(
            original_prompt=prompt,
            enhanced_prompt=enhanced_prompt,
            context_data=context_data,
            executed_commands=executions,
            processing_time=0.0,  # Will be set by caller
            mode_used=InterceptionMode.LIGHTWEIGHT,
            confidence=context.confidence
        )

    def _smart_enhancement(self, prompt: str, context: PromptContext) -> SupplementedPrompt:
        """Apply smart enhancement with balanced context gathering.
        
        Args:
            prompt: Original prompt
            context: Prompt context analysis
            
        Returns:
            Enhanced prompt with balanced context
        """
        # Generate optimized commands
        commands = self.command_generator.generate_context_commands(context)
        optimized_commands = self.command_generator.optimize_command_sequence(commands, context.prompt_type)
        
        # Limit execution time to reasonable amount
        time_budget = 30  # seconds
        selected_commands = []
        estimated_time = 0
        
        for cmd in optimized_commands:
            cmd_time = cmd.get("timeout", 15)
            if estimated_time + cmd_time <= time_budget:
                selected_commands.append(cmd)
                estimated_time += cmd_time
            else:
                break
        
        # Execute commands and gather context
        context_data, executions = self._execute_commands(selected_commands, context.working_directory)
        
        # Create enhanced prompt
        enhanced_prompt = self._build_enhanced_prompt(prompt, context_data, "smart")
        
        return SupplementedPrompt(
            original_prompt=prompt,
            enhanced_prompt=enhanced_prompt,
            context_data=context_data,
            executed_commands=executions,
            processing_time=0.0,  # Will be set by caller
            mode_used=InterceptionMode.SMART,
            confidence=context.confidence
        )

    def _full_enhancement(self, prompt: str, context: PromptContext) -> SupplementedPrompt:
        """Apply full enhancement with comprehensive context gathering.
        
        Args:
            prompt: Original prompt
            context: Prompt context analysis
            
        Returns:
            Enhanced prompt with comprehensive context
        """
        # Generate all relevant commands
        commands = self.command_generator.generate_context_commands(context)
        optimized_commands = self.command_generator.optimize_command_sequence(commands, context.prompt_type)
        
        # Execute all commands (with reasonable limits)
        max_commands = 15
        selected_commands = optimized_commands[:max_commands]
        
        # Execute commands and gather context
        context_data, executions = self._execute_commands(selected_commands, context.working_directory)
        
        # Create enhanced prompt
        enhanced_prompt = self._build_enhanced_prompt(prompt, context_data, "full")
        
        return SupplementedPrompt(
            original_prompt=prompt,
            enhanced_prompt=enhanced_prompt,
            context_data=context_data,
            executed_commands=executions,
            processing_time=0.0,  # Will be set by caller
            mode_used=InterceptionMode.FULL,
            confidence=context.confidence
        )

    def _execute_commands(self, commands: List[Dict[str, str]], working_directory: str) -> Tuple[Dict[str, str], List[CommandExecution]]:
        """Execute context-gathering commands.
        
        Args:
            commands: List of commands to execute
            working_directory: Directory to execute commands in
            
        Returns:
            Tuple of (context_data, command_executions)
        """
        import subprocess
        import time
        
        context_data = {}
        executions = []
        
        for cmd_info in commands:
            start_time = time.time()
            
            try:
                # Execute the command
                result = subprocess.run(
                    cmd_info["command"],
                    shell=True,
                    cwd=working_directory,
                    capture_output=True,
                    text=True,
                    timeout=cmd_info.get("timeout", 15)
                )
                
                execution_time = time.time() - start_time
                
                if result.returncode == 0:
                    output = result.stdout.strip()
                    context_data[cmd_info["name"]] = output
                    
                    executions.append(CommandExecution(
                        command=cmd_info["command"],
                        success=True,
                        output=output,
                        error="",
                        execution_time=execution_time,
                        context_type=cmd_info["name"]
                    ))
                else:
                    error = result.stderr.strip()
                    executions.append(CommandExecution(
                        command=cmd_info["command"],
                        success=False,
                        output="",
                        error=error,
                        execution_time=execution_time,
                        context_type=cmd_info["name"]
                    ))
                    
            except subprocess.TimeoutExpired:
                execution_time = time.time() - start_time
                executions.append(CommandExecution(
                    command=cmd_info["command"],
                    success=False,
                    output="",
                    error="Command timed out",
                    execution_time=execution_time,
                    context_type=cmd_info["name"]
                ))
                
            except Exception as e:
                execution_time = time.time() - start_time
                executions.append(CommandExecution(
                    command=cmd_info["command"],
                    success=False,
                    output="",
                    error=str(e),
                    execution_time=execution_time,
                    context_type=cmd_info["name"]
                ))
        
        return context_data, executions

    def _build_enhanced_prompt(self, original_prompt: str, context_data: Dict[str, str], mode: str) -> str:
        """Build an enhanced prompt with context information.
        
        Args:
            original_prompt: The original user prompt
            context_data: Context information gathered
            mode: Enhancement mode used
            
        Returns:
            Enhanced prompt string
        """
        if not context_data:
            return original_prompt
        
        # Build context section
        context_sections = []
        
        # Add repository state if available
        if "get_repository_state" in context_data:
            context_sections.append(f"## Repository Context\n{context_data['get_repository_state']}")
        
        # Add file listings if available
        if "list_files" in context_data:
            context_sections.append(f"## Project Files\n{context_data['list_files']}")
        
        # Add code structure if available
        if "analyze_code_structure" in context_data:
            context_sections.append(f"## Code Structure\n{context_data['analyze_code_structure']}")
        
        # Add technologies if available
        if "analyze_technologies" in context_data:
            context_sections.append(f"## Technologies & Dependencies\n{context_data['analyze_technologies']}")
        
        # Add any other context data
        for key, value in context_data.items():
            if key not in ["get_repository_state", "list_files", "analyze_code_structure", "analyze_technologies"] and value:
                # Format key as readable title
                title = key.replace("_", " ").replace("-", " ").title()
                context_sections.append(f"## {title}\n{value}")
        
        # Combine context sections
        if context_sections:
            context_info = "\n\n".join(context_sections)
            enhanced_prompt = f"""Based on the following project context:

{context_info}

---

{original_prompt}"""
        else:
            enhanced_prompt = original_prompt
        
        return enhanced_prompt

    def get_enhancement_statistics(self) -> Dict[str, Any]:
        """Get statistics about prompt enhancement usage.
        
        Returns:
            Dictionary with enhancement statistics
        """
        # This would typically track statistics across multiple enhancement calls
        # For now, return basic information
        return {
            "available_modes": [mode.value for mode in InterceptionMode],
            "context_providers": list(self.context_analyzer.context_providers.keys()),
            "command_templates": list(self.command_generator.command_templates.keys())
        }
