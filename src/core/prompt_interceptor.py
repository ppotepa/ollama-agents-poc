"""Prompt Interceptor - Intelligent prompt interceptor and context decorator.

This system intercepts every user prompt, analyzes the intent and context,
and automatically supplements prompts with relevant repository data and context
before sending them to the agent.
"""

import re
import os
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import time


class InterceptionMode(Enum):
    """Modes for prompt interception processing."""
    FULL = "full"          # Complete context analysis with all commands
    LIGHTWEIGHT = "lightweight"  # Fast analysis with minimal context
    SMART = "smart"        # Adaptive mode based on prompt complexity
    

class PromptType(Enum):
    """Types of prompts that can be intercepted and processed."""
    REPOSITORY_ANALYSIS = "repository_analysis"
    CODE_ANALYSIS = "code_analysis"
    FILE_OPERATIONS = "file_operations"
    GENERAL_CODING = "general_coding"
    QUESTION_ANSWERING = "question_answering"
    PROJECT_NAVIGATION = "project_navigation"
    TECHNOLOGY_INQUIRY = "technology_inquiry"
    ARCHITECTURE_REVIEW = "architecture_review"
    DEBUGGING = "debugging"
    GENERAL = "general"


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


@dataclass
class CommandExecution:
    """Information about an executed command."""
    command_name: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    result_length: int
    error_message: Optional[str] = None


@dataclass
class SupplementedPrompt:
    """A prompt that has been enhanced with contextual information."""
    original_prompt: str
    supplemented_prompt: str
    context_used: List[str]
    commands_executed: List[CommandExecution]
    metadata: Dict[str, Any]


class PromptInterceptor:
    """Intelligent prompt interceptor that enhances every prompt with contextual information."""
    
    def __init__(self):
        """Initialize the prompt interceptor."""
        self.prompt_patterns = self._build_prompt_patterns()
        self.context_providers = self._initialize_context_providers()
        self.command_registry = self._initialize_command_registry()
        
    def _build_prompt_patterns(self) -> Dict[PromptType, Dict[str, Any]]:
        """Build patterns for recognizing different types of prompts."""
        return {
            PromptType.REPOSITORY_ANALYSIS: {
                "patterns": [
                    r"analyz[e|ing]*\s+repositor[y|ies]*",
                    r"repositor[y|ies]*\s+analys[is|e]*",
                    r"analyz[e|ing]*\s+project\s+structure",
                    r"project\s+structure",
                    r"code\s+structure",
                    r"codebase\s+analys[is|e]*",
                    r"what.*language.*project",
                    r"what.*technologies.*used",
                    r"project\s+overview",
                    r"repository\s+structure",
                    r"file\s+structure",
                    r"directory\s+structure",
                    r"how\s+is.*organized",
                    r"show.*project.*layout",
                    r"project.*composition"
                ],
                "context_commands": [
                    "analyze_virtual_repository",
                    "analyze_repository_context",
                    "analyze_repo_languages", 
                    "analyze_repo_directories",
                    "get_repository_state"
                ],
                "description": "Repository and project structure analysis"
            },
            
            PromptType.CODE_ANALYSIS: {
                "patterns": [
                    r"analyz[e|ing]*.*code",
                    r"code\s+review",
                    r"code\s+quality",
                    r"find.*issues",
                    r"check.*code",
                    r"lint.*code",
                    r"code.*problems",
                    r"improve.*code",
                    r"optimize.*code",
                    r"refactor.*code"
                ],
                "context_commands": [
                    "analyze_repository_context",
                    "search_files",
                    "get_repository_state"
                ],
                "description": "Code analysis and review"
            },
            
            PromptType.FILE_OPERATIONS: {
                "patterns": [
                    r"create.*file",
                    r"write.*file",
                    r"read.*file",
                    r"find.*file",
                    r"search.*file",
                    r"list.*files",
                    r"show.*files",
                    r"file.*content",
                    r"open.*file",
                    r"edit.*file"
                ],
                "context_commands": [
                    "search_files",
                    "get_repository_state"
                ],
                "description": "File operations and content access"
            },
            
            PromptType.GENERAL_CODING: {
                "patterns": [
                    r"how.*implement",
                    r"how.*create",
                    r"how.*build",
                    r"write.*function",
                    r"create.*class",
                    r"implement.*feature",
                    r"add.*functionality",
                    r"help.*with.*coding",
                    r"programming.*question"
                ],
                "context_commands": [
                    "analyze_repo_languages",
                    "get_repository_state"
                ],
                "description": "General coding assistance"
            },
            
            PromptType.PROJECT_NAVIGATION: {
                "patterns": [
                    r"where.*is",
                    r"find.*in.*project",
                    r"locate.*file",
                    r"which.*file.*contains",
                    r"show.*me.*where",
                    r"navigate.*to",
                    r"explore.*project"
                ],
                "context_commands": [
                    "search_files",
                    "analyze_repo_directories",
                    "get_repository_state"
                ],
                "description": "Project navigation and file location"
            },
            
            PromptType.TECHNOLOGY_INQUIRY: {
                "patterns": [
                    r"what.*framework",
                    r"what.*library",
                    r"what.*technology",
                    r"what.*stack",
                    r"dependencies",
                    r"what.*using",
                    r"built.*with",
                    r"tech.*stack",
                    r"technologies.*used"
                ],
                "context_commands": [
                    "analyze_repository_context",
                    "analyze_repo_languages"
                ],
                "description": "Technology and framework identification"
            },
            
            PromptType.ARCHITECTURE_REVIEW: {
                "patterns": [
                    r"architecture",
                    r"design.*pattern",
                    r"system.*design",
                    r"project.*architecture",
                    r"how.*structured",
                    r"overall.*design",
                    r"architectural.*overview"
                ],
                "context_commands": [
                    "analyze_repository_context",
                    "analyze_repo_directories",
                    "get_repository_state"
                ],
                "description": "System architecture and design analysis"
            },
            
            PromptType.DEBUGGING: {
                "patterns": [
                    r"debug",
                    r"error",
                    r"bug",
                    r"issue",
                    r"problem",
                    r"not.*working",
                    r"fix.*this",
                    r"troubleshoot",
                    r"help.*with.*error"
                ],
                "context_commands": [
                    "get_repository_state",
                    "search_files"
                ],
                "description": "Debugging and error resolution"
            }
        }
    
    def _initialize_context_providers(self) -> Dict[str, Any]:
        """Initialize context providers for different types of information."""
        return {
            "repository_context": self._get_repository_context,
            "file_context": self._get_file_context,
            "language_context": self._get_language_context,
            "directory_context": self._get_directory_context,
            "technology_context": self._get_technology_context
        }
    
    def _initialize_command_registry(self) -> Dict[str, Any]:
        """Initialize the command registry with available commands."""
        return {
            "analyze_repository_context": {
                "function": self._execute_analyze_repository_context,
                "description": "Analyze overall repository structure and composition",
                "category": "repository"
            },
            "analyze_virtual_repository": {
                "function": self._execute_analyze_virtual_repository,
                "description": "Analyze virtual repository loaded from ZIP in memory",
                "category": "virtual"
            },
            "analyze_repo_languages": {
                "function": self._execute_analyze_repo_languages,
                "description": "Analyze programming languages used in the repository",
                "category": "language"
            },
            "analyze_repo_directories": {
                "function": self._execute_analyze_repo_directories,
                "description": "Analyze directory structure and organization",
                "category": "structure"
            },
            "get_repository_state": {
                "function": self._execute_get_repository_state,
                "description": "Get current repository state and statistics",
                "category": "state"
            },
            "search_files": {
                "function": self._execute_search_files,
                "description": "Search for files and content in the repository",
                "category": "search"
            }
        }
    
    def intercept_and_enhance(self, prompt: str, working_directory: str = None, mode: InterceptionMode = InterceptionMode.SMART) -> SupplementedPrompt:
        """Main entry point: intercept a prompt and enhance it with contextual information.
        
        Args:
            prompt: The original user prompt
            working_directory: Current working directory (optional)
            mode: Interception mode (FULL, LIGHTWEIGHT, or SMART)
            
        Returns:
            SupplementedPrompt with enhanced context
        """
        print(f"🔍 Prompt Interceptor: Analyzing prompt (mode: {mode.value})...")
        
        # Analyze the prompt context
        context = self._analyze_prompt_context(prompt, working_directory)
        
        print(f"🎯 Detected Intent: {context.detected_intent}")
        print(f"📊 Confidence: {context.confidence:.2f}")
        
        # Determine execution mode
        execution_mode = self._determine_execution_mode(context, mode)
        print(f"⚡ Execution Mode: {execution_mode.value}")
        
        if execution_mode == InterceptionMode.LIGHTWEIGHT:
            enhanced_result = self._lightweight_enhancement(prompt, context)
        else:
            print(f"🔧 Planned Commands: {', '.join(context.supplemental_commands)}")
            enhanced_result = self._full_enhancement(prompt, context)
        
        # **FINAL ENHANCED PROMPT LOGGING**
        print(f"\n" + "="*100)
        print(f"🚀 FINAL ENHANCED PROMPT ({execution_mode.value.upper()} MODE)")
        print(f"="*100)
        print(enhanced_result.supplemented_prompt)
        print(f"="*100)
        print(f"📈 Context: {len(enhanced_result.context_used)} types | Characters: {len(enhanced_result.supplemented_prompt)}")
        print(f"="*100 + "\n")
        
        return enhanced_result
    
    def _get_command_recommendations(self, prompt: str) -> List[Dict[str, Any]]:
        """Get command recommendations using the interceptor agent."""
        try:
            from src.agents.interceptor.agent import InterceptorAgent
            
            # Create minimal config for the interceptor agent
            config = {
                "name": "Prompt Interceptor Agent",
                "backend_image": "phi3:mini",
                "parameters": {"temperature": 0.1, "num_ctx": 2048}
            }
            
            interceptor_agent = InterceptorAgent("interceptor", config)
            print(f"     🤖 Analyzing prompt with phi3:mini interceptor agent...")
            print(f"     📝 Original query: '{prompt[:100]}{'...' if len(prompt) > 100 else ''}'")
            recommendations = interceptor_agent.analyze_prompt(prompt, InterceptionMode.LIGHTWEIGHT)
            print(f"     ✅ Interceptor agent returned {len(recommendations)} recommendations")
            
            # Log what was detected/resolved from the query
            if recommendations:
                print(f"     🔍 Resolved from query:")
                for rec in recommendations:
                    print(f"         • {rec.command} → {rec.description} (confidence: {rec.confidence:.1%})")
            
            # Convert to serializable format
            serialized_recs = [
                {
                    "command": rec.command,
                    "confidence": rec.confidence,
                    "description": rec.description,
                    "category": rec.category,
                    "required_context": rec.required_context
                }
                for rec in recommendations
            ]
            
            return serialized_recs
            
        except Exception as e:
            print(f"     ⚠️  Could not get command recommendations: {e}")
            return []
    
    def _determine_execution_mode(self, context: PromptContext, requested_mode: InterceptionMode) -> InterceptionMode:
        """Determine the best execution mode based on context and request."""
        if requested_mode == InterceptionMode.LIGHTWEIGHT:
            return InterceptionMode.LIGHTWEIGHT
        elif requested_mode == InterceptionMode.FULL:
            return InterceptionMode.FULL
        else:  # SMART mode
            # Use lightweight for simple questions and high-confidence general queries
            if (context.prompt_type == PromptType.GENERAL and context.confidence < 0.3) or \
               (context.prompt_type == PromptType.QUESTION_ANSWERING and context.confidence > 0.7):
                return InterceptionMode.LIGHTWEIGHT
            # Use full mode for repository/code analysis
            elif context.prompt_type in [PromptType.REPOSITORY_ANALYSIS, PromptType.CODE_ANALYSIS, 
                                       PromptType.ARCHITECTURE_REVIEW]:
                return InterceptionMode.FULL
            # Default to lightweight for faster responses
            else:
                return InterceptionMode.LIGHTWEIGHT
    
    def _lightweight_enhancement(self, prompt: str, context: PromptContext) -> SupplementedPrompt:
        """Provide lightweight enhancement with minimal context using the interceptor agent."""
        print(f"⚡ Using lightweight mode for fast response...")
        
        # Use the interceptor agent for command analysis
        command_recommendations = self._get_command_recommendations(prompt)
        
        # **LOG COMMAND RESOLUTION**
        print(f"🔍 Command Resolution Results:")
        if command_recommendations:
            print(f"     📊 Total commands recommended: {len(command_recommendations)}")
            for i, cmd_rec in enumerate(command_recommendations, 1):
                print(f"     [{i}] {cmd_rec['command']} - {cmd_rec['confidence']:.1%} confidence")
                print(f"         📝 {cmd_rec.get('description', 'No description')}")
                print(f"         🏷️  Category: {cmd_rec.get('category', 'unknown')}")
        else:
            print(f"     ❌ No commands recommended by interceptor agent")
        print()
        
        # Execute recommended commands and gather results
        context_data = {}
        command_executions = []
        
        if command_recommendations:
            print(f"     ⚡ Executing {len(command_recommendations)} recommended commands...")
            
            # Execute top 3 commands for lightweight mode
            top_commands = command_recommendations[:3]
            for i, cmd_rec in enumerate(top_commands, 1):
                command_name = cmd_rec['command']
                confidence = cmd_rec['confidence']
                
                print(f"     🔧 [{i}/3] Executing: {command_name} (confidence: {confidence:.1%})")
                
                try:
                    start_time = time.time()
                    result = self._execute_command_safely(command_name)
                    end_time = time.time()
                    
                    if result:
                        context_data[f"command_{command_name}"] = result
                        print(f"         ✅ Result: {len(result)} chars")
                        
                        command_executions.append(CommandExecution(
                            command_name=command_name,
                            start_time=start_time,
                            end_time=end_time,
                            duration=end_time - start_time,
                            success=True,
                            result_length=len(result)
                        ))
                    else:
                        print(f"         ⚠️  No result returned")
                        
                except Exception as e:
                    print(f"         ❌ Failed: {e}")
                    command_executions.append(CommandExecution(
                        command_name=command_name,
                        start_time=time.time(),
                        end_time=time.time(),
                        duration=0,
                        success=False,
                        result_length=0,
                        error_message=str(e)
                    ))
        
        # Add any essential context based on prompt type
        if context.prompt_type == PromptType.REPOSITORY_ANALYSIS and not context_data:
            # Fallback: get basic repository info if no commands executed
            try:
                from src.tools.context import get_repository_state
                result = get_repository_state()
                context_data["basic_repository_info"] = result
                print(f"     ⚡ Basic repository info retrieved as fallback")
            except Exception as e:
                print(f"     ⚠️  Could not get basic repository info: {e}")
        
        # Create enhanced prompt with command results
        enhanced_prompt = self._create_lightweight_prompt_with_results(prompt, context, context_data, command_executions)
        
        # Log the enhanced prompt for debugging
        print(f"\n📝 Enhanced Prompt Generated:")
        print("=" * 80)
        print(enhanced_prompt)
        print("=" * 80)
        print(f"📊 Enhancement Summary: {len(command_executions)} commands executed, {len(enhanced_prompt)} characters total\n")
        
        return SupplementedPrompt(
            original_prompt=prompt,
            supplemented_prompt=enhanced_prompt,
            context_used=list(context_data.keys()) if context_data else [],
            commands_executed=[],
            metadata={
                "mode": "lightweight",
                "detected_intent": context.detected_intent,
                "confidence": context.confidence,
                "context_types": list(context_data.keys()),
                "execution_time": 0.001  # Very fast
            }
        )
    
    def _full_enhancement(self, prompt: str, context: PromptContext) -> SupplementedPrompt:
        """Provide full enhancement with complete context analysis."""
        print(f"🔧 Planned Commands: {', '.join(context.supplemental_commands)}")
        
        # Gather contextual information by executing commands
        context_data, command_executions = self._gather_context_data_with_logging(context)
        
        # Log command execution summary
        self._log_command_execution_summary(command_executions)
        
        # Supplement the prompt
        supplemented = self._supplement_prompt(prompt, context, context_data, command_executions)
        
        # Log the enhanced prompt for debugging
        print(f"\n📝 Enhanced Prompt Generated (Full Mode):")
        print("=" * 80)
        print(supplemented.supplemented_prompt)
        print("=" * 80)
        print(f"📊 Enhancement Summary: {len(command_executions)} commands executed, {len(supplemented.supplemented_prompt)} characters total\n")
        
        return supplemented
    
    def _analyze_prompt_context(self, prompt: str, working_directory: str = None) -> PromptContext:
        """Analyze the prompt to understand intent and required context."""
        prompt_lower = prompt.lower().strip()
        
        best_match_type = PromptType.GENERAL
        highest_confidence = 0.0
        detected_intent = "General query"
        required_context = []
        supplemental_commands = []
        
        # Check each prompt pattern
        for prompt_type, config in self.prompt_patterns.items():
            confidence = self._calculate_pattern_confidence(prompt_lower, config["patterns"])
            
            if confidence > highest_confidence:
                highest_confidence = confidence
                best_match_type = prompt_type
                detected_intent = config["description"]
                supplemental_commands = config["context_commands"].copy()
        
        # Determine required context based on prompt type
        required_context = self._determine_required_context(best_match_type, prompt_lower)
        
        # Check if we're in a repository context
        has_repository = self._check_repository_context(working_directory)
        
        return PromptContext(
            original_prompt=prompt,
            prompt_type=best_match_type,
            confidence=highest_confidence,
            detected_intent=detected_intent,
            required_context=required_context,
            supplemental_commands=supplemental_commands,
            working_directory=working_directory or os.getcwd(),
            has_repository=has_repository
        )
    
    def _calculate_pattern_confidence(self, prompt: str, patterns: List[str]) -> float:
        """Calculate confidence score for pattern matching."""
        max_score = 0.0
        
        for pattern in patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                # Base score for pattern match
                score = 0.7
                
                # Boost score for longer, more specific patterns
                if len(pattern) > 20:
                    score += 0.2
                
                # Boost score based on pattern coverage
                pattern_words = len(re.findall(r'\w+', pattern))
                prompt_words = len(prompt.split())
                if prompt_words > 0:
                    coverage = min(pattern_words / prompt_words, 1.0)
                    score += coverage * 0.1
                
                max_score = max(max_score, score)
        
        return min(max_score, 1.0)
    
    def _determine_required_context(self, prompt_type: PromptType, prompt: str) -> List[str]:
        """Determine what contextual information is needed for this prompt."""
        context_needs = []
        
        if prompt_type in [PromptType.REPOSITORY_ANALYSIS, PromptType.ARCHITECTURE_REVIEW]:
            context_needs.extend(["repository_structure", "languages", "directories", "technologies"])
        elif prompt_type == PromptType.CODE_ANALYSIS:
            context_needs.extend(["file_contents", "languages", "repository_structure"])
        elif prompt_type == PromptType.FILE_OPERATIONS:
            context_needs.extend(["file_listing", "directory_structure"])
        elif prompt_type == PromptType.PROJECT_NAVIGATION:
            context_needs.extend(["file_listing", "directory_structure", "file_search"])
        elif prompt_type == PromptType.TECHNOLOGY_INQUIRY:
            context_needs.extend(["languages", "technologies", "dependencies"])
        elif prompt_type == PromptType.GENERAL_CODING:
            context_needs.extend(["languages", "repository_structure"])
        
        return context_needs
    
    def _check_repository_context(self, working_directory: str = None) -> bool:
        """Check if we're currently in a repository context."""
        try:
            # Check if repository context is available
            from src.tools.context import get_repository_state
            state = get_repository_state()
            return "Total Files: 0" not in state
        except Exception:
            return False
    
    def _gather_context_data_with_logging(self, context: PromptContext) -> Tuple[Dict[str, str], List[CommandExecution]]:
        """Gather contextual information with detailed command execution logging."""
        context_data = {}
        command_executions = []
        
        if not context.has_repository:
            print(f"⚠️  No repository context available - skipping command execution")
            return context_data, command_executions
        
        print(f"📋 Executing {len(context.supplemental_commands)} context commands...")
        
        try:
            # Import repository context tools
            from src.tools.context import (
                analyze_repository_context,
                analyze_repo_languages,
                analyze_repo_directories,
                get_repository_state,
                search_files
            )
            
            # Execute each supplemental command with logging
            for i, command in enumerate(context.supplemental_commands, 1):
                print(f"  🔧 [{i}/{len(context.supplemental_commands)}] Executing: {command}")
                
                start_time = time.time()
                success = False
                result_length = 0
                error_message = None
                
                try:
                    if command == "analyze_virtual_repository":
                        result = self._execute_analyze_virtual_repository()
                        context_data["virtual_repository"] = result
                        success = True
                        result_length = len(result)
                        print(f"     ✅ Virtual repository analyzed ({result_length} chars)")
                        
                    elif command == "analyze_repository_context":
                        result = analyze_repository_context()
                        context_data["repository_context"] = result
                        success = True
                        result_length = len(result)
                        print(f"     ✅ Repository context analyzed ({result_length} chars)")
                        
                    elif command == "analyze_repo_languages":
                        result = analyze_repo_languages()
                        context_data["language_analysis"] = result
                        success = True
                        result_length = len(result)
                        print(f"     ✅ Language analysis completed ({result_length} chars)")
                        
                    elif command == "analyze_repo_directories":
                        result = analyze_repo_directories()
                        context_data["directory_analysis"] = result
                        success = True
                        result_length = len(result)
                        print(f"     ✅ Directory analysis completed ({result_length} chars)")
                        
                    elif command == "get_repository_state":
                        result = get_repository_state()
                        context_data["repository_state"] = result
                        success = True
                        result_length = len(result)
                        print(f"     ✅ Repository state retrieved ({result_length} chars)")
                        
                    elif command == "get_virtual_repository_summary":
                        from src.tools.context import get_virtual_repository_summary
                        result = get_virtual_repository_summary()
                        context_data["virtual_repository"] = result
                        success = True
                        result_length = len(result)
                        print(f"     ✅ Virtual repository summary ({result_length} chars)")
                        
                    elif command == "search_files":
                        # For search_files, we use basic file listing for now
                        result = get_repository_state()
                        context_data["file_overview"] = result
                        success = True
                        result_length = len(result)
                        print(f"     ✅ File overview retrieved ({result_length} chars)")
                        
                    else:
                        error_message = f"Unknown command: {command}"
                        print(f"     ❌ Unknown command: {command}")
                        
                except Exception as e:
                    error_message = str(e)
                    print(f"     ❌ Command failed: {e}")
                
                end_time = time.time()
                duration = end_time - start_time
                
                # Record command execution
                execution = CommandExecution(
                    command_name=command,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    success=success,
                    result_length=result_length,
                    error_message=error_message
                )
                command_executions.append(execution)
                    
        except ImportError as e:
            print(f"⚠️  Repository context tools not available: {e}")
        
        return context_data, command_executions
    
    def _log_command_execution_summary(self, executions: List[CommandExecution]):
        """Log a summary of command executions."""
        if not executions:
            return
        
        successful_commands = [e for e in executions if e.success]
        failed_commands = [e for e in executions if not e.success]
        total_duration = sum(e.duration for e in executions)
        total_data_size = sum(e.result_length for e in successful_commands)
        
        print(f"📊 Command Execution Summary:")
        print(f"   ✅ Successful: {len(successful_commands)}/{len(executions)}")
        print(f"   ⏱️  Total Time: {total_duration:.3f}s")
        print(f"   📦 Total Data: {total_data_size:,} characters")
        
        if failed_commands:
            print(f"   ❌ Failed Commands:")
            for failed in failed_commands:
                print(f"      - {failed.command_name}: {failed.error_message}")
    
    def _supplement_prompt(self, original_prompt: str, context: PromptContext, 
                         context_data: Dict[str, str], command_executions: List[CommandExecution]) -> SupplementedPrompt:
        """Create a supplemented prompt with contextual information."""
        
        if not context_data:
            # No contextual information available, return original prompt
            return SupplementedPrompt(
                original_prompt=original_prompt,
                supplemented_prompt=original_prompt,
                context_used=[],
                commands_executed=command_executions,
                metadata={"prompt_type": context.prompt_type.value, "confidence": context.confidence}
            )
        
        # Build the supplemented prompt
        prompt_parts = [
            f"**User Query:** {original_prompt}",
            "",
            "**Context Information:**"
        ]
        
        context_used = []
        
        # Add relevant context data
        if "repository_context" in context_data:
            prompt_parts.extend([
                "**Repository Analysis:**",
                context_data["repository_context"],
                ""
            ])
            context_used.append("repository_analysis")
        
        if "language_analysis" in context_data:
            prompt_parts.extend([
                "**Language Analysis:**", 
                context_data["language_analysis"],
                ""
            ])
            context_used.append("language_analysis")
        
        if "directory_analysis" in context_data:
            prompt_parts.extend([
                "**Directory Structure:**",
                context_data["directory_analysis"], 
                ""
            ])
            context_used.append("directory_analysis")
        
        if "repository_state" in context_data:
            prompt_parts.extend([
                "**Repository State:**",
                context_data["repository_state"],
                ""
            ])
            context_used.append("repository_state")
        
        # Add command execution summary to the prompt
        successful_commands = [e for e in command_executions if e.success]
        if successful_commands:
            prompt_parts.extend([
                "**Context Generation Details:**",
                f"Generated using {len(successful_commands)} repository analysis commands:",
                ", ".join([f"{e.command_name} ({e.duration:.2f}s)" for e in successful_commands]),
                ""
            ])
        
        # Add instructions for the agent
        prompt_parts.extend([
            "**Instructions:**",
            f"Based on the above context information and the user's query '{original_prompt}', provide a comprehensive response.",
            f"The user is asking about: {context.detected_intent}",
            "Use the provided context information to give accurate, specific answers about this repository/project.",
            ""
        ])
        
        supplemented_prompt = "\n".join(prompt_parts)
        
        print(f"✨ Prompt supplemented with {len(context_used)} context types")
        
        return SupplementedPrompt(
            original_prompt=original_prompt,
            supplemented_prompt=supplemented_prompt,
            context_used=context_used,
            commands_executed=command_executions,
            metadata={
                "prompt_type": context.prompt_type.value,
                "confidence": context.confidence,
                "detected_intent": context.detected_intent,
                "has_repository": context.has_repository,
                "execution_time": sum(e.duration for e in command_executions),
                "data_size": sum(e.result_length for e in command_executions if e.success)
            }
        )
    
    def _create_lightweight_prompt(self, original_prompt: str, context: PromptContext, context_data: Dict[str, str]) -> str:
        """Create a lightweight enhanced prompt with minimal context."""
        if not context_data:
            # No context available, return original prompt
            return original_prompt
        
        # Create a minimal context summary
        context_summary = []
        if "basic_repository_info" in context_data:
            context_summary.append("🔍 Repository Context Available")
        
        # Add command recommendations if available
        prompt_parts = []
        if "command_recommendations" in context_data:
            recommendations = context_data["command_recommendations"]
            if recommendations:
                prompt_parts.append("**Recommended Commands Available:**")
                for rec in recommendations[:3]:  # Top 3 recommendations
                    prompt_parts.append(f"- {rec['command']} (confidence: {rec['confidence']:.1%}) - {rec['description']}")
                prompt_parts.append("")
        
        if context_summary:
            prompt_parts.append(f"**Context**: {' | '.join(context_summary)}")
        
        prompt_parts.extend([
            f"**Query**: {original_prompt}",
            "",
            "Provide a focused response using available context and command recommendations."
        ])
        
        return "\n".join(prompt_parts) if prompt_parts else original_prompt
    
    def _execute_command_safely(self, command_name: str) -> Optional[str]:
        """Safely execute a command and return its result."""
        try:
            # Map command names to their execution functions
            command_map = {
                "analyze_repo_structure": self._execute_analyze_repository_context,
                "analyze_repo_languages": self._execute_analyze_repo_languages,
                "analyze_repo_directories": self._execute_analyze_repo_directories,
                "get_repository_state": self._execute_get_repository_state,
                "read_file": lambda: "File reading requires specific file path",
                "write_file": lambda: "File writing requires content and path",
                "list_files": self._execute_list_files,
                "duck_search": lambda: "Web search requires specific query",
                "fetch_url": lambda: "URL fetch requires specific URL",
                "run_python": lambda: "Python execution requires specific code"
            }
            
            if command_name in command_map:
                return command_map[command_name]()
            else:
                return f"Unknown command: {command_name}"
                
        except Exception as e:
            print(f"     ⚠️  Command {command_name} failed: {e}")
            return None
    
    def _execute_list_files(self) -> str:
        """Execute file listing in current directory."""
        try:
            import os
            files = os.listdir(".")
            dirs = [f for f in files if os.path.isdir(f)]
            files = [f for f in files if os.path.isfile(f)]
            
            result = []
            if dirs:
                result.append(f"Directories ({len(dirs)}): {', '.join(dirs)}")
            
            if files:
                result.append(f"Files ({len(files)}): {', '.join(files)}")
            
            return "\n".join(result) if result else "No files or directories found"
            
        except Exception as e:
            return f"Error listing files: {e}"
    
    def _create_lightweight_prompt_with_results(self, original_prompt: str, context: PromptContext, 
                                               context_data: Dict[str, str], command_executions: List[CommandExecution]) -> str:
        """Create a lightweight enhanced prompt with actual command execution results."""
        if not context_data:
            return original_prompt
        
        prompt_parts = [
            f"**User Query:** {original_prompt}",
            "",
            "**Context Information:**"
        ]
        
        # Add command execution results
        executed_commands = [cmd for cmd in command_executions if cmd.success]
        if executed_commands:
            prompt_parts.append("**Command Execution Results:**")
            
            for cmd in executed_commands:
                command_key = f"command_{cmd.command_name}"
                if command_key in context_data:
                    result = context_data[command_key]
                    prompt_parts.extend([
                        f"",
                        f"**{cmd.command_name}** (executed in {cmd.duration:.2f}s):",
                        result[:1000] + ("..." if len(result) > 1000 else ""),  # Truncate long results
                        ""
                    ])
        
        # Add any other context data
        for key, value in context_data.items():
            if not key.startswith("command_"):
                prompt_parts.extend([
                    f"**{key.replace('_', ' ').title()}:**",
                    value[:800] + ("..." if len(value) > 800 else ""),
                    ""
                ])
        
        # Add instructions
        prompt_parts.extend([
            "**Instructions:**",
            f"Based on the above context information and command results, please answer: {original_prompt}",
            "Use the provided context to give accurate, specific answers about this repository/project.",
            ""
        ])
        
        return "\n".join(prompt_parts)
    
    # Command execution methods
    def _execute_analyze_repository_context(self) -> str:
        """Execute repository context analysis."""
        from src.tools.context import analyze_repository_context
        return analyze_repository_context()
    
    def _execute_analyze_virtual_repository(self) -> str:
        """Execute virtual repository context analysis."""
        from src.tools.context import get_virtual_repository_context
        
        # Check if we have repository information from the current session
        repo_url = getattr(self, '_current_repo_url', None)
        if repo_url:
            virtual_context = get_virtual_repository_context(repo_url)
            if virtual_context:
                return virtual_context
        
        # Fallback to regular repository analysis
        return self._execute_analyze_repository_context()
    
    def _execute_analyze_repo_languages(self) -> str:
        """Execute repository language analysis."""
        from src.tools.context import analyze_repo_languages
        return analyze_repo_languages()
    
    def _execute_analyze_repo_directories(self) -> str:
        """Execute repository directory analysis."""
        from src.tools.context import analyze_repo_directories
        return analyze_repo_directories()
    
    def _execute_get_repository_state(self) -> str:
        """Execute repository state retrieval."""
        from src.tools.context import get_repository_state
        return get_repository_state()
    
    def _execute_search_files(self, query: str = None) -> str:
        """Execute file search."""
        from src.tools.context import search_files, get_repository_state
        if query:
            return search_files(query)
        else:
            return get_repository_state()  # Fallback to general state
    
    # Context provider methods (legacy - kept for compatibility)
    def _get_repository_context(self) -> str:
        """Get repository context information."""
        try:
            return self._execute_analyze_repository_context()
        except Exception as e:
            return f"Repository context not available: {e}"
    
    def _get_file_context(self, file_path: str = None) -> str:
        """Get file context information."""
        try:
            from src.tools.context import get_file_context
            return get_file_context(file_path) if file_path else "No file specified"
        except Exception as e:
            return f"File context not available: {e}"
    
    def _get_language_context(self) -> str:
        """Get language analysis context."""
        try:
            return self._execute_analyze_repo_languages()
        except Exception as e:
            return f"Language context not available: {e}"
    
    def _get_directory_context(self) -> str:
        """Get directory structure context."""
        try:
            return self._execute_analyze_repo_directories()
        except Exception as e:
            return f"Directory context not available: {e}"
    
    def _get_technology_context(self) -> str:
        """Get technology stack context."""
        try:
            return self._execute_analyze_repository_context()
        except Exception as e:
            return f"Technology context not available: {e}"


# Global instance for system-wide use
_prompt_interceptor_instance = None


def get_prompt_interceptor() -> PromptInterceptor:
    """Get the global prompt interceptor instance."""
    global _prompt_interceptor_instance
    if _prompt_interceptor_instance is None:
        _prompt_interceptor_instance = PromptInterceptor()
    return _prompt_interceptor_instance


def _ensure_virtual_repository(repo_url: str):
    """Ensure virtual repository is downloaded and cached for the given URL."""
    try:
        from src.core.helpers import get_virtual_repository, download_github_zip_to_memory
        from pathlib import Path
        
        print(f"🔍 Checking virtual repository for: {repo_url}")
        
        # Check if already cached
        virtual_repo = get_virtual_repository(repo_url)
        if virtual_repo:
            print(f"✅ Virtual repository already cached ({len(virtual_repo.files)} files)")
            return  # Already available
        
        print(f"📥 Downloading and caching repository...")
        # Use the data directory for persistence
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        success = download_github_zip_to_memory(repo_url, data_dir)
        if success:
            print(f"✅ Repository downloaded and cached successfully")
        else:
            print(f"⚠️  Failed to download repository: {repo_url}")
            
    except Exception as e:
        print(f"⚠️  Error ensuring virtual repository: {e}")
        import traceback
        traceback.print_exc()


def intercept_and_enhance_prompt(prompt: str, working_directory: str = None, repository_url: str = None, mode: InterceptionMode = InterceptionMode.SMART) -> SupplementedPrompt:
    """Convenience function to intercept and enhance a prompt."""
    interceptor = get_prompt_interceptor()
    if repository_url:
        interceptor._current_repo_url = repository_url
        # Ensure virtual repository is initialized for the URL
        _ensure_virtual_repository(repository_url)
    return interceptor.intercept_and_enhance(prompt, working_directory, mode)


if __name__ == "__main__":
    # Test the prompt interceptor
    interceptor = PromptInterceptor()
    
    test_prompts = [
        "analyze repository structure",
        "what language is this project using", 
        "analyze the codebase",
        "show me the project overview",
        "how is this project organized",
        "what technologies are used here",
        "find files containing authentication",
        "help me debug this error",
        "create a new feature",
        "random question about math"
    ]
    
    print("🚀 Prompt Interceptor Test")
    print("=" * 60)
    
    for prompt in test_prompts:
        print(f"\n📝 Testing prompt: '{prompt}'")
        
        # Analyze context
        context = interceptor._analyze_prompt_context(prompt)
        print(f"   🎯 Detected: {context.detected_intent}")
        print(f"   📊 Confidence: {context.confidence:.2f}")
        print(f"   🔧 Commands: {context.supplemental_commands}")
        print(f"   📋 Context needs: {context.required_context}")
