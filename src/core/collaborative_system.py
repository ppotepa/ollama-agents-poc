"""Collaborative Agent System - Iterative back-and-forth between main agent and interceptor."""
from __future__ import annotations

import time
import json
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

from src.agents.interceptor.agent import InterceptorAgent, CommandRecommendation
from src.core.prompt_interceptor import InterceptionMode
from src.tools.file_ops import list_files, list_files_recurse, read_file, get_file_info
import subprocess
import os


class ExecutionNodeType(Enum):
    """Types of nodes in the execution tree."""
    USER_QUERY = "user_query"
    INTERCEPTOR_ANALYSIS = "interceptor_analysis"
    COMMAND_EXECUTION = "command_execution"
    AGENT_RESPONSE = "agent_response"
    FOLLOW_UP_QUERY = "follow_up_query"
    FINAL_RESULT = "final_result"


@dataclass
class ExecutionNode:
    """A node in the execution tree representing a step in the collaborative process."""
    node_type: ExecutionNodeType
    content: str
    metadata: Dict[str, Any]
    children: List['ExecutionNode']
    timestamp: float
    execution_time: float = 0.0
    success: bool = True
    parent: Optional['ExecutionNode'] = None

    def add_child(self, child: 'ExecutionNode') -> 'ExecutionNode':
        """Add a child node and set parent reference."""
        child.parent = self
        self.children.append(child)
        return child


@dataclass
class CollaborationContext:
    """Context shared between agents during collaboration."""
    original_query: str
    current_step: int
    max_steps: int
    execution_tree: ExecutionNode
    discovered_files: List[str]
    executed_commands: List[str]
    intermediate_results: Dict[str, Any]
    working_directory: str


class CollaborativeAgentSystem:
    """System that orchestrates iterative collaboration between main agent and interceptor."""
    
    def __init__(self, main_agent, interceptor_agent: InterceptorAgent, max_iterations: int = 5):
        self.main_agent = main_agent
        self.interceptor_agent = interceptor_agent
        self.max_iterations = max_iterations
        self.current_context = None
        self._agent_factory = None
        self._original_main_agent_id = None
        
        # Store original agent info for potential switching
        if hasattr(main_agent, '_model_id'):
            self._original_main_agent_id = main_agent._model_id
        elif hasattr(main_agent, 'agent_id'):
            self._original_main_agent_id = main_agent.agent_id
        
    def _get_agent_factory(self):
        """Get or create the agent factory."""
        if self._agent_factory is None:
            from src.core.agent_factory import get_agent_factory
            # Default to streaming=True for collaborative system
            self._agent_factory = get_agent_factory(streaming=True)
        return self._agent_factory
    
    def _should_switch_agent(self, query: str, current_iteration: int) -> Optional[str]:
        """Determine if we should switch to a different agent based on query analysis."""
        # Only consider switching after the first iteration
        if current_iteration <= 1:
            return None
            
        # Use agent resolver to find optimal agent for the query
        try:
            from src.core.agent_resolver import create_agent_resolver
            resolver = create_agent_resolver(max_size_b=14.0)
            recommended_agent = resolver.resolve_best_agent(query)
            
            # Check if recommended agent is different from current
            current_agent_id = getattr(self.main_agent, '_model_id', self._original_main_agent_id)
            
            if recommended_agent and recommended_agent != current_agent_id:
                print(f"🔄 Considering agent switch: {current_agent_id} -> {recommended_agent}")
                return recommended_agent
                
        except Exception as e:
            print(f"⚠️ Agent switch analysis failed: {e}")
            
        return None
    
    def _switch_main_agent(self, new_agent_id: str) -> bool:
        """Switch the main agent to a different model."""
        try:
            factory = self._get_agent_factory()
            new_agent = factory.switch_agent(self.main_agent, new_agent_id)
            
            if new_agent:
                self.main_agent = new_agent
                print(f"✅ Successfully switched to agent: {new_agent_id}")
                return True
            else:
                print(f"❌ Failed to switch to agent: {new_agent_id}")
                return False
                
        except Exception as e:
            print(f"❌ Error switching agent: {e}")
            return False
    
    def _handle_model_swap_request(self, agent_response: str) -> bool:
        """Handle a model swap request from the agent response."""
        try:
            import json
            import re
            
            # Extract the swap request data
            match = re.search(r'SWAP_REQUEST_DATA:\s*(\{.*?\})', agent_response, re.DOTALL)
            if not match:
                print("⚠️ Model swap request detected but couldn't parse data")
                return False
            
            swap_data = json.loads(match.group(1))
            reason = swap_data.get('reason', 'No reason provided')
            preferred_model = swap_data.get('preferred_model')
            task_type = swap_data.get('task_type')
            
            print(f"🔄 MODEL SWAP REQUEST DETECTED:")
            print(f"   Reason: {reason}")
            print(f"   Preferred model: {preferred_model or 'Auto-select'}")
            print(f"   Task type: {task_type or 'General'}")
            
            # Determine the best model to switch to
            target_model = preferred_model
            if not target_model:
                # Auto-select based on task type
                if task_type and task_type.lower() == 'coding':
                    target_model = 'qwen2.5-coder:7b'
                elif task_type and task_type.lower() == 'analysis':
                    target_model = 'qwen2.5:7b-instruct-q4_K_M'
                else:
                    # Use agent resolver for intelligent selection
                    from src.core.agent_resolver import create_agent_resolver
                    resolver = create_agent_resolver(max_size_b=14.0)
                    original_query = getattr(self.current_context, 'original_query', reason)
                    target_model = resolver.resolve_best_agent(original_query)
            
            if target_model:
                current_agent_id = getattr(self.main_agent, '_model_id', 'unknown')
                if target_model != current_agent_id:
                    print(f"🎯 Proceeding with model swap: {current_agent_id} -> {target_model}")
                    if self._switch_main_agent(target_model):
                        print(f"✅ Model swap successful! Continuing with {target_model}")
                        return True
                    else:
                        print(f"❌ Model swap failed, continuing with {current_agent_id}")
                        return False
                else:
                    print(f"ℹ️ Already using the requested model: {target_model}")
                    return False
            else:
                print("❌ Could not determine target model for swap")
                return False
                
        except Exception as e:
            print(f"❌ Error handling model swap request: {e}")
            return False
    
    def _switch_interceptor_agent(self, new_agent_id: str) -> bool:
        """Switch the interceptor agent to a different model."""
        try:
            factory = self._get_agent_factory()
            
            # Create new interceptor agent
            new_base_agent = factory.create_agent(new_agent_id)
            if new_base_agent:
                # Create new InterceptorAgent wrapper
                from src.agents.interceptor.agent import InterceptorAgent
                self.interceptor_agent = InterceptorAgent("interceptor", {})
                self.interceptor_agent.base_agent = new_base_agent
                print(f"✅ Successfully switched interceptor to: {new_agent_id}")
                return True
            else:
                print(f"❌ Failed to create interceptor agent: {new_agent_id}")
                return False
                
        except Exception as e:
            print(f"❌ Error switching interceptor agent: {e}")
            return False
        
    def execute_command_safely(self, command: str, args: Dict[str, Any]) -> Tuple[bool, str, float]:
        """Execute a command safely and return success, result, and execution time."""
        start_time = time.time()
        
        try:
            if command == "list_files":
                directory = args.get("directory", ".")
                pattern = args.get("pattern", "*")
                result = list_files(directory, pattern, detailed=True)
                return True, result, time.time() - start_time
                
            elif command == "list_files_recurse":
                directory = args.get("directory", ".")
                pattern = args.get("pattern", "*")
                max_depth = args.get("max_depth", 10)
                detailed = args.get("detailed", True)
                result = list_files_recurse(directory, pattern, detailed, max_depth)
                return True, result, time.time() - start_time
                
            elif command == "read_file":
                filepath = args.get("filepath", "")
                if not filepath:
                    return False, "No filepath provided", time.time() - start_time
                result = read_file(filepath, show_info=True)
                return True, result, time.time() - start_time
                
            elif command == "get_file_info":
                filepath = args.get("filepath", "")
                if not filepath:
                    return False, "No filepath provided", time.time() - start_time
                result = get_file_info(filepath)
                return True, result, time.time() - start_time
                
            elif command == "run_powershell":
                cmd = args.get("command", "")
                if not cmd:
                    return False, "No command provided", time.time() - start_time
                
                # Execute PowerShell command safely
                try:
                    result = subprocess.run(
                        ["powershell", "-Command", cmd],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=self.current_context.working_directory if self.current_context else "."
                    )
                    output = result.stdout
                    if result.stderr:
                        output += f"\\nErrors: {result.stderr}"
                    return result.returncode == 0, output, time.time() - start_time
                except subprocess.TimeoutExpired:
                    return False, "Command timed out", time.time() - start_time
                except Exception as e:
                    return False, f"Command execution failed: {e}", time.time() - start_time
                    
            elif command == "run_cmd":
                cmd = args.get("command", "")
                if not cmd:
                    return False, "No command provided", time.time() - start_time
                
                # Execute command line safely
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        shell=True,
                        cwd=self.current_context.working_directory if self.current_context else "."
                    )
                    output = result.stdout
                    if result.stderr:
                        output += f"\\nErrors: {result.stderr}"
                    return result.returncode == 0, output, time.time() - start_time
                except subprocess.TimeoutExpired:
                    return False, "Command timed out", time.time() - start_time
                except Exception as e:
                    return False, f"Command execution failed: {e}", time.time() - start_time
                    
            else:
                return False, f"Unknown command: {command}", time.time() - start_time
                
        except Exception as e:
            return False, f"Error executing {command}: {e}", time.time() - start_time
    
    def analyze_command_results_for_followup(self, command: str, result: str, context: CollaborationContext) -> List[CommandRecommendation]:
        """Analyze command results to suggest intelligent follow-up commands."""
        if not result or len(result.strip()) == 0:
            return []
            
        analysis_prompt = f"""Analyze the following command execution result and suggest intelligent follow-up commands:

COMMAND EXECUTED: {command}
RESULT: {result[:1000]}{'...' if len(result) > 1000 else ''}

CURRENT CONTEXT:
- Original query: {context.original_query}
- Step: {context.current_step}/{context.max_steps}
- Files discovered: {', '.join(context.discovered_files[:5])}
- Previous commands: {', '.join(context.executed_commands)}

Based on this result, what are the most logical next steps? Consider:
1. If files were listed, should we read specific interesting files?
2. If a file was read, should we examine related files or run analysis commands?
3. If information seems incomplete, what additional commands would provide clarity?
4. If errors occurred, what commands might help resolve them?

Provide 1-3 specific, actionable follow-up command recommendations in JSON format:
{{
    "recommendations": [
        {{
            "command": "command_name",
            "args": {{"key": "value"}},
            "reasoning": "why this command would be helpful",
            "confidence": 0.8,
            "priority": "high|medium|low"
        }}
    ]
}}"""

        try:
            # Use interceptor's LLM for analysis
            response = self.interceptor_agent.analyze_prompt_with_llm(analysis_prompt)
            
            # Parse JSON response
            try:
                parsed = json.loads(response)
                recommendations = []
                
                for rec in parsed.get("recommendations", []):
                    command_name = rec.get("command", "")
                    confidence = rec.get("confidence", 0.5)
                    reasoning = rec.get("reasoning", "")
                    priority = rec.get("priority", "medium")
                    
                    # Adjust confidence based on priority
                    if priority == "high":
                        confidence = min(confidence + 0.2, 1.0)
                    elif priority == "low":
                        confidence = max(confidence - 0.2, 0.1)
                    
                    recommendations.append(CommandRecommendation(
                        command=command_name,
                        confidence=confidence,
                        description=f"[Follow-up] {reasoning}",
                        category="follow_up_analysis",
                        required_context=[]
                    ))
                    
                return recommendations
                
            except json.JSONDecodeError:
                # Fallback: Create logical follow-ups based on command type
                return self._create_fallback_followups(command, result, context)
                
        except Exception as e:
            print(f"⚠️  Error analyzing command results for follow-up: {e}")
            return self._create_fallback_followups(command, result, context)
    
    def _create_fallback_followups(self, command: str, result: str, context: CollaborationContext) -> List[CommandRecommendation]:
        """Create logical follow-up commands based on command type and result."""
        recommendations = []
        
        if command == "list_files" and result:
            # If we listed files and found directories, suggest recursive listing
            if "<DIR>" in result or any(line.endswith('/') for line in result.split('\n')):
                recommendations.append(CommandRecommendation(
                    command="list_files_recurse",
                    confidence=0.8,
                    description="Recursively explore subdirectories for comprehensive file listing",
                    category="follow_up_analysis",
                    required_context=[]
                ))
            
            # Suggest reading interesting files
            lines = result.split('\n')
            for line in lines:
                if any(ext in line.lower() for ext in ['.py', '.js', '.ts', '.md', '.txt', '.json', '.yml', '.yaml']):
                    parts = line.split()
                    if len(parts) >= 4:
                        filename = parts[3]
                        recommendations.append(CommandRecommendation(
                            command="read_file",
                            confidence=0.7,
                            description=f"Read {filename} to understand its contents",
                            category="follow_up_analysis",
                            required_context=[]
                        ))
                        if len(recommendations) >= 3:  # Increased to allow for both recursive and read suggestions
                            break
                            
        elif command == "list_files_recurse" and result:
            # If we did recursive listing, suggest reading key files found
            lines = result.split('\n')
            key_files = []
            for line in lines:
                if any(keyword in line.lower() for keyword in ['main.py', 'readme', 'config', 'requirements', '__init__']):
                    # Extract file path from recursive listing
                    if '📄' in line:
                        parts = line.split('📄')
                        if len(parts) > 1:
                            filepath = parts[1].strip().split()[0]
                            key_files.append(filepath)
            
            for filepath in key_files[:2]:  # Suggest reading top 2 key files
                recommendations.append(CommandRecommendation(
                    command="read_file",
                    confidence=0.8,
                    description=f"Read {filepath} - appears to be a key project file",
                    category="follow_up_analysis",
                    required_context=[]
                ))
                            
        elif command == "read_file" and result:
            # If we read a file, suggest related files or analysis
            if any(keyword in result.lower() for keyword in ['import', 'require', 'include']):
                recommendations.append(CommandRecommendation(
                    command="list_files",
                    confidence=0.6,
                    description="List files to find dependencies mentioned in this file",
                    category="follow_up_analysis", 
                    required_context=[]
                ))
                
        elif command == "get_file_info" and result:
            # If we got file info, suggest reading the file
            recommendations.append(CommandRecommendation(
                command="read_file",
                confidence=0.8,
                description="Read the file content to analyze its implementation",
                category="follow_up_analysis",
                required_context=[]
            ))
            
        return recommendations
    
    def ask_interceptor_for_next_steps(self, current_state: str, previous_results: str) -> List[CommandRecommendation]:
        """Ask the interceptor agent for next step recommendations."""
        analysis_prompt = f"""You are an intelligent command interceptor helping to analyze a user query step by step.

CURRENT STATE:
{current_state}

PREVIOUS RESULTS:
{previous_results}

CONTEXT:
- Working directory: {self.current_context.working_directory}
- Step {self.current_context.current_step} of {self.current_context.max_steps}
- Files discovered so far: {', '.join(self.current_context.discovered_files[:10])}
- Commands executed: {', '.join(self.current_context.executed_commands)}

What should we do next to gather more information or complete the analysis?
Focus on:
1. File system exploration (list_files, list_files_recurse, read_file, get_file_info)
2. Command execution (run_powershell, run_cmd)
3. Data analysis based on discovered information

Available commands:
- list_files: List files in a single directory with detailed info
- list_files_recurse: Recursively list all files in directory tree
- read_file: Read and display file contents
- get_file_info: Get detailed metadata about a specific file
- run_powershell: Execute PowerShell commands
- run_cmd: Execute command prompt commands

Recommend 1-3 specific next steps."""

        try:
            # Use the interceptor's LLM to get recommendations
            response = self.interceptor_agent.analyze_prompt_with_llm(analysis_prompt)
            
            # Parse the JSON response to extract recommendations
            try:
                parsed = json.loads(response)
                recommendations = []
                
                for rec in parsed.get("recommendations", []):
                    recommendations.append(CommandRecommendation(
                        command=rec.get("command", ""),
                        confidence=rec.get("confidence", 0.5),
                        description=rec.get("reasoning", ""),
                        category="iterative_analysis",
                        required_context=[]
                    ))
                    
                return recommendations
                
            except json.JSONDecodeError:
                # Fallback: create simple recommendations based on context
                if self.current_context.current_step == 1:
                    return [CommandRecommendation("list_files", 0.9, "Start by exploring directory structure", "exploration", [])]
                else:
                    return [CommandRecommendation("read_file", 0.7, "Analyze discovered files", "analysis", [])]
                    
        except Exception as e:
            print(f"⚠️  Error getting interceptor recommendations: {e}")
            return []
        """Ask the interceptor agent what to do next based on current state."""
        analysis_prompt = f"""Based on the current analysis state, recommend the next steps:

CURRENT STATE:
{current_state}

PREVIOUS RESULTS:
{previous_results}

CONTEXT:
- Working directory: {self.current_context.working_directory}
- Step {self.current_context.current_step} of {self.current_context.max_steps}
- Files discovered so far: {', '.join(self.current_context.discovered_files[:10])}
- Commands executed: {', '.join(self.current_context.executed_commands)}

What should we do next to gather more information or complete the analysis?
Focus on:
1. File system exploration (list_files, read_file, get_file_info)
2. Command execution (run_powershell, run_cmd)
3. Data analysis based on discovered information

Recommend 1-3 specific next steps."""

        try:
            # Use the interceptor's LLM to get recommendations
            response = self.interceptor_agent.analyze_prompt_with_llm(analysis_prompt)
            
            # Parse the JSON response to extract recommendations
            try:
                parsed = json.loads(response)
                recommendations = []
                
                for rec in parsed.get("recommendations", []):
                    recommendations.append(CommandRecommendation(
                        command=rec.get("command", ""),
                        confidence=rec.get("confidence", 0.5),
                        description=rec.get("reasoning", ""),
                        category="iterative_analysis",
                        required_context=[]
                    ))
                    
                return recommendations
                
            except json.JSONDecodeError:
                # Fallback: create simple recommendations based on context
                if self.current_context.current_step == 1:
                    return [CommandRecommendation("list_files", 0.9, "Start by exploring directory structure", "exploration", [])]
                else:
                    return [CommandRecommendation("read_file", 0.7, "Analyze discovered files", "analysis", [])]
                    
        except Exception as e:
            print(f"⚠️  Error getting interceptor recommendations: {e}")
            return []
    
    def collaborative_execution(self, query: str, working_directory: str = ".", max_steps: int = None) -> Dict[str, Any]:
        """Execute a query using collaborative back-and-forth between agents."""
        if max_steps is None:
            max_steps = self.max_iterations
            
        print(f"🤝 Starting collaborative execution: '{query}'")
        print(f"📁 Working directory: {working_directory}")
        print(f"🔄 Max steps: {max_steps}")
        print("=" * 60)
        
        # Initialize execution tree
        root_node = ExecutionNode(
            node_type=ExecutionNodeType.USER_QUERY,
            content=query,
            metadata={"query": query, "working_directory": working_directory},
            children=[],
            timestamp=time.time()
        )
        
        # Initialize collaboration context
        self.current_context = CollaborationContext(
            original_query=query,
            current_step=0,
            max_steps=max_steps,
            execution_tree=root_node,
            discovered_files=[],
            executed_commands=[],
            intermediate_results={},
            working_directory=working_directory
        )
        
        current_state = f"Initial query: {query}"
        previous_results = ""
        current_node = root_node
        
        # Iterative collaboration loop
        for step in range(1, max_steps + 1):
            self.current_context.current_step = step
            print(f"\\n🔄 STEP {step}/{max_steps}")
            print("-" * 40)
            
            # Check if we should switch agents based on query analysis
            if step > 1:  # Only consider switching after first step
                recommended_agent = self._should_switch_agent(query, step)
                if recommended_agent:
                    # Ask user or make automatic decision
                    current_agent_id = getattr(self.main_agent, '_model_id', self._original_main_agent_id)
                    print(f"🎯 Agent switching opportunity detected:")
                    print(f"   Current: {current_agent_id}")
                    print(f"   Recommended: {recommended_agent}")
                    print(f"   Proceeding with agent switch for better performance...")
                    
                    if self._switch_main_agent(recommended_agent):
                        print(f"🔄 Continuing with new agent: {recommended_agent}")
                    else:
                        print(f"⚠️ Continuing with current agent: {current_agent_id}")
            
            # 1. Ask interceptor for next steps
            print(f"🧠 Asking interceptor for recommendations...")
            start_time = time.time()
            recommendations = self.ask_interceptor_for_next_steps(current_state, previous_results)
            analysis_time = time.time() - start_time
            
            # Create analysis node
            analysis_node = current_node.add_child(ExecutionNode(
                node_type=ExecutionNodeType.INTERCEPTOR_ANALYSIS,
                content=f"Step {step} analysis",
                metadata={
                    "recommendations": [
                        {"command": r.command, "confidence": r.confidence, "description": r.description}
                        for r in recommendations
                    ],
                    "step": step
                },
                children=[],
                timestamp=time.time(),
                execution_time=analysis_time
            ))
            
            if not recommendations:
                print(f"⚠️  No recommendations from interceptor, stopping")
                break
                
            print(f"📋 Received {len(recommendations)} recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec.command} (confidence: {rec.confidence:.2%}) - {rec.description}")
            
            # 2. Execute recommended commands
            step_results = []
            for rec in recommendations[:2]:  # Execute top 2 recommendations
                print(f"\\n🔧 Executing: {rec.command}")
                
                # Determine command arguments based on command and context
                args = self._determine_command_args(rec, self.current_context)
                print(f"   📝 Args: {args}")
                
                # Execute command
                success, result, exec_time = self.execute_command_safely(rec.command, args)
                
                # Create execution node
                exec_node = analysis_node.add_child(ExecutionNode(
                    node_type=ExecutionNodeType.COMMAND_EXECUTION,
                    content=result[:500] + "..." if len(result) > 500 else result,
                    metadata={
                        "command": rec.command,
                        "args": args,
                        "success": success,
                        "full_result": result
                    },
                    children=[],
                    timestamp=time.time(),
                    execution_time=exec_time,
                    success=success
                ))
                
                if success:
                    print(f"   ✅ Success ({exec_time:.3f}s): {len(result)} chars")
                    step_results.append(f"{rec.command}: {result[:200]}...")
                    self.current_context.executed_commands.append(rec.command)
                    
                    # Store full result in intermediate_results for final prompt
                    result_key = f"step_{self.current_context.current_step}_{rec.command}"
                    self.current_context.intermediate_results[result_key] = {
                        "command": rec.command,
                        "args": args,
                        "success": True,
                        "result": result,
                        "execution_time": exec_time,
                        "step": self.current_context.current_step
                    }
                    
                    # Track discovered files
                    if rec.command == "list_files" and result:
                        # Extract file names from detailed listing
                        for line in result.split('\\n'):
                            if line.strip() and not line.startswith(('Date/Time', '-', '📁', '📊', '📏')):
                                parts = line.split()
                                if len(parts) >= 4 and not parts[3].endswith('/'):
                                    self.current_context.discovered_files.append(parts[3])
                    
                    elif rec.command == "list_files_recurse" and result:
                        # Extract file paths from recursive listing
                        for line in result.split('\\n'):
                            if '📄' in line and line.strip():
                                # Parse recursive listing format: "📄 path/to/file.py"
                                parts = line.split('📄')
                                if len(parts) > 1:
                                    filepath = parts[1].strip().split()[0]
                                    if filepath and not filepath.startswith(('Date/Time', '-', '=', '📊')):
                                        self.current_context.discovered_files.append(filepath)
                    
                    # Analyze command results for intelligent follow-ups
                    print(f"   🔍 Analyzing results for follow-up opportunities...")
                    followup_recommendations = self.analyze_command_results_for_followup(rec.command, result, self.current_context)
                    
                    if followup_recommendations:
                        print(f"   💡 Found {len(followup_recommendations)} follow-up opportunities:")
                        for i, followup in enumerate(followup_recommendations, 1):
                            print(f"      {i}. {followup.command} - {followup.description}")
                        
                        # Execute the highest confidence follow-up (if we have time in this step)
                        best_followup = max(followup_recommendations, key=lambda x: x.confidence)
                        if best_followup.confidence > 0.7 and len(step_results) < 3:  # Limit to 3 commands per step
                            print(f"   🚀 Executing best follow-up: {best_followup.command}")
                            
                            followup_args = self._determine_command_args(best_followup, self.current_context)
                            followup_success, followup_result, followup_time = self.execute_command_safely(best_followup.command, followup_args)
                            
                            if followup_success:
                                print(f"   ✅ Follow-up success ({followup_time:.3f}s): {len(followup_result)} chars")
                                step_results.append(f"[Follow-up] {best_followup.command}: {followup_result[:200]}...")
                                self.current_context.executed_commands.append(f"[follow-up] {best_followup.command}")
                                
                                # Store follow-up result in intermediate_results too
                                followup_key = f"step_{self.current_context.current_step}_{best_followup.command}_followup"
                                self.current_context.intermediate_results[followup_key] = {
                                    "command": best_followup.command,
                                    "args": followup_args,
                                    "success": True,
                                    "result": followup_result,
                                    "execution_time": followup_time,
                                    "step": self.current_context.current_step,
                                    "type": "follow_up"
                                }
                                
                                # Track discovered files from follow-up results too
                                if best_followup.command == "list_files_recurse" and followup_result:
                                    for line in followup_result.split('\\n'):
                                        if '📄' in line and line.strip():
                                            parts = line.split('📄')
                                            if len(parts) > 1:
                                                filepath = parts[1].strip().split()[0]
                                                if filepath and not filepath.startswith(('Date/Time', '-', '=', '📊')):
                                                    self.current_context.discovered_files.append(filepath)
                            else:
                                print(f"   ❌ Follow-up failed: {followup_result}")
                
                else:
                    print(f"   ❌ Failed ({exec_time:.3f}s): {result}")
                    # Store failed result too for context
                    result_key = f"step_{self.current_context.current_step}_{rec.command}_failed"
                    self.current_context.intermediate_results[result_key] = {
                        "command": rec.command,
                        "args": args,
                        "success": False,
                        "result": result,
                        "execution_time": exec_time,
                        "step": self.current_context.current_step
                    }
            
            # 3. Ask main agent to analyze results and determine next steps
            print(f"\\n🤖 Main agent analyzing results...")
            agent_prompt = f"""Based on the following execution results, analyze what we've learned and determine if we need more information:

ORIGINAL QUERY: {query}

STEP {step} RESULTS:
{chr(10).join(step_results)}

CONTEXT:
- Files discovered: {', '.join(self.current_context.discovered_files[:10])}
- Commands executed: {', '.join(self.current_context.executed_commands)}

ANALYSIS NEEDED:
1. What have we learned so far?
2. Is this sufficient to answer the original query?
3. What specific information is still missing?
4. Should we continue or can we provide a final answer?

Provide a brief analysis and indicate if more exploration is needed."""

            try:
                if hasattr(self.main_agent, 'run'):
                    agent_response = self.main_agent.run(agent_prompt)
                elif hasattr(self.main_agent, 'run_query'):
                    agent_response = self.main_agent.run_query(agent_prompt)
                elif hasattr(self.main_agent, 'invoke'):
                    agent_response = self.main_agent.invoke(agent_prompt)
                else:
                    agent_response = f"Agent {self.main_agent.agent_name} processed the prompt but cannot provide response"
                    
                print(f"🎯 Agent analysis: {agent_response[:300]}...")
                
                # Check for model swap requests in agent response
                if "SWAP_REQUEST_DATA:" in agent_response:
                    self._handle_model_swap_request(agent_response)
                
                # Create agent response node
                response_node = analysis_node.add_child(ExecutionNode(
                    node_type=ExecutionNodeType.AGENT_RESPONSE,
                    content=agent_response,
                    metadata={"step": step, "analysis": agent_response},
                    children=[],
                    timestamp=time.time()
                ))
                
                # Update state for next iteration
                current_state = f"Step {step} completed. Agent analysis: {agent_response[:200]}..."
                previous_results = chr(10).join(step_results)
                current_node = response_node
                
                # Check if agent thinks we're done
                if any(phrase in agent_response.lower() for phrase in [
                    "sufficient", "complete", "final answer", "no more", "enough information"
                ]):
                    print(f"🎯 Agent indicates analysis is sufficient, proceeding to final answer")
                    break
                    
            except Exception as e:
                print(f"⚠️  Error getting agent analysis: {e}")
                break
        
        # Generate final comprehensive answer
        print(f"\\n🎯 GENERATING FINAL ANSWER")
        print("-" * 40)
        
        # Format collected information from all command executions
        collected_info = []
        for key, result_data in self.current_context.intermediate_results.items():
            command = result_data.get("command", "unknown")
            success = result_data.get("success", False)
            result = result_data.get("result", "")
            step = result_data.get("step", 0)
            result_type = result_data.get("type", "primary")
            
            status = "✅" if success else "❌"
            type_label = "[Follow-up]" if result_type == "follow_up" else ""
            
            # Truncate very long results but include more than before
            display_result = result[:1000] + "..." if len(result) > 1000 else result
            
            collected_info.append(f"""Step {step} {type_label} - {command} {status}:
{display_result}
""")
        
        # Create enhanced final prompt with all collected data
        final_prompt = f"""Based on the comprehensive collaborative analysis, provide a detailed answer to the original query:

ORIGINAL QUERY: {query}

EXECUTION SUMMARY:
- Total steps executed: {self.current_context.current_step}
- Commands executed: {', '.join(self.current_context.executed_commands)}
- Files discovered: {len(self.current_context.discovered_files)}
- Working directory: {self.current_context.working_directory}

DETAILED COMMAND EXECUTION RESULTS:
{chr(10).join(collected_info)}

DISCOVERED FILES: {', '.join(self.current_context.discovered_files[:20])}

INSTRUCTIONS:
Using all the above information, provide a comprehensive, detailed answer to the original query. 
Include specific details from the command execution results, file listings, and any patterns you observe.
Be thorough and reference the actual data collected during the collaborative execution."""

        # Debug: Show the final prompt being sent
        print(f"🔍 Debug: Final prompt length: {len(final_prompt)} characters")
        print(f"📝 Debug: First 500 chars of final prompt:")
        print(final_prompt[:500])
        print("..." if len(final_prompt) > 500 else "")
        print("-" * 40)

        try:
            if hasattr(self.main_agent, 'run'):
                final_answer = self.main_agent.run(final_prompt)
            elif hasattr(self.main_agent, 'run_query'):
                final_answer = self.main_agent.run_query(final_prompt)
            elif hasattr(self.main_agent, 'invoke'):
                final_answer = self.main_agent.invoke(final_prompt)
            else:
                final_answer = "Unable to generate final answer - agent not available"
                
            # Create final result node
            final_node = current_node.add_child(ExecutionNode(
                node_type=ExecutionNodeType.FINAL_RESULT,
                content=final_answer,
                metadata={
                    "total_steps": self.current_context.current_step,
                    "commands_executed": self.current_context.executed_commands,
                    "files_discovered": len(self.current_context.discovered_files)
                },
                children=[],
                timestamp=time.time()
            ))
            
            print(f"✅ Final answer generated ({len(final_answer)} chars)")
            
        except Exception as e:
            final_answer = f"Error generating final answer: {e}"
        
        # Return comprehensive results
        return {
            "final_answer": final_answer,
            "execution_tree": root_node,
            "total_steps": self.current_context.current_step,
            "commands_executed": self.current_context.executed_commands,
            "files_discovered": self.current_context.discovered_files,
            "working_directory": working_directory,
            "success": True
        }
    
    def _determine_command_args(self, recommendation: CommandRecommendation, context: CollaborationContext) -> Dict[str, Any]:
        """Determine appropriate arguments for a command based on context."""
        if recommendation.command == "list_files":
            # Smart directory selection based on context
            if context.current_step == 1:
                return {"directory": ".", "pattern": "*"}
            elif "src" in context.discovered_files or any("src" in f for f in context.discovered_files):
                return {"directory": "src", "pattern": "*"}
            else:
                return {"directory": ".", "pattern": "*.py"}
                
        elif recommendation.command == "list_files_recurse":
            # Smart recursive listing with appropriate depth
            if context.current_step == 1:
                # First step: get overview with limited depth
                return {"directory": ".", "pattern": "*", "max_depth": 3, "detailed": True}
            elif any("src" in f for f in context.discovered_files):
                # Focus on source directory if discovered
                return {"directory": "src", "pattern": "*.py", "max_depth": 5, "detailed": True}
            else:
                # General recursive search for Python files
                return {"directory": ".", "pattern": "*.py", "max_depth": 10, "detailed": True}
                
        elif recommendation.command == "read_file":
            # Smart file selection
            if context.discovered_files:
                # Prioritize important files
                important_files = [f for f in context.discovered_files if any(name in f.lower() for name in ["main", "readme", "config", "requirements"])]
                if important_files:
                    return {"filepath": important_files[0]}
                else:
                    return {"filepath": context.discovered_files[0]}
            else:
                return {"filepath": "main.py"}  # fallback
                
        elif recommendation.command == "get_file_info":
            if context.discovered_files:
                return {"filepath": context.discovered_files[0]}
            else:
                return {"filepath": "."}
                
        elif recommendation.command == "run_powershell":
            # Smart PowerShell commands based on context
            if context.current_step == 1:
                return {"command": "Get-ChildItem -Name"}
            else:
                return {"command": "Get-Location"}
                
        elif recommendation.command == "run_cmd":
            if context.current_step == 1:
                return {"command": "dir"}
            else:
                return {"command": "echo Current directory exploration"}
        
        return {}


def create_collaborative_system(main_agent, max_iterations: int = 5) -> CollaborativeAgentSystem:
    """Create a collaborative agent system with interceptor."""
    # Create interceptor agent
    interceptor_config = {
        "name": "Collaborative Interceptor",
        "backend_image": "phi3:mini",
        "parameters": {"temperature": 0.1, "num_ctx": 2048}
    }
    
    interceptor_agent = InterceptorAgent("collaborative_interceptor", interceptor_config)
    
    return CollaborativeAgentSystem(main_agent, interceptor_agent, max_iterations)
