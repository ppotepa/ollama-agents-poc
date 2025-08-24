"""Collaborative Agent System - Iterative back-and-forth between main agent and interceptor."""
from __future__ import annotations

import time
import json
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

from src.agents.interceptor.agent import InterceptorAgent, CommandRecommendation
from src.core.prompt_interceptor import InterceptionMode
from src.tools.file_ops import list_files, read_file, get_file_info
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
        
    def execute_command_safely(self, command: str, args: Dict[str, Any]) -> Tuple[bool, str, float]:
        """Execute a command safely and return success, result, and execution time."""
        start_time = time.time()
        
        try:
            if command == "list_files":
                directory = args.get("directory", ".")
                pattern = args.get("pattern", "*")
                result = list_files(directory, pattern, detailed=True)
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
    
    def ask_interceptor_for_next_steps(self, current_state: str, previous_results: str) -> List[CommandRecommendation]:
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
                    
                    # Track discovered files
                    if rec.command == "list_files" and result:
                        # Extract file names from detailed listing
                        for line in result.split('\\n'):
                            if line.strip() and not line.startswith(('Date/Time', '-', '📁', '📊', '📏')):
                                parts = line.split()
                                if len(parts) >= 4 and not parts[3].endswith('/'):
                                    self.current_context.discovered_files.append(parts[3])
                else:
                    print(f"   ❌ Failed ({exec_time:.3f}s): {result}")
            
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
        
        final_prompt = f"""Based on all the collaborative analysis, provide a comprehensive answer to the original query:

ORIGINAL QUERY: {query}

EXECUTION SUMMARY:
- Total steps executed: {self.current_context.current_step}
- Commands executed: {', '.join(self.current_context.executed_commands)}
- Files discovered: {len(self.current_context.discovered_files)}

COLLECTED INFORMATION:
{chr(10).join([f"Step {i+1}: {result}" for i, result in enumerate(self.current_context.intermediate_results.values())])}

Please provide a detailed, comprehensive answer to the original query using all the information gathered."""

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
