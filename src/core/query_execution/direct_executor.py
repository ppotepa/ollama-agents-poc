"""Direct query execution for single-shot queries."""

import os
from typing import Any, Dict, List, Optional


class DirectQueryExecutor:
    """Handles direct execution of single queries without collaboration."""

    def __init__(self):
        """Initialize the direct query executor."""
        self.execution_history = []

    def run_single_query(self, query: str, agent_name: str, connection_mode: str = "hybrid", 
                        repository_url: str = None, interception_mode: str = "smart", 
                        force_streaming: bool = False, collaborative: bool = False, 
                        max_iterations: int = 5, intelligent_investigation: bool = False, 
                        no_tools: bool = False) -> str:
        """Run a single query with various execution modes.
        
        Args:
            query: The query to execute
            agent_name: Name of the agent to use
            connection_mode: Connection mode (hybrid, direct, server)
            repository_url: Optional repository URL
            interception_mode: Prompt interception mode
            force_streaming: Force streaming output
            collaborative: Use collaborative mode
            max_iterations: Max iterations for collaborative mode
            intelligent_investigation: Use intelligent investigation
            no_tools: Disable tool usage
            
        Returns:
            Query result as a string
        """
        try:
            print(f"🎯 Starting single query execution")
            print(f"🔍 Query: {query[:100]}...")
            print(f"🤖 Agent: {agent_name}")
            print(f"🔗 Mode: {connection_mode}")
            print("=" * 60)

            # Record execution start
            execution_record = {
                "query": query,
                "agent_name": agent_name,
                "connection_mode": connection_mode,
                "start_time": __import__('time').time()
            }

            # Route to appropriate execution method
            if intelligent_investigation:
                from .intelligent_executor import IntelligentInvestigationExecutor
                executor = IntelligentInvestigationExecutor()
                result = executor.run_investigation(query, not no_tools)
                
            elif collaborative:
                from .collaborative_executor import CollaborativeQueryExecutor
                executor = CollaborativeQueryExecutor()
                result = executor.run_query(query, agent_name, max_iterations, not no_tools)
                
            elif connection_mode == "server":
                result = self._run_server_query(query, agent_name)
                
            elif connection_mode == "direct":
                result = self._run_direct_ollama_query(query, agent_name, no_tools)
                
            else:  # hybrid mode
                result = self._run_hybrid_query(query, agent_name, interception_mode, no_tools)

            # Record execution completion
            execution_record["end_time"] = __import__('time').time()
            execution_record["duration"] = execution_record["end_time"] - execution_record["start_time"]
            execution_record["success"] = True
            execution_record["result_length"] = len(result)
            self.execution_history.append(execution_record)

            return result

        except Exception as e:
            print(f"❌ Error in single query execution: {e}")
            import traceback
            traceback.print_exc()
            
            # Record failed execution
            execution_record["end_time"] = __import__('time').time()
            execution_record["duration"] = execution_record["end_time"] - execution_record["start_time"]
            execution_record["success"] = False
            execution_record["error"] = str(e)
            self.execution_history.append(execution_record)
            
            return f"Query execution failed: {str(e)}"

    def _run_hybrid_query(self, query: str, agent_name: str, interception_mode: str, no_tools: bool) -> str:
        """Run query in hybrid mode with prompt interception."""
        from src.core.prompt_interceptor import PromptInterceptor, InterceptionMode
        from src.core.agent_factory import get_agent_factory

        # Map interception mode string to enum
        mode_mapping = {
            "smart": InterceptionMode.SMART,
            "lightweight": InterceptionMode.LIGHTWEIGHT,
            "full": InterceptionMode.FULL,
            "disabled": InterceptionMode.DISABLED
        }
        
        interceptor_mode = mode_mapping.get(interception_mode, InterceptionMode.SMART)
        
        # Create agent
        agent_factory = get_agent_factory(streaming=True)
        agent = agent_factory.create_agent(agent_name)
        
        if agent is None:
            return f"❌ Failed to create agent: {agent_name}"

        # Apply prompt interception if not disabled
        if not no_tools and interceptor_mode != InterceptionMode.DISABLED:
            interceptor = PromptInterceptor()
            supplemented_prompt = interceptor.intercept_and_enhance(
                query, 
                os.getcwd(), 
                interceptor_mode
            )
            
            print(f"🎭 Prompt enhanced (mode: {interception_mode})")
            print(f"📊 Confidence: {supplemented_prompt.confidence:.2f}")
            
            # Use enhanced prompt
            enhanced_query = supplemented_prompt.enhanced_prompt
        else:
            enhanced_query = query

        # Execute query with agent
        return self._execute_with_agent(agent, enhanced_query)

    def _run_direct_ollama_query(self, query: str, agent_name: str, no_tools: bool) -> str:
        """Run query directly with Ollama without interception."""
        from src.core.agent_factory import get_agent_factory

        # Create agent
        agent_factory = get_agent_factory(streaming=True)
        agent = agent_factory.create_agent(agent_name)
        
        if agent is None:
            return f"❌ Failed to create agent: {agent_name}"

        return self._execute_with_agent(agent, query)

    def _run_server_query(self, query: str, agent_name: str) -> str:
        """Run query via server endpoint."""
        # This would implement server-based query execution
        # For now, fall back to direct execution
        return self._run_direct_ollama_query(query, agent_name, False)

    def _execute_with_agent(self, agent, query: str) -> str:
        """Execute query with a specific agent instance."""
        try:
            # Handle streaming vs non-streaming execution
            if hasattr(agent, 'stream_query'):
                # Streaming execution
                result_parts = []
                
                def collect_token_safely(token):
                    if token:
                        result_parts.append(str(token))
                        print(str(token), end='', flush=True)

                for token in agent.stream_query(query):
                    collect_token_safely(token)
                
                print()  # New line after streaming
                return ''.join(result_parts)
                
            else:
                # Non-streaming execution
                return agent.query(query)
                
        except Exception as e:
            return f"❌ Agent execution error: {str(e)}"

    def query_via_server(self, query: str, agent_name: str, server_url: str) -> str:
        """Query via server endpoint.
        
        Args:
            query: The query to send
            agent_name: Name of the agent to use
            server_url: Server URL to query
            
        Returns:
            Server response
        """
        try:
            import requests
            
            payload = {
                "query": query,
                "agent_name": agent_name,
                "stream": False
            }
            
            response = requests.post(f"{server_url}/query", json=payload, timeout=120)
            response.raise_for_status()
            
            return response.json().get("response", "No response received")
            
        except requests.exceptions.RequestException as e:
            return f"❌ Server request failed: {str(e)}"
        except Exception as e:
            return f"❌ Server query error: {str(e)}"

    def get_agent_instance(self, agent_name: str, streaming: bool = True):
        """Get an agent instance for the specified name.
        
        Args:
            agent_name: Name of the agent to create
            streaming: Whether to enable streaming
            
        Returns:
            Agent instance or None if creation failed
        """
        try:
            from src.core.agent_factory import get_agent_factory
            
            agent_factory = get_agent_factory(streaming=streaming)
            return agent_factory.create_agent(agent_name)
            
        except Exception as e:
            print(f"❌ Failed to create agent {agent_name}: {e}")
            return None

    def execute_resolved_command(self, resolved_command: Dict[str, Any], original_query: str) -> Optional[str]:
        """Execute a resolved command from command analysis.
        
        Args:
            resolved_command: Command information to execute
            original_query: Original query for context
            
        Returns:
            Execution result or None if failed
        """
        try:
            import subprocess
            
            command = resolved_command.get("command")
            if not command:
                return None
                
            # Basic safety check
            dangerous_commands = ["rm", "del", "format", "shutdown"]
            if any(dangerous in command.lower() for dangerous in dangerous_commands):
                return f"❌ Dangerous command blocked: {command}"
            
            # Execute command safely
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"❌ Command failed: {result.stderr.strip()}"
                
        except subprocess.TimeoutExpired:
            return "❌ Command timed out"
        except Exception as e:
            return f"❌ Command execution error: {str(e)}"

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get statistics about direct query execution."""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "average_duration": 0.0
            }

        total = len(self.execution_history)
        successful = len([e for e in self.execution_history if e.get("success", False)])
        durations = [e.get("duration", 0) for e in self.execution_history if e.get("duration")]
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "average_duration": sum(durations) / len(durations) if durations else 0.0,
            "recent_executions": self.execution_history[-5:]  # Last 5 executions
        }
