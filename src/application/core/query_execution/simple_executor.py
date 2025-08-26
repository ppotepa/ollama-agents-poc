"""Simple agent execution for basic queries."""

import os
import time
from typing import Any, Dict, Optional


class SimpleAgentExecutor:
    """Handles simple agent execution without complex orchestration."""

    def __init__(self):
        """Initialize the simple agent executor."""
        self.execution_count = 0
        self.last_execution_time = None

    def execute_with_simple_agent(self, query: str, agent_name: str = "qwen2.5-coder:7b-instruct", 
                                 no_tools: bool = False) -> str:
        """Execute query using a simple agent approach.
        
        Args:
            query: The query to execute
            agent_name: Name of the agent to use
            no_tools: Whether to disable tool usage
            
        Returns:
            Agent response
        """
        self.execution_count += 1
        self.last_execution_time = time.time()
        
        try:
            print(f"🎯 Simple Agent Execution #{self.execution_count}")
            print(f"🔍 Query: {query[:100]}...")
            print(f"🤖 Agent: {agent_name}")
            print("=" * 50)

            # Get agent instance
            agent = self._get_agent_instance(agent_name)
            if agent is None:
                return f"❌ Failed to create agent: {agent_name}"

            # Execute query
            if hasattr(agent, 'stream_query') and not no_tools:
                return self._execute_streaming(agent, query)
            else:
                return self._execute_direct(agent, query)

        except Exception as e:
            print(f"❌ Simple agent execution error: {e}")
            import traceback
            traceback.print_exc()
            return f"Simple agent execution failed: {str(e)}"

    def _get_agent_instance(self, agent_name: str):
        """Get an agent instance for execution."""
        try:
            from src.core.agent_factory import get_agent_factory
            
            agent_factory = get_agent_factory(streaming=True)
            return agent_factory.create_agent(agent_name)
            
        except Exception as e:
            print(f"❌ Failed to create agent {agent_name}: {e}")
            return None

    def _execute_streaming(self, agent, query: str) -> str:
        """Execute query with streaming output."""
        try:
            result_parts = []
            
            def collect_token_safely(token):
                if token:
                    result_parts.append(str(token))
                    print(str(token), end='', flush=True)

            print("📡 Streaming response:")
            for token in agent.stream_query(query):
                collect_token_safely(token)
            
            print()  # New line after streaming
            return ''.join(result_parts)
            
        except Exception as e:
            print(f"❌ Streaming execution error: {e}")
            return f"Streaming failed: {str(e)}"

    def _execute_direct(self, agent, query: str) -> str:
        """Execute query with direct output."""
        try:
            print("💬 Direct response:")
            result = agent.query(query)
            print(result)
            return result
            
        except Exception as e:
            print(f"❌ Direct execution error: {e}")
            return f"Direct execution failed: {str(e)}"

    def run_agent_test(self, agent_name: str = "qwen2.5-coder:7b-instruct") -> bool:
        """Run a simple test to verify agent functionality.
        
        Args:
            agent_name: Name of the agent to test
            
        Returns:
            True if test passed, False otherwise
        """
        try:
            test_query = "What is 2+2? Please respond with just the number."
            result = self.execute_with_simple_agent(test_query, agent_name, no_tools=True)
            
            # Simple validation - should contain "4"
            success = "4" in result
            
            if success:
                print(f"✅ Agent test passed for {agent_name}")
            else:
                print(f"❌ Agent test failed for {agent_name}")
                print(f"Expected '4' in response, got: {result}")
                
            return success
            
        except Exception as e:
            print(f"❌ Agent test error: {e}")
            return False

    def quick_query(self, query: str, agent_name: str = "qwen2.5-coder:7b-instruct") -> str:
        """Execute a quick query with minimal overhead.
        
        Args:
            query: The query to execute
            agent_name: Name of the agent to use
            
        Returns:
            Quick response
        """
        try:
            agent = self._get_agent_instance(agent_name)
            if agent is None:
                return f"❌ Agent not available: {agent_name}"

            # Direct execution without streaming or complex processing
            return agent.query(query)
            
        except Exception as e:
            return f"❌ Quick query failed: {str(e)}"

    def batch_execute(self, queries: list, agent_name: str = "qwen2.5-coder:7b-instruct") -> Dict[str, Any]:
        """Execute multiple queries in batch.
        
        Args:
            queries: List of queries to execute
            agent_name: Name of the agent to use
            
        Returns:
            Batch execution results
        """
        results = {
            "total_queries": len(queries),
            "completed": 0,
            "failed": 0,
            "results": [],
            "execution_time": 0
        }
        
        start_time = time.time()
        
        try:
            agent = self._get_agent_instance(agent_name)
            if agent is None:
                results["error"] = f"Failed to create agent: {agent_name}"
                return results

            for i, query in enumerate(queries):
                try:
                    print(f"🔄 Executing query {i+1}/{len(queries)}")
                    result = agent.query(query)
                    
                    results["results"].append({
                        "query": query,
                        "result": result,
                        "success": True,
                        "index": i
                    })
                    results["completed"] += 1
                    
                except Exception as e:
                    results["results"].append({
                        "query": query,
                        "error": str(e),
                        "success": False,
                        "index": i
                    })
                    results["failed"] += 1

        except Exception as e:
            results["error"] = str(e)
        
        results["execution_time"] = time.time() - start_time
        return results

    def execute_with_fallback(self, query: str, preferred_agent: str, 
                            fallback_agents: list = None) -> str:
        """Execute query with fallback to other agents if primary fails.
        
        Args:
            query: The query to execute
            preferred_agent: Preferred agent name
            fallback_agents: List of fallback agent names
            
        Returns:
            Execution result
        """
        if fallback_agents is None:
            fallback_agents = ["llama3.2:3b", "qwen2.5:7b", "phi3:mini"]

        # Try primary agent first
        try:
            result = self.execute_with_simple_agent(query, preferred_agent, no_tools=True)
            if not result.startswith("❌"):
                return result
        except Exception as e:
            print(f"❌ Primary agent {preferred_agent} failed: {e}")

        # Try fallback agents
        for fallback_agent in fallback_agents:
            try:
                print(f"🔄 Trying fallback agent: {fallback_agent}")
                result = self.execute_with_simple_agent(query, fallback_agent, no_tools=True)
                if not result.startswith("❌"):
                    return f"[Fallback: {fallback_agent}] {result}"
            except Exception as e:
                print(f"❌ Fallback agent {fallback_agent} failed: {e}")
                continue

        return "❌ All agents failed to execute query"

    def get_execution_info(self) -> Dict[str, Any]:
        """Get information about simple agent execution state."""
        return {
            "execution_count": self.execution_count,
            "last_execution_time": self.last_execution_time,
            "last_execution_ago": time.time() - self.last_execution_time if self.last_execution_time else None
        }

    def warm_up_agent(self, agent_name: str) -> bool:
        """Warm up an agent with a simple query to ensure it's ready.
        
        Args:
            agent_name: Name of the agent to warm up
            
        Returns:
            True if warm-up successful, False otherwise
        """
        try:
            print(f"🔥 Warming up agent: {agent_name}")
            warm_up_query = "Hello"
            result = self.quick_query(warm_up_query, agent_name)
            
            success = len(result) > 0 and not result.startswith("❌")
            
            if success:
                print(f"✅ Agent {agent_name} warmed up successfully")
            else:
                print(f"❌ Agent {agent_name} warm-up failed")
                
            return success
            
        except Exception as e:
            print(f"❌ Agent warm-up error: {e}")
            return False
