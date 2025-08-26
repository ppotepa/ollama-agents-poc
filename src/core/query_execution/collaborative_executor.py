"""Collaborative query execution."""

import time
from typing import Any, Dict, Optional


class CollaborativeQueryExecutor:
    """Handles execution of collaborative queries between agents."""

    def __init__(self):
        """Initialize the collaborative query executor."""
        pass

    def run_query(self, query: str, agent_name: str, max_iterations: int = 5, streaming: bool = True) -> str:
        """Run a collaborative query with iterative back-and-forth between agents.
        
        Args:
            query: The query to process
            agent_name: Name of the primary agent to use
            max_iterations: Maximum number of collaboration iterations
            streaming: Whether to enable streaming output
            
        Returns:
            Query result as a string
        """
        try:
            from src.core.collaborative_system import create_collaborative_system
            from src.core.agent_factory import get_agent_factory

            print(f"🤝 Starting collaborative query execution")
            print(f"🔍 Query: {query[:100]}...")
            print(f"🤖 Agent: {agent_name}")
            print(f"🔄 Max iterations: {max_iterations}")
            print(f"🎬 Streaming: {streaming}")
            print("=" * 60)

            # Get agent factory and create main agent
            agent_factory = get_agent_factory(streaming=streaming)
            main_agent = agent_factory.create_agent(agent_name)

            if main_agent is None:
                return f"❌ Failed to create agent: {agent_name}"

            # Create collaborative system
            collaborative_system = create_collaborative_system(main_agent, max_iterations)

            # Execute collaborative query
            start_time = time.time()
            result = collaborative_system.collaborative_execution(
                query=query,
                working_directory=".",
                max_steps=max_iterations
            )
            execution_time = time.time() - start_time

            # Extract and format results
            return self._format_collaborative_results(result, execution_time)

        except Exception as e:
            print(f"❌ Error in collaborative execution: {e}")
            import traceback
            traceback.print_exc()
            return f"Collaborative execution failed: {str(e)}"

    def _format_collaborative_results(self, result: Dict[str, Any], execution_time: float) -> str:
        """Format the collaborative execution results for display.
        
        Args:
            result: Results from collaborative execution
            execution_time: Total execution time
            
        Returns:
            Formatted result string
        """
        output_parts = []
        
        # Add execution summary
        if result.get("execution_completed"):
            output_parts.append("✅ Collaborative execution completed successfully")
            output_parts.append(f"⏱️  Total execution time: {execution_time:.2f}s")
            output_parts.append(f"🔄 Steps completed: {result.get('total_steps', 0)}")
            
            # Add discovered files info
            discovered_files = result.get("discovered_files", [])
            if discovered_files:
                output_parts.append(f"📁 Files discovered: {len(discovered_files)}")
                if len(discovered_files) <= 5:
                    output_parts.append("   " + ", ".join(discovered_files))
                else:
                    output_parts.append("   " + ", ".join(discovered_files[:5]) + f" (and {len(discovered_files)-5} more)")
            
            # Add command execution info
            executed_commands = result.get("executed_commands", [])
            if executed_commands:
                output_parts.append(f"⚡ Commands executed: {len(executed_commands)}")
            
            # Add intermediate results if available
            intermediate_results = result.get("intermediate_results", {})
            if intermediate_results:
                output_parts.append("\\n📋 Key findings:")
                for key, value in list(intermediate_results.items())[:3]:  # Show top 3
                    if isinstance(value, str) and len(value) > 0:
                        preview = value[:200] + "..." if len(value) > 200 else value
                        output_parts.append(f"   • {key}: {preview}")
        
        else:
            output_parts.append("❌ Collaborative execution did not complete successfully")
        
        return "\\n".join(output_parts)

    def validate_agent_availability(self, agent_name: str) -> bool:
        """Validate that the specified agent is available.
        
        Args:
            agent_name: Name of the agent to validate
            
        Returns:
            True if agent is available, False otherwise
        """
        try:
            from src.core.agent_factory import get_agent_factory
            
            agent_factory = get_agent_factory()
            available_agents = agent_factory.get_available_agents()
            
            return agent_name in available_agents
        
        except Exception:
            return False

    def get_recommended_iterations(self, query: str) -> int:
        """Get recommended number of iterations based on query complexity.
        
        Args:
            query: The query to analyze
            
        Returns:
            Recommended number of iterations
        """
        query_lower = query.lower()
        
        # Complex query indicators
        complex_indicators = [
            "analyze", "investigate", "comprehensive", "detailed", "thorough",
            "explain", "understand", "explore", "research"
        ]
        
        # Simple query indicators  
        simple_indicators = [
            "list", "show", "display", "get", "find", "what is"
        ]
        
        complex_score = sum(1 for indicator in complex_indicators if indicator in query_lower)
        simple_score = sum(1 for indicator in simple_indicators if indicator in query_lower)
        
        # Determine based on indicators and length
        if complex_score >= 2 or len(query.split()) > 15:
            return 7  # High complexity
        elif complex_score >= 1 or len(query.split()) > 10:
            return 5  # Medium complexity
        elif simple_score >= 1 or len(query.split()) <= 5:
            return 3  # Low complexity
        else:
            return 5  # Default

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get statistics about collaborative query execution."""
        return {
            "executor_type": "collaborative",
            "default_max_iterations": 5,
            "supported_modes": ["collaborative", "iterative"]
        }
