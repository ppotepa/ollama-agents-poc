"""Single query mode for command-line usage - Modular executor delegation."""

# Backwards compatibility facade that delegates to specialized executors
from .query_execution import (
    IntelligentInvestigationExecutor,
    CollaborativeQueryExecutor,
    DirectQueryExecutor,
    SimpleAgentExecutor
)


def run_intelligent_investigation(query: str, streaming: bool = True) -> str:
    """Run a query using the intelligent orchestrator for complex investigations."""
    executor = IntelligentInvestigationExecutor()
    return executor.run_investigation(query, tools_enabled=True)


def run_collaborative_query(query: str, agent_name: str, max_iterations: int = 5, streaming: bool = True) -> str:
    """Run a collaborative query with iterative agent interaction."""
    executor = CollaborativeQueryExecutor()
    return executor.run_query(query, agent_name, max_iterations, tools_enabled=True)


def run_single_query(query: str, agent_name: str, connection_mode: str = "hybrid", 
                    repository_url: str = None, interception_mode: str = "smart", 
                    force_streaming: bool = False, collaborative: bool = False, 
                    max_iterations: int = 5, intelligent_investigation: bool = False, 
                    no_tools: bool = False) -> str:
    """Run a single query with various execution modes."""
    executor = DirectQueryExecutor()
    return executor.run_single_query(
        query=query,
        agent_name=agent_name,
        connection_mode=connection_mode,
        repository_url=repository_url,
        interception_mode=interception_mode,
        force_streaming=force_streaming,
        collaborative=collaborative,
        max_iterations=max_iterations,
        intelligent_investigation=intelligent_investigation,
        no_tools=no_tools
    )


def query_via_server(query: str, agent_name: str, server_url: str) -> str:
    """Query via server endpoint."""
    executor = DirectQueryExecutor()
    return executor.query_via_server(query, agent_name, server_url)


def run_single_query_direct_ollama(query: str, agent_name: str, interceptor_data: dict = None, 
                                  streaming: bool = True, no_tools: bool = False) -> str:
    """Run query directly with Ollama without interception."""
    executor = DirectQueryExecutor()
    return executor._run_direct_ollama_query(query, agent_name, no_tools)


def execute_resolved_command(resolved_command, original_query: str):
    """Execute a resolved command from command analysis."""
    executor = DirectQueryExecutor()
    return executor.execute_resolved_command(resolved_command, original_query)


def get_agent_instance(agent_name: str, streaming: bool = True):
    """Get an agent instance for the specified name."""
    executor = DirectQueryExecutor()
    return executor.get_agent_instance(agent_name, streaming)


def execute_with_simple_agent(query: str, agent_name: str = "qwen2.5-coder:7b-instruct", no_tools: bool = False) -> str:
    """Execute query using a simple agent approach."""
    executor = SimpleAgentExecutor()
    return executor.execute_with_simple_agent(query, agent_name, no_tools)


# Legacy compatibility
class SimpleQueryAgent:
    """Simple query agent wrapper for backwards compatibility."""
    
    def __init__(self, agent_name: str, config: dict = None):
        self.agent_name = agent_name
        self.config = config or {}
        self.executor = SimpleAgentExecutor()
    
    def query(self, query: str) -> str:
        """Execute a query."""
        return self.executor.execute_with_simple_agent(query, self.agent_name)
    
    def stream_query(self, query: str):
        """Stream a query (generator)."""
        # For backwards compatibility, return the result as a single yield
        result = self.executor.execute_with_simple_agent(query, self.agent_name)
        yield result
