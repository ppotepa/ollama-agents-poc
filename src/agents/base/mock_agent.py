"""Mock agent for testing purposes."""

from typing import Any, Dict, List
from src.agents.base.base_agent import AbstractAgent


class MockAgent(AbstractAgent):
    """Simple mock agent for testing collaborative system."""
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        if config is None:
            config = {
                "name": "Mock Agent",
                "capabilities": {
                    "file_operations": True,
                    "command_execution": True,
                    "collaborative": True
                }
            }
        super().__init__(agent_id, config)
    
    def _build_llm(self) -> Any:
        """Return a mock LLM for testing."""
        return MockLLM()
    
    def _build_tools(self) -> List[Any]:
        """Return mock tools for testing."""
        return [MockTool("file_ops"), MockTool("command_exec")]
    
    def run(self, prompt: str, **kwargs) -> str:
        """Mock run method that returns a simple response."""
        return f"Mock response to: {prompt[:50]}..."
    
    def stream(self, prompt: str, **kwargs):
        """Mock stream method that yields response chunks."""
        response = f"Mock streaming response to: {prompt[:50]}..."
        for word in response.split():
            yield word + " "


class MockLLM:
    """Mock LLM for testing."""
    
    def predict(self, text: str) -> str:
        return f"Mock LLM response to: {text[:30]}..."


class MockTool:
    """Mock tool for testing."""
    
    def __init__(self, name: str):
        self.name = name
    
    def run(self, *args, **kwargs) -> str:
        return f"Mock {self.name} executed"
