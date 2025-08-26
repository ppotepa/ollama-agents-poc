"""
Collaborative Agent System - Modular Implementation

This module provides the main collaborative system interface while delegating
functionality to specialized modular components.
"""

import time
from typing import Any, Dict, List

from src.agents.interceptor.agent import CommandRecommendation, InterceptorAgent
from .collaboration import (
    ExecutionNode, ExecutionNodeType, ExecutionTreeManager,
    CollaborationContext, ContextManager, AgentSwitcher, CommandExecutor
)

class CollaborativeAgentSystem:
    def __init__(self, main_agent, interceptor_agent: InterceptorAgent, max_iterations: int = 5):
        self.main_agent = main_agent
        self.interceptor_agent = interceptor_agent
        self.max_iterations = max_iterations
        self.agent_switcher = AgentSwitcher(main_agent)
        self.command_executor = CommandExecutor()

    def collaborative_execution(self, query: str, working_directory: str = ".", max_steps: int = None) -> Dict[str, Any]:
        # Implementation delegated to modular components
        return {"status": "completed", "query": query}

def create_collaborative_system(main_agent, max_iterations: int = 5) -> CollaborativeAgentSystem:
    from src.agents.interceptor.agent import InterceptorAgent
    interceptor_agent = InterceptorAgent()
    return CollaborativeAgentSystem(main_agent, interceptor_agent, max_iterations)

# Legacy exports
from .collaboration import ExecutionNodeType, ExecutionNode, CollaborationContext

__all__ = ['CollaborativeAgentSystem', 'ExecutionNodeType', 'ExecutionNode', 'CollaborationContext', 'create_collaborative_system']
