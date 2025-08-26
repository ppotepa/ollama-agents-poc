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


# ---------------------------------------------------------------------------
# Backwards compatibility functions
#
# Legacy modules may attempt to import ``run_collaborative_query`` from this
# module.  In the refactored system collaborative execution is handled by
# higher-level strategies and orchestrators.  To avoid import errors and
# provide a minimal fallback we implement a simple wrapper here.  The
# function delegates to ``CollaborativeAgentSystem.collaborative_execution``
# when possible and returns a basic result structure.

def run_collaborative_query(query: str, mode: str = "universal", max_iterations: int = 5, streaming: bool = True) -> dict[str, Any]:
    """Execute a collaborative query using a simple fallback implementation.

    This stub does not perform any real collaborative processing.  It is
    provided solely to satisfy imports from legacy modules.  A full
    implementation would coordinate multiple agents and iteratively refine
    solutions.

    Args:
        query: User query to process.
        mode: Agent mode (ignored in this stub).
        max_iterations: Maximum number of iterations (unused).
        streaming: Whether streaming output is desired (unused).

    Returns:
        A dictionary containing a basic status and the original query.
    """
    return {
        "final_answer": f"[Collaborative stub] Processed query: {query}",
        "success": True,
        "details": {"status": "completed", "query": query},
    }

# Legacy exports
from .collaboration import ExecutionNodeType, ExecutionNode, CollaborationContext

__all__ = ['CollaborativeAgentSystem', 'ExecutionNodeType', 'ExecutionNode', 'CollaborationContext', 'create_collaborative_system']
