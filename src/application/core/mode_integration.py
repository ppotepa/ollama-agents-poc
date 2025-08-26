"""Updated main integration using strategy orchestrator."""

import asyncio
import sys
from typing import Any, Dict, Optional

from .orchestrators import get_default_orchestrator


async def run_query_with_strategy(query: str, mode: Optional[str] = None, 
                                 streaming: bool = True, 
                                 metadata: Optional[Dict[str, Any]] = None) -> str:
    """Run a query using the strategy orchestrator."""
    orchestrator = get_default_orchestrator()
    
    # Prepare metadata
    query_metadata = metadata or {}
    query_metadata["streaming"] = streaming
    
    try:
        result = await orchestrator.execute_query(query, mode, query_metadata)
        return result
    except Exception as e:
        return f"Query execution failed: {e}"


def run_intelligent_investigation(query: str, streaming: bool = True) -> str:
    """Run an intelligent investigation using single query strategy."""
    try:
        # Run the async query
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            
        return asyncio.run(run_query_with_strategy(
            query, 
            mode="investigation", 
            streaming=streaming
        ))
    except Exception as e:
        return f"Investigation failed: {e}"


def run_collaborative_query(query: str, mode: str = "universal", 
                           max_iterations: int = 5, streaming: bool = True) -> str:
    """Run a collaborative query using collaborative strategy."""
    try:
        metadata = {
            "max_iterations": max_iterations,
            "original_mode": mode
        }
        
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            
        return asyncio.run(run_query_with_strategy(
            query,
            mode="collaborative", 
            streaming=streaming,
            metadata=metadata
        ))
    except Exception as e:
        return f"Collaborative query failed: {e}"


# Backward compatibility functions
def run_single_query(query: str, streaming: bool = True) -> str:
    """Run a single query using single query strategy."""
    try:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            
        return asyncio.run(run_query_with_strategy(
            query,
            mode="single_query",
            streaming=streaming
        ))
    except Exception as e:
        return f"Single query failed: {e}"
