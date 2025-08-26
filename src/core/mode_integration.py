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
def run_single_query(query: str, streaming: bool = True, repository_url: str = None, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Run a single query using single query strategy."""
    try:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            
        # Prepare metadata with repository URL if provided
        query_metadata = metadata or {}
        if repository_url:
            query_metadata["repository_url"] = repository_url
            
            # Import repository functions directly
            from .io.clone_operations import get_clone_directory, clone_repository
            
            # Debug output for repository URL
            print(f"🔍 Processing repository: {repository_url}")
            
            # Get expected clone directory and create/clone it directly
            clone_dir = get_clone_directory(repository_url, "./data")
            print(f"📁 Target directory: {clone_dir}")
            
            # Clone or create the repository directly
            if clone_repository(repository_url, clone_dir):
                print(f"✅ Repository successfully set up at: {clone_dir}")
                query_metadata["working_directory"] = str(clone_dir)
            else:
                print(f"❌ Failed to set up repository: {repository_url}")
                # Continue without repository
            
        return asyncio.run(run_query_with_strategy(
            query,
            mode="single_query",
            streaming=streaming,
            metadata=query_metadata
        ))
    except Exception as e:
        return f"Single query failed: {e}"
