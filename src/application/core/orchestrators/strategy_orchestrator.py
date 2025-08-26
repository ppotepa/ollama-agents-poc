"""Orchestrator implementation using strategy pattern and clean interfaces."""

import asyncio
import logging
import uuid
from typing import Any, Dict, Optional

from ..interfaces.orchestrator_interface import OrchestratorInterface, ExecutionSession
from ..strategies import ExecutionContext, get_strategy_registry


class StrategyOrchestrator(OrchestratorInterface):
    """Orchestrator implementation using strategy pattern."""
    
    def __init__(self, enable_streaming: bool = True):
        self.logger = logging.getLogger(__name__)
        self.enable_streaming = enable_streaming
        self.strategy_registry = get_strategy_registry()
        self.active_sessions: Dict[str, ExecutionSession] = {}
        
    async def execute_query(self, query: str, mode: Optional[str] = None, 
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Execute a query using appropriate strategy."""
        try:
            # Create execution context
            context = ExecutionContext(
                query=query,
                metadata=metadata or {}
            )
            
            # Add mode to metadata if specified
            if mode:
                context.metadata["mode"] = mode
                
            # Add streaming preference
            context.metadata["streaming"] = self.enable_streaming
            
            # Select and execute strategy
            strategy = self.strategy_registry.select_strategy(context)
            
            if not strategy:
                raise RuntimeError("No suitable strategy found for query")
                
            result = strategy.execute(context)
            
            if result.get("success", False):
                return result.get("result", "No result available")
            else:
                error_msg = result.get("error", "Unknown error occurred")
                raise RuntimeError(f"Strategy execution failed: {error_msg}")
                
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Interface compliance methods
    #
    # The OrchestratorInterface defines ``set_strategy_registry`` and
    # ``get_execution_context`` as abstract methods.  These no-op
    # implementations satisfy the interface requirements without altering
    # existing logic.

    def set_strategy_registry(self, registry: Any) -> None:
        """Set the strategy registry for this orchestrator.

        Args:
            registry: A strategy registry instance.  The strategy
                orchestrator stores it for later use.  If ``None`` is
                provided the orchestrator keeps the existing registry.
        """
        if registry is not None:
            self.strategy_registry = registry

    def get_execution_context(self, query: str, **kwargs: Any) -> Any:
        """Build a minimal execution context for a query.

        Args:
            query: The user query.
            **kwargs: Additional metadata for the context.

        Returns:
            An ``ExecutionContext`` instance populated with the query and
            metadata.  If the ``ExecutionContext`` type is not available the
            method returns a simple dictionary.
        """
        try:
            from ..strategies.base_strategy import ExecutionContext  # type: ignore
            return ExecutionContext(query=query, metadata=kwargs or {})
        except Exception:
            # Fallback to a plain dict if ExecutionContext cannot be imported
            return {"query": query, "metadata": kwargs or {}}
            
    async def start_session(self, query: str, mode: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start a new execution session."""
        session_id = str(uuid.uuid4())
        
        try:
            # Create session
            session = ExecutionSession(
                session_id=session_id,
                query=query,
                mode=mode,
                metadata=metadata or {},
                status="initializing"
            )
            
            self.active_sessions[session_id] = session
            self.logger.info(f"Started session {session_id} for query: {query[:100]}...")
            
            # Execute query asynchronously
            asyncio.create_task(self._execute_session(session))
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to start session: {e}")
            if session_id in self.active_sessions:
                self.active_sessions[session_id].status = "error"
                self.active_sessions[session_id].error = str(e)
            raise
            
    async def _execute_session(self, session: ExecutionSession):
        """Execute a session asynchronously."""
        try:
            session.status = "executing"
            
            # Execute the query
            result = await self.execute_query(
                session.query, 
                session.mode, 
                session.metadata
            )
            
            # Update session
            session.status = "completed"
            session.result = result
            
            self.logger.info(f"Session {session.session_id} completed successfully")
            
        except Exception as e:
            session.status = "error"
            session.error = str(e)
            self.logger.error(f"Session {session.session_id} failed: {e}")
            
    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a session."""
        if session_id not in self.active_sessions:
            return None
            
        session = self.active_sessions[session_id]
        return {
            "session_id": session.session_id,
            "status": session.status,
            "query": session.query,
            "mode": session.mode,
            "result": session.result,
            "error": session.error,
            "metadata": session.metadata
        }
        
    async def pause_session(self, session_id: str) -> bool:
        """Pause a session."""
        if session_id not in self.active_sessions:
            return False
            
        session = self.active_sessions[session_id]
        if session.status == "executing":
            session.status = "paused"
            self.logger.info(f"Paused session {session_id}")
            return True
            
        return False
        
    async def resume_session(self, session_id: str) -> bool:
        """Resume a paused session."""
        if session_id not in self.active_sessions:
            return False
            
        session = self.active_sessions[session_id]
        if session.status == "paused":
            # Re-execute the session
            asyncio.create_task(self._execute_session(session))
            self.logger.info(f"Resumed session {session_id}")
            return True
            
        return False
        
    async def cancel_session(self, session_id: str) -> bool:
        """Cancel a session."""
        if session_id not in self.active_sessions:
            return False
            
        session = self.active_sessions[session_id]
        session.status = "cancelled"
        self.logger.info(f"Cancelled session {session_id}")
        return True
        
    def list_active_sessions(self) -> list[str]:
        """List all active session IDs."""
        return [
            session_id for session_id, session in self.active_sessions.items()
            if session.status in ["initializing", "executing", "paused"]
        ]
        
    def cleanup_completed_sessions(self) -> int:
        """Clean up completed and error sessions."""
        completed_sessions = [
            session_id for session_id, session in self.active_sessions.items()
            if session.status in ["completed", "error", "cancelled"]
        ]
        
        for session_id in completed_sessions:
            del self.active_sessions[session_id]
            
        self.logger.info(f"Cleaned up {len(completed_sessions)} completed sessions")
        return len(completed_sessions)


# Factory function for creating orchestrator instances
def create_strategy_orchestrator(enable_streaming: bool = True) -> StrategyOrchestrator:
    """Create a strategy-based orchestrator instance."""
    return StrategyOrchestrator(enable_streaming=enable_streaming)


# Default orchestrator instance
_default_orchestrator = None


def get_default_orchestrator() -> StrategyOrchestrator:
    """Get the default orchestrator instance."""
    global _default_orchestrator
    if _default_orchestrator is None:
        _default_orchestrator = create_strategy_orchestrator()
    return _default_orchestrator
