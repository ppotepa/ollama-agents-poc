"""Session management for orchestration processes."""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.core.execution_planner import ExecutionPlan
from src.core.investigation_strategies import InvestigationPlan
from src.core.task_decomposer import TaskDecomposition


class OrchestrationState(Enum):
    """States of the orchestration process."""
    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    ADAPTING = "adapting"
    COMPLETED = "completed"
    ERROR = "error"
    PAUSED = "paused"


class ExecutionMode(Enum):
    """Execution modes for the orchestrator."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    SMART_ROUTING = "smart_routing"


@dataclass
class OrchestrationSession:
    """Represents an active orchestration session."""
    session_id: str
    original_query: str
    current_state: OrchestrationState
    execution_mode: ExecutionMode
    investigation_plan: Optional[InvestigationPlan]
    execution_plan: Optional[ExecutionPlan]
    task_decomposition: Optional[TaskDecomposition] = None
    current_step: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    model_switches: int = 0
    total_execution_time: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "original_query": self.original_query,
            "current_state": self.current_state.value,
            "execution_mode": self.execution_mode.value,
            "current_step": self.current_step,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "model_switches": self.model_switches,
            "total_execution_time": self.total_execution_time,
            "created_at": self.created_at,
            "last_activity": self.last_activity
        }


class SessionManager:
    """Manages orchestration sessions and their lifecycle."""

    def __init__(self, max_session_duration: int = 3600):
        """Initialize session manager.
        
        Args:
            max_session_duration: Maximum session duration in seconds
        """
        self.active_sessions: Dict[str, OrchestrationSession] = {}
        self.max_session_duration = max_session_duration

    def create_session(self, 
                      query: str, 
                      execution_mode: ExecutionMode = ExecutionMode.ADAPTIVE) -> OrchestrationSession:
        """Create a new orchestration session.
        
        Args:
            query: User query to investigate
            execution_mode: How to execute the investigation
            
        Returns:
            New orchestration session
        """
        session_id = str(uuid.uuid4())
        
        session = OrchestrationSession(
            session_id=session_id,
            original_query=query,
            current_state=OrchestrationState.INITIALIZING,
            execution_mode=execution_mode,
            investigation_plan=None,
            execution_plan=None,
            current_step=None
        )
        
        self.active_sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[OrchestrationSession]:
        """Get session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session if found, None otherwise
        """
        return self.active_sessions.get(session_id)

    def update_session_state(self, session_id: str, state: OrchestrationState) -> bool:
        """Update session state.
        
        Args:
            session_id: Session ID
            state: New state
            
        Returns:
            True if updated, False if session not found
        """
        session = self.active_sessions.get(session_id)
        if session:
            session.current_state = state
            session.update_activity()
            return True
        return False

    def pause_session(self, session_id: str) -> bool:
        """Pause a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if paused, False if session not found
        """
        return self.update_session_state(session_id, OrchestrationState.PAUSED)

    def resume_session(self, session_id: str) -> bool:
        """Resume a paused session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if resumed, False if session not found or not paused
        """
        session = self.active_sessions.get(session_id)
        if session and session.current_state == OrchestrationState.PAUSED:
            session.current_state = OrchestrationState.EXECUTING
            session.update_activity()
            return True
        return False

    def complete_session(self, session_id: str) -> bool:
        """Mark session as completed.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if completed, False if session not found
        """
        return self.update_session_state(session_id, OrchestrationState.COMPLETED)

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session status.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session status dictionary or None if not found
        """
        session = self.active_sessions.get(session_id)
        return session.to_dict() if session else None

    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions.
        
        Returns:
            List of session status dictionaries
        """
        return [session.to_dict() for session in self.active_sessions.values()]

    def cleanup_completed_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up old completed sessions.
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
            
        Returns:
            Number of sessions cleaned up
        """
        current_time = time.time()
        cleanup_threshold = current_time - (max_age_hours * 3600)
        
        sessions_to_remove = []
        for session_id, session in self.active_sessions.items():
            if (session.current_state in [OrchestrationState.COMPLETED, OrchestrationState.ERROR] and
                session.last_activity < cleanup_threshold):
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
        
        return len(sessions_to_remove)

    def cleanup_expired_sessions(self) -> int:
        """Clean up sessions that have exceeded max duration.
        
        Returns:
            Number of sessions cleaned up
        """
        current_time = time.time()
        
        sessions_to_remove = []
        for session_id, session in self.active_sessions.items():
            if current_time - session.created_at > self.max_session_duration:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
        
        return len(sessions_to_remove)
