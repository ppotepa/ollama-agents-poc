"""Main reflection system orchestrating self-evaluation and model switching."""

import time
from typing import Any, Dict, List, Optional

from src.core.context_manager import ContextManager, get_context_manager
from src.utils.enhanced_logging import get_logger

from .types import ConfidenceLevel, ReflectionTrigger
from .checkpoint import ReflectionCheckpoint
from .result import ReflectionResult


class ReflectionSystem:
    """Manages reflection checkpoints and self-evaluation."""

    def __init__(self, context_manager: ContextManager = None):
        """Initialize the reflection system.

        Args:
            context_manager: Context manager instance (optional)
        """
        self.context_manager = context_manager or get_context_manager()
        self.logger = get_logger()
        self.checkpoints: Dict[str, ReflectionCheckpoint] = {}
        self.reflection_history: List[ReflectionResult] = []

        # Configuration
        self.config = {
            "auto_reflection_interval": 300,  # 5 minutes
            "confidence_threshold": ConfidenceLevel.LOW,
            "max_checkpoint_history": 100,
            "enable_auto_swap": True
        }

    def create_checkpoint(self, session_id: str, step_id: str,
                         trigger: ReflectionTrigger) -> str:
        """Create a new reflection checkpoint.

        Args:
            session_id: Session identifier
            step_id: Step identifier
            trigger: What triggered this checkpoint

        Returns:
            Checkpoint ID
        """
        checkpoint_id = f"{session_id}_{step_id}_{int(time.time())}"
        checkpoint = ReflectionCheckpoint(checkpoint_id, trigger)
        self.checkpoints[checkpoint_id] = checkpoint

        self.logger.debug(f"Created reflection checkpoint {checkpoint_id} "
                         f"for {trigger.value}")

        return checkpoint_id

    def evaluate_step(self, session_id: str, step_id: str, current_model: str,
                     step_result: Any, execution_time: float,
                     trigger: ReflectionTrigger = ReflectionTrigger.STEP_COMPLETION) -> ReflectionResult:
        """Evaluate a completed step and generate reflection result.

        Args:
            session_id: Session identifier
            step_id: Step identifier
            current_model: Current model being used
            step_result: Result of the step execution
            execution_time: Time taken to execute
            trigger: What triggered this evaluation

        Returns:
            ReflectionResult with evaluation and recommendations
        """
        checkpoint_id = self.create_checkpoint(session_id, step_id, trigger)
        checkpoint = self.checkpoints[checkpoint_id]

        result = checkpoint.evaluate_step_performance(
            session_id, step_id, current_model, step_result, execution_time
        )

        # Store in history
        self.reflection_history.append(result)

        # Maintain history size
        if len(self.reflection_history) > self.config["max_checkpoint_history"]:
            self.reflection_history = self.reflection_history[-self.config["max_checkpoint_history"]:]

        # Store in context
        self.context_manager.add_context_data(
            session_id,
            f"reflection_{checkpoint_id}",
            result.to_dict(),
            f"reflection_system_{checkpoint_id}"
        )

        self.logger.info(f"Reflection evaluation complete for step {step_id}: "
                        f"Confidence={result.confidence_level.value}, "
                        f"Swap recommended={result.should_swap}")

        return result

    def should_trigger_automatic_reflection(self, session_id: str) -> bool:
        """Check if automatic reflection should be triggered.

        Args:
            session_id: Session identifier

        Returns:
            True if automatic reflection should be triggered
        """
        context = self.context_manager.get_context(session_id)
        if not context:
            return False

        # Check time since last reflection
        last_reflection_time = 0
        for entry in context.entries.values():
            if entry.key.startswith("reflection_"):
                last_reflection_time = max(last_reflection_time, entry.timestamp)

        time_since_last = time.time() - last_reflection_time

        # Trigger if enough time has passed
        if time_since_last > self.config["auto_reflection_interval"]:
            return True

        # Trigger if recent errors
        recent_errors = sum(1 for record in context.execution_history[-3:]
                          if 'error' in record.get('result', '').lower())
        return recent_errors >= 2

    def get_session_reflections(self, session_id: str) -> List[ReflectionResult]:
        """Get all reflections for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of reflection results for the session
        """
        return [r for r in self.reflection_history if r.session_id == session_id]

    def get_latest_reflection(self, session_id: str) -> Optional[ReflectionResult]:
        """Get the most recent reflection for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Latest reflection result or None
        """
        session_reflections = self.get_session_reflections(session_id)
        return max(session_reflections, key=lambda r: r.timestamp) if session_reflections else None

    def clear_session_history(self, session_id: str):
        """Clear reflection history for a specific session.
        
        Args:
            session_id: Session identifier
        """
        self.reflection_history = [r for r in self.reflection_history 
                                 if r.session_id != session_id]
        
        # Clear checkpoints for the session
        session_checkpoints = [cid for cid in self.checkpoints.keys() 
                             if cid.startswith(session_id)]
        for checkpoint_id in session_checkpoints:
            del self.checkpoints[checkpoint_id]

        self.logger.info(f"Cleared reflection history for session {session_id}")

    def get_reflection_statistics(self) -> Dict[str, Any]:
        """Get overall reflection system statistics.
        
        Returns:
            Dictionary with system statistics
        """
        if not self.reflection_history:
            return {
                "total_reflections": 0,
                "sessions_analyzed": 0,
                "average_confidence": 0.0,
                "swap_recommendation_rate": 0.0
            }

        # Calculate statistics
        total_reflections = len(self.reflection_history)
        unique_sessions = len(set(r.session_id for r in self.reflection_history))
        
        # Confidence distribution
        confidence_counts = {}
        swap_count = 0
        
        for reflection in self.reflection_history:
            level = reflection.confidence_level.value
            confidence_counts[level] = confidence_counts.get(level, 0) + 1
            if reflection.should_swap:
                swap_count += 1

        # Calculate average confidence (weighted)
        confidence_weights = {
            ConfidenceLevel.VERY_LOW.value: 0.1,
            ConfidenceLevel.LOW.value: 0.3,
            ConfidenceLevel.MEDIUM.value: 0.5,
            ConfidenceLevel.HIGH.value: 0.7,
            ConfidenceLevel.VERY_HIGH.value: 0.9
        }
        
        weighted_confidence = sum(
            count * confidence_weights.get(level, 0.5)
            for level, count in confidence_counts.items()
        )
        average_confidence = weighted_confidence / total_reflections

        return {
            "total_reflections": total_reflections,
            "sessions_analyzed": unique_sessions,
            "average_confidence": average_confidence,
            "swap_recommendation_rate": swap_count / total_reflections,
            "confidence_distribution": confidence_counts,
            "active_checkpoints": len(self.checkpoints)
        }

    def update_config(self, **kwargs):
        """Update reflection system configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                self.logger.info(f"Updated reflection config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown config parameter: {key}")

    def export_reflections(self, session_id: str = None) -> List[Dict[str, Any]]:
        """Export reflection data as dictionaries.
        
        Args:
            session_id: Optional session ID to filter by
            
        Returns:
            List of reflection dictionaries
        """
        reflections = self.reflection_history
        if session_id:
            reflections = [r for r in reflections if r.session_id == session_id]
            
        return [reflection.to_dict() for reflection in reflections]
