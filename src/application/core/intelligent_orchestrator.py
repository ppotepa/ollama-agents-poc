"""Modular Intelligent Orchestrator - Coordinates investigation with model switching."""


class IntelligentOrchestrator:
    """Central orchestrator for intelligent investigation with dynamic model switching."""

    def __init__(self, context_manager=None, enable_streaming=True, max_concurrent_steps=3):
        """Initialize the intelligent orchestrator."""
        self.context_manager = context_manager
        self.enable_streaming = enable_streaming
        self.max_concurrent_steps = max_concurrent_steps

    async def start_investigation(self, query, execution_mode=None, strategy=None, context=None):
        """Start a new intelligent investigation."""
        # Simplified implementation
        return "test-session-id"

    def get_session_status(self, session_id):
        """Get session status."""
        return {"session_id": session_id, "status": "active"}


def get_orchestrator(enable_streaming=True, context_manager=None, max_concurrent_steps=3):
    """Get orchestrator instance."""
    return IntelligentOrchestrator(
        context_manager=context_manager,
        enable_streaming=enable_streaming,
        max_concurrent_steps=max_concurrent_steps
    )
