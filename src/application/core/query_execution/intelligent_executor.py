"""Intelligent investigation query execution."""

import asyncio
import os
import sys
from typing import Any, Dict, Optional


class IntelligentInvestigationExecutor:
    """Handles execution of intelligent investigation queries."""

    def __init__(self):
        """Initialize the intelligent investigation executor."""
        self._orchestrator = None

    def _get_orchestrator(self, streaming: bool = True):
        """Get or create the orchestrator instance."""
        if self._orchestrator is None:
            from src.core.intelligent_orchestrator import get_orchestrator
            self._orchestrator = get_orchestrator(enable_streaming=streaming)
        return self._orchestrator

    def run_investigation(self, query: str, streaming: bool = True) -> str:
        """Run a query using the intelligent orchestrator for complex investigations.
        
        Args:
            query: The query to investigate
            streaming: Whether to enable streaming output
            
        Returns:
            Investigation results as a string
        """
        try:
            from src.core.intelligence_enhancer import apply_intelligence_improvements, is_enhanced_investigation_enabled

            print("🧠 Starting intelligent investigation")
            print(f"🔍 Query: {query[:100]}...")
            print(f"🎬 Streaming: {streaming}")
            print("=" * 60)

            # Get the orchestrator instance
            orchestrator = self._get_orchestrator(streaming)

            # Apply intelligence improvements if enabled
            if is_enhanced_investigation_enabled():
                print("🚀 Using enhanced investigation capabilities")
                apply_intelligence_improvements(orchestrator)

            # Run the investigation asynchronously
            return self._run_async_investigation(orchestrator, query)

        except Exception as e:
            print(f"❌ Error in intelligent investigation: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to collaborative mode
            print("🔄 Falling back to collaborative mode")
            from .collaborative_executor import CollaborativeQueryExecutor
            collaborative_executor = CollaborativeQueryExecutor()
            return collaborative_executor.run_query(query, "universal", 5, streaming)

    def _run_async_investigation(self, orchestrator, query: str) -> str:
        """Run the investigation asynchronously."""
        async def run_investigation():
            from src.core.intelligent_orchestrator import ExecutionMode
            
            session_id = await orchestrator.start_investigation(
                query=query,
                execution_mode=ExecutionMode.ADAPTIVE,
                context={"working_directory": os.getcwd()}
            )

            # Wait for completion and get results
            max_wait_time = 600  # 10 minutes
            wait_interval = 5  # 5 seconds
            elapsed_time = 0

            while elapsed_time < max_wait_time:
                status = orchestrator.get_session_status(session_id)
                if status:
                    state = status["session"]["current_state"]
                    if state == "completed":
                        print("✅ Investigation completed successfully")
                        return self._extract_results(orchestrator, session_id)

                    elif state == "error":
                        print("❌ Investigation failed")
                        return "Investigation failed due to an error."

                    else:
                        # Print progress
                        self._print_progress(status)

                await asyncio.sleep(wait_interval)
                elapsed_time += wait_interval

            # Timeout reached
            print(f"⏱️  Investigation timed out after {max_wait_time} seconds")
            await orchestrator.pause_session(session_id)
            return "Investigation timed out. Please try a more specific query."

        # Run the async investigation
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        return asyncio.run(run_investigation())

    def _extract_results(self, orchestrator, session_id: str) -> str:
        """Extract final results from the orchestrator."""
        # Get final results from context
        context = orchestrator.context_manager.get_context(session_id)
        if context:
            # Compile final result from execution history
            final_results = []
            for record in context.execution_history:
                if record.get("result") and "ERROR" not in record.get("result", ""):
                    final_results.append(record["result"])

            if final_results:
                return "\n\n".join(final_results[-3:])  # Last 3 successful results
            else:
                return "Investigation completed but no results found."
        else:
            return "Investigation completed successfully."

    def _print_progress(self, status: Dict[str, Any]) -> None:
        """Print investigation progress."""
        progress = status.get("progress", {})
        completed = progress.get("completed_steps", 0)
        total = progress.get("total_steps", 0)
        if total > 0:
            percentage = progress.get("completion_percentage", 0)
            print(f"🔄 Progress: {completed}/{total} steps ({percentage:.1f}%)")

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get statistics about intelligent investigation execution."""
        return {
            "orchestrator_available": self._orchestrator is not None,
            "platform": sys.platform,
            "working_directory": os.getcwd()
        }
