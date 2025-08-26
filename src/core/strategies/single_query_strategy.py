"""Single query strategy implementation."""

import asyncio
import logging
import os
import sys
from typing import Any, Dict

from .base_strategy import BaseStrategy, ExecutionContext, StrategyResult, StrategyType


class SingleQueryStrategy(BaseStrategy):
    """Strategy for single query execution using intelligent orchestrator."""
    
    def _get_strategy_type(self) -> StrategyType:
        return StrategyType.SINGLE_QUERY
        
    def can_handle(self, context: ExecutionContext) -> bool:
        """Single query strategy can handle most basic queries."""
        # This is the fallback strategy, so it can handle anything
        return True
        
    def execute(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute single query using intelligent orchestrator."""
        self.log_execution_start(context)
        
        try:
            result = self._run_intelligent_investigation(
                context.query,
                streaming=context.metadata.get("streaming", True)
            )
            
            self.log_execution_end(context, True)
            return {
                "success": True,
                "result": result,
                "strategy": self.strategy_type.value
            }
            
        except Exception as e:
            self.log_execution_end(context, False)
            return self.handle_error(context, e)
            
    def _run_intelligent_investigation(self, query: str, streaming: bool = True) -> str:
        """Run a query using the intelligent orchestrator for complex investigations."""
        try:
            from src.core.intelligence_enhancer import apply_intelligence_improvements, is_enhanced_investigation_enabled
            from src.core.intelligent_orchestrator import ExecutionMode, get_orchestrator

            self.logger.info("🧠 Starting intelligent investigation")
            self.logger.info(f"🔍 Query: {query[:100]}...")
            self.logger.info(f"🎬 Streaming: {streaming}")

            # Get the orchestrator instance
            orchestrator = get_orchestrator(enable_streaming=streaming)

            # Apply intelligence improvements if enabled
            if is_enhanced_investigation_enabled():
                self.logger.info("🚀 Using enhanced investigation capabilities")
                apply_intelligence_improvements(orchestrator)

            # Run the investigation asynchronously
            async def run_investigation():
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
                            self.logger.info("✅ Investigation completed successfully")

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

                        elif state == "error":
                            self.logger.error("❌ Investigation failed")
                            return "Investigation failed due to an error."

                        else:
                            # Log progress
                            progress = status.get("progress", {})
                            completed = progress.get("completed_steps", 0)
                            total = progress.get("total_steps", 0)
                            if total > 0:
                                percentage = progress.get("completion_percentage", 0)
                                self.logger.debug(f"🔄 Progress: {completed}/{total} steps ({percentage:.1f}%)")

                    await asyncio.sleep(wait_interval)
                    elapsed_time += wait_interval

                # Timeout reached
                self.logger.warning(f"⏱️  Investigation timed out after {max_wait_time} seconds")
                await orchestrator.pause_session(session_id)
                return "Investigation timed out. Please try a more specific query."

            # Run the async investigation
            if sys.platform == 'win32':
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

            return asyncio.run(run_investigation())

        except Exception as e:
            self.logger.error(f"❌ Error in intelligent investigation: {e}")
            # Fallback to collaborative mode
            self.logger.info("🔄 Falling back to collaborative mode")
            return self._fallback_collaborative_query(query, streaming)
            
    def _fallback_collaborative_query(self, query: str, streaming: bool) -> str:
        """Fallback to collaborative query execution."""
        try:
            # Import collaborative functionality
            from src.core.collaborative_system import run_collaborative_query
            return run_collaborative_query(query, "universal", 5, streaming)
        except Exception as e:
            self.logger.error(f"❌ Fallback collaborative query failed: {e}")
            return f"Unable to process query: {e}"
