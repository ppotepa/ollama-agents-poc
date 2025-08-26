"""Collaborative strategy implementation."""

import logging
from typing import Any, Dict

from .base_strategy import BaseStrategy, ExecutionContext, StrategyResult, StrategyType


class CollaborativeStrategy(BaseStrategy):
    """Strategy for collaborative query execution with multiple agents."""
    
    def _get_strategy_type(self) -> StrategyType:
        return StrategyType.COLLABORATIVE
        
    def can_handle(self, context: ExecutionContext) -> bool:
        """Collaborative strategy handles complex queries requiring agent coordination."""
        # Check if explicitly requested collaborative mode
        if context.metadata.get("mode") == "collaborative":
            return True
            
        # Check if query suggests collaboration needed
        collaborative_indicators = [
            "analysis", "investigate", "research", "explore", "examine",
            "compare", "review", "understand", "explain", "debug"
        ]
        
        query_lower = context.query.lower()
        return any(indicator in query_lower for indicator in collaborative_indicators)
        
    def execute(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute collaborative query using multiple agents."""
        self.log_execution_start(context)
        
        try:
            # Extract parameters from context
            mode = context.metadata.get("mode", "universal")
            max_iterations = context.metadata.get("max_iterations", 5)
            streaming = context.metadata.get("streaming", True)
            
            result = self._run_collaborative_query(
                context.query,
                mode,
                max_iterations,
                streaming
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
            
    def _run_collaborative_query(self, query: str, mode: str = "universal", 
                                max_iterations: int = 5, streaming: bool = True) -> str:
        """Run a collaborative query with iterative agent interaction."""
        try:
            from src.core.collaborative_system import run_collaborative_query
            
            self.logger.info(f"🤝 Starting collaborative query execution")
            self.logger.info(f"🔍 Query: {query[:100]}...")
            self.logger.info(f"🎯 Mode: {mode}")
            self.logger.info(f"🔄 Max iterations: {max_iterations}")
            self.logger.info(f"🎬 Streaming: {streaming}")
            
            return run_collaborative_query(query, mode, max_iterations, streaming)
            
        except Exception as e:
            self.logger.error(f"❌ Error in collaborative query: {e}")
            # Fallback to basic response
            return f"Unable to execute collaborative query: {e}"
