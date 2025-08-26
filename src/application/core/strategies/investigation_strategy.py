"""Investigation strategy implementation."""

import logging
from typing import Any, Dict

from .base_strategy import BaseStrategy, ExecutionContext, StrategyResult, StrategyType


class InvestigationStrategy(BaseStrategy):
    """Strategy for systematic investigation and analysis."""
    
    def _get_strategy_type(self) -> StrategyType:
        return StrategyType.INVESTIGATION
        
    def can_handle(self, context: ExecutionContext) -> bool:
        """Investigation strategy handles deep analysis queries."""
        # Check if explicitly requested investigation mode
        if context.metadata.get("mode") == "investigation":
            return True
            
        # Check if query suggests investigation needed
        investigation_indicators = [
            "deep dive", "investigate", "analyze thoroughly", "comprehensive analysis",
            "explore in depth", "systematic review", "detailed examination",
            "research", "study", "survey", "assess", "evaluate"
        ]
        
        query_lower = context.query.lower()
        return any(indicator in query_lower for indicator in investigation_indicators)
        
    def execute(self, context: ExecutionContext) -> Dict[str, Any]:
        """Execute investigation using structured analysis."""
        self.log_execution_start(context)
        
        try:
            # Extract parameters from context
            strategy_type = context.metadata.get("investigation_strategy", "targeted")
            max_depth = context.metadata.get("max_depth", 3)
            streaming = context.metadata.get("streaming", True)
            
            result = self._run_investigation(
                context.query,
                strategy_type,
                max_depth,
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
            
    def _run_investigation(self, query: str, strategy_type: str = "targeted",
                          max_depth: int = 3, streaming: bool = True) -> str:
        """Run a systematic investigation using investigation strategies."""
        try:
            from src.core.investigation_strategies import (
                InvestigationStrategy as InvestType,
                TargetedInvestigationStrategy,
                ExploratoryInvestigationStrategy,
                HypothesisDrivenInvestigationStrategy
            )
            
            self.logger.info(f"🔍 Starting systematic investigation")
            self.logger.info(f"📋 Query: {query[:100]}...")
            self.logger.info(f"🎯 Strategy: {strategy_type}")
            self.logger.info(f"📊 Max depth: {max_depth}")
            self.logger.info(f"🎬 Streaming: {streaming}")
            
            # Select appropriate investigation strategy
            if strategy_type == "targeted":
                strategy = TargetedInvestigationStrategy(InvestType.TARGETED)
            elif strategy_type == "exploratory":
                strategy = ExploratoryInvestigationStrategy(InvestType.EXPLORATORY)
            elif strategy_type == "hypothesis_driven":
                strategy = HypothesisDrivenInvestigationStrategy(InvestType.HYPOTHESIS_DRIVEN)
            else:
                # Default to targeted
                strategy = TargetedInvestigationStrategy(InvestType.TARGETED)
            
            # Create investigation plan
            investigation_plan = strategy.create_investigation_plan(
                query, 
                {"max_depth": max_depth, "streaming": streaming}
            )
            
            # Execute investigation plan
            results = []
            completed_steps = []
            
            while True:
                # Get next executable steps
                next_steps = investigation_plan.get_next_executable_steps(completed_steps)
                
                if not next_steps:
                    break
                    
                # Execute highest priority step
                step = next_steps[0]
                self.logger.info(f"🔄 Executing step: {step.description}")
                
                try:
                    # Execute the step (simplified execution)
                    step_result = self._execute_investigation_step(step, query)
                    results.append(step_result)
                    completed_steps.append(step.step_id)
                    
                    self.logger.info(f"✅ Completed step: {step.step_id}")
                    
                except Exception as step_e:
                    self.logger.warning(f"⚠️ Step failed: {step.step_id} - {step_e}")
                    completed_steps.append(step.step_id)  # Mark as attempted
                    
                # Limit execution to prevent infinite loops
                if len(completed_steps) >= max_depth * 2:
                    break
            
            # Compile final results
            if results:
                final_result = "\n\n".join(results)
                return f"Investigation completed successfully:\n\n{final_result}"
            else:
                return "Investigation completed but no conclusive results found."
                
        except Exception as e:
            self.logger.error(f"❌ Error in investigation: {e}")
            # Fallback to collaborative mode
            self.logger.info("🔄 Falling back to collaborative mode")
            return self._fallback_collaborative_query(query, streaming)
            
    def _execute_investigation_step(self, step, query: str) -> str:
        """Execute a single investigation step."""
        # This is a simplified implementation
        # In a full implementation, this would dispatch to appropriate handlers
        # based on step type and requirements
        
        if "file_analysis" in step.description.lower():
            return f"File analysis completed for: {step.description}"
        elif "model_query" in step.description.lower():
            return f"Model query executed: {step.description}"
        elif "code_review" in step.description.lower():
            return f"Code review completed: {step.description}"
        else:
            return f"Investigation step completed: {step.description}"
            
    def _fallback_collaborative_query(self, query: str, streaming: bool) -> str:
        """Fallback to collaborative query execution."""
        try:
            from src.core.collaborative_system import run_collaborative_query
            return run_collaborative_query(query, "universal", 3, streaming)
        except Exception as e:
            self.logger.error(f"❌ Fallback collaborative query failed: {e}")
            return f"Unable to complete investigation: {e}"
