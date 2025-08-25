"""Integration for improved investigation strategies and model selection."""

import os
import time
from typing import Dict, Any, List, Optional

from src.utils.enhanced_logging import get_logger
from src.core.investigation_strategies import (
    InvestigationStrategyManager, 
    BreadthFirstStrategy,
    DepthFirstStrategy,
    TargetedStrategy,
    AdaptiveStrategy
)

from src.core.improved_investigation_strategies import (
    ImprovedBreadthFirstStrategy,
    ImprovedDepthFirstStrategy,
    model_scoring_cache
)

# Import original orchestrator components
from src.core.intelligent_orchestrator import (
    IntelligentOrchestrator,
    ExecutionStep,
    OrchestrationSession
)

def use_improved_strategies(strategy_manager: InvestigationStrategyManager) -> None:
    """Replace default strategies with improved versions.
    
    Args:
        strategy_manager: The strategy manager to update
    """
    logger = get_logger()
    logger.info("Applying improved investigation strategies")
    
    # Store original context manager
    context_manager = strategy_manager.context_manager
    
    # Replace strategies with improved versions
    strategy_manager.strategies["breadth-first"] = ImprovedBreadthFirstStrategy(context_manager)
    strategy_manager.strategies["depth-first"] = ImprovedDepthFirstStrategy(context_manager)
    
    # Keep other strategies for now
    # In a future update, replace these as well
    
    logger.info("Investigation strategies upgraded successfully")

def patch_orchestrator_model_selection(orchestrator: IntelligentOrchestrator) -> None:
    """Patch the orchestrator to use improved model selection.
    
    Args:
        orchestrator: The orchestrator to patch
    """
    logger = get_logger()
    logger.info("Applying improved model selection to orchestrator")
    
    # Store original methods for potential fallback
    original_execute_step = orchestrator._execute_step
    
    # Define improved wrapper method
    async def improved_execute_step(session: OrchestrationSession, step: ExecutionStep) -> Dict[str, Any]:
        """Improved execution step with better model selection and feedback loop."""
        # Record the start of execution with the assigned model
        start_time = time.time()
        model_name = step.assigned_model
        task_type = step.metadata.get("task_type", "UNKNOWN")
        
        logger.info(f"Executing step {step.id} with model {model_name} for task {task_type}")
        
        try:
            # Use the original method to execute the step
            result = await original_execute_step(session, step)
            
            # Check success status
            success = result.get("success", True)
            
            # Record model performance in cache
            if success:
                model_scoring_cache.record_success(task_type, model_name)
                logger.debug(f"Recorded successful use of {model_name} for {task_type}")
            else:
                model_scoring_cache.record_failure(task_type, model_name)
                logger.debug(f"Recorded failed use of {model_name} for {task_type}")
            
            return result
            
        except Exception as e:
            # Record failure
            model_scoring_cache.record_failure(task_type, model_name)
            logger.error(f"Error executing step with {model_name}: {e}")
            # Re-raise to let original error handling take over
            raise
    
    # Replace the method
    orchestrator._execute_step = improved_execute_step
    
    logger.info("Model selection improvements applied successfully")

def apply_intelligence_improvements(orchestrator: IntelligentOrchestrator) -> None:
    """Apply all intelligence improvements to the orchestrator.
    
    Args:
        orchestrator: The orchestrator to enhance
    """
    # Replace strategies
    use_improved_strategies(orchestrator.strategy_manager)
    
    # Patch model selection
    patch_orchestrator_model_selection(orchestrator)
    
    # Log success
    logger = get_logger()
    logger.info("Intelligence improvements applied successfully")
    
    # Set marker to avoid double application
    orchestrator._intelligence_enhanced = True

# Standalone utility function to check if enhanced features are enabled
def is_enhanced_investigation_enabled() -> bool:
    """Check if enhanced investigation features are enabled."""
    # Default to enabled, but allow override through environment variable
    return os.environ.get('USE_ENHANCED_INVESTIGATION', '1').lower() in ('1', 'true', 'yes')
