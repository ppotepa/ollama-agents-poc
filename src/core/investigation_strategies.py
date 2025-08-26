"""
Investigation Strategies - Legacy Interface

This module provides backwards compatibility for the investigation strategies system.
The core functionality has been refactored into modular components under investigation/
"""

# Import from the new modular system
from .investigation import (
    InvestigationStrategy,
    InvestigationPriority, 
    InvestigationStep,
    InvestigationPlan,
    BaseInvestigationStrategy,
    DepthFirstStrategy
)

# Add missing strategies needed by other components
class TargetedInvestigationStrategy(BaseInvestigationStrategy):
    """Targeted investigation strategy - focuses on specific targets."""
    
    def __init__(self, strategy_type=None):
        self.strategy_type = strategy_type or InvestigationStrategy.TARGETED
        
    def create_investigation_plan(self, query: str, context: dict = None) -> InvestigationPlan:
        """Create a basic targeted investigation plan.
        
        Args:
            query: The query to investigate
            context: Optional context dictionary
            
        Returns:
            A simple investigation plan
        """
        import uuid
        plan_id = str(uuid.uuid4())[:8]
        
        # Create basic step
        step = InvestigationStep(
            step_id=f"step-{plan_id}-1",
            description=f"Analyze query: {query}",
            strategy=self.strategy_type,
            priority=InvestigationPriority.HIGH,
            estimated_duration=5.0,
            required_models=["gpt-4"]
        )
        
        return InvestigationPlan(
            investigation_id=plan_id,
            query=query,
            strategy=self.strategy_type,
            steps=[step],
            total_estimated_duration=5.0,
            success_criteria=["Answer found"]
        )

class ExploratoryInvestigationStrategy(BaseInvestigationStrategy):
    """Exploratory investigation strategy stub."""
    
    def __init__(self, strategy_type=None):
        self.strategy_type = strategy_type or InvestigationStrategy.EXPLORATORY
        
    def create_investigation_plan(self, query: str, context: dict = None) -> InvestigationPlan:
        """Create a basic exploratory investigation plan."""
        import uuid
        plan_id = str(uuid.uuid4())[:8]
        
        # Create basic step
        step = InvestigationStep(
            step_id=f"step-{plan_id}-1",
            description=f"Explore: {query}",
            strategy=self.strategy_type,
            priority=InvestigationPriority.MEDIUM,
            estimated_duration=10.0,
            required_models=["gpt-4"]
        )
        
        return InvestigationPlan(
            investigation_id=plan_id,
            query=query,
            strategy=self.strategy_type,
            steps=[step],
            total_estimated_duration=10.0,
            success_criteria=["Exploration complete"]
        )
        
class HypothesisDrivenInvestigationStrategy(BaseInvestigationStrategy):
    """Hypothesis-driven investigation strategy stub."""
    
    def __init__(self, strategy_type=None):
        self.strategy_type = strategy_type or InvestigationStrategy.HYPOTHESIS_DRIVEN
        
    def create_investigation_plan(self, query: str, context: dict = None) -> InvestigationPlan:
        """Create a basic hypothesis-driven investigation plan."""
        import uuid
        plan_id = str(uuid.uuid4())[:8]
        
        # Create basic step
        step = InvestigationStep(
            step_id=f"step-{plan_id}-1",
            description=f"Test hypothesis for: {query}",
            strategy=self.strategy_type,
            priority=InvestigationPriority.HIGH,
            estimated_duration=15.0,
            required_models=["gpt-4"]
        )
        
        return InvestigationPlan(
            investigation_id=plan_id,
            query=query,
            strategy=self.strategy_type,
            steps=[step],
            total_estimated_duration=15.0,
            success_criteria=["Hypothesis tested"]
        )

# Typing imports for stubs
from typing import Any

# Legacy compatibility - create default strategy instances
def get_depth_first_strategy():
    return DepthFirstStrategy()

# ---------------------------------------------------------------------------
# Backwards compatibility stubs
#
# Historically, this module exposed additional classes such as ``TaskAnalyzer``
# and ``InvestigationStrategyManager`` which were used by advanced
# investigation and intelligence features.  During the refactoring to a
# clean architecture these types were relocated or removed.  To avoid
# ``ImportError`` exceptions from legacy code we provide minimal stub
# implementations here.  These stubs return sensible defaults and can be
# extended in the future to restore full functionality.


class TaskAnalyzer:
    """Basic task analyzer used for backwards compatibility.

    The original implementation provided heuristics to determine whether a
    user query required tool usage and other metadata.  The stub simply
    returns the original query and indicates that no tools are required.
    """

    def analyze_task(self, query: str) -> dict[str, object]:
        """Analyze a query and return a basic description.

        Args:
            query: The input query string to analyze.

        Returns:
            A dictionary with rudimentary analysis results.  The default
            implementation always indicates that tools are not required and
            includes the original query for reference.
        """
        return {
            "original_query": query,
            "tools_required": False,
            "reason": "No specific analysis available (TaskAnalyzer stub)",
        }


class InvestigationStrategyManager:
    """Basic strategy manager stub for backwards compatibility.

    This manager maintains a dictionary of available strategies keyed by
    their names.  In the refactored system only a depth-first strategy is
    provided by default.  The class exposes methods for retrieving a
    registered strategy and for selecting a default when none is specified.
    """

    def __init__(self, context_manager: Any | None = None) -> None:
        # Lazily import the depth-first strategy to avoid circular imports
        from .investigation import DepthFirstStrategy

        self.context_manager = context_manager
        # Initialize with a single depth-first strategy instance
        self.strategies: dict[str, BaseInvestigationStrategy] = {
            "depth-first": DepthFirstStrategy(),
        }

    def get_strategy(self, name: str) -> BaseInvestigationStrategy:
        """Return a strategy by name if available, otherwise fallback to the first."""
        return self.strategies.get(name, next(iter(self.strategies.values())))

    def select_best_strategy(self, query: str) -> BaseInvestigationStrategy:
        """Select the best strategy for a given query.

        The stub always returns the depth-first strategy; in a full
        implementation this would analyze the query and context.
        """
        return next(iter(self.strategies.values()))

# Legacy exports
__all__ = [
    'InvestigationStrategy',
    'InvestigationPriority',
    'InvestigationStep', 
    'InvestigationPlan',
    'BaseInvestigationStrategy',
    'DepthFirstStrategy',
    'get_depth_first_strategy',
    # New investigation strategies
    'TargetedInvestigationStrategy',
    'ExploratoryInvestigationStrategy',
    'HypothesisDrivenInvestigationStrategy',
    # Backwards compatibility stubs
    'TaskAnalyzer',
    'InvestigationStrategyManager'
]
