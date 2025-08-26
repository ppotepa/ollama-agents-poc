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
    'get_depth_first_strategy'
    ,
    # Backwards compatibility stubs
    'TaskAnalyzer',
    'InvestigationStrategyManager'
]
