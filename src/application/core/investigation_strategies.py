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

# ---------------------------------------------------------------------------
# Backwards compatibility: TaskAnalyzer stub
#
# The improved investigation strategies module historically attempted to
# inherit from a ``TaskAnalyzer`` class defined in this module. However, the
# original codebase never actually provided such a class.  During the
# refactoring to a clean architecture this missing dependency caused
# ``ImportError`` exceptions at runtime.  To maintain backwards
# compatibility for clients importing ``TaskAnalyzer`` from
# ``src.application.core.investigation_strategies`` we provide a simple stub
# implementation here.

class TaskAnalyzer:
    """Basic task analyzer used for backwards compatibility.

    This class provides a trivial ``analyze_task`` method which returns
    minimal metadata about a user query.  It can be used as a base class for
    enhanced analyzers (e.g. :class:`ImprovedTaskAnalyzer`) without causing
    import errors in legacy code.  Feel free to extend the implementation
    with additional heuristics if needed.
    """

    def analyze_task(self, query: str) -> dict[str, object]:
        """Analyze a query and return a basic description.

        Args:
            query: The input query string to analyze.

        Returns:
            A dictionary with rudimentary analysis results.  The default
            implementation always indicates that tools are not required and
            includes the original query for reference.  Subclasses should
            override this method to provide richer analysis.
        """
        return {
            "original_query": query,
            "tools_required": False,
            "reason": "No specific analysis available (TaskAnalyzer stub)"
        }

# Legacy compatibility - create default strategy instances
def get_depth_first_strategy():
    return DepthFirstStrategy()

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
    # Expose TaskAnalyzer for backwards compatibility
    'TaskAnalyzer'
]
