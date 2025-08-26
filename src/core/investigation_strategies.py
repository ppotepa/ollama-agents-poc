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
]
