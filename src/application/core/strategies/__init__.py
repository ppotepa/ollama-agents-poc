"""Strategy pattern implementation for execution strategies."""

from .base_strategy import (
    BaseStrategy,
    ExecutionContext,
    StrategyResult,
    StrategyType,
)
from .single_query_strategy import SingleQueryStrategy
from .collaborative_strategy import CollaborativeStrategy
from .investigation_strategy import InvestigationStrategy
from .strategy_registry import StrategyRegistry, get_strategy_registry

__all__ = [
    "BaseStrategy",
    "ExecutionContext", 
    "StrategyResult",
    "StrategyType",
    "SingleQueryStrategy",
    "CollaborativeStrategy",
    "InvestigationStrategy", 
    "StrategyRegistry",
    "get_strategy_registry"
]
