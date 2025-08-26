"""Strategy registry for managing and selecting execution strategies."""

import logging
from typing import Dict, List, Optional, Type

from .base_strategy import BaseStrategy, ExecutionContext, StrategyType
from .single_query_strategy import SingleQueryStrategy
from .collaborative_strategy import CollaborativeStrategy
from .investigation_strategy import InvestigationStrategy


class StrategyRegistry:
    """Registry for managing and selecting execution strategies."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._strategies: Dict[StrategyType, Type[BaseStrategy]] = {}
        self._instances: Dict[StrategyType, BaseStrategy] = {}
        
        # Register default strategies
        self._register_default_strategies()
        
    def _register_default_strategies(self):
        """Register default built-in strategies."""
        self.register_strategy(StrategyType.SINGLE_QUERY, SingleQueryStrategy)
        self.register_strategy(StrategyType.COLLABORATIVE, CollaborativeStrategy)
        self.register_strategy(StrategyType.INVESTIGATION, InvestigationStrategy)
        
    def register_strategy(self, strategy_type: StrategyType, strategy_class: Type[BaseStrategy]):
        """Register a strategy class for a given strategy type."""
        self._strategies[strategy_type] = strategy_class
        self.logger.debug(f"Registered strategy: {strategy_type.value} -> {strategy_class.__name__}")
        
    def get_strategy(self, strategy_type: StrategyType) -> Optional[BaseStrategy]:
        """Get a strategy instance by type."""
        if strategy_type not in self._instances:
            if strategy_type in self._strategies:
                self._instances[strategy_type] = self._strategies[strategy_type]()
                self.logger.debug(f"Created strategy instance: {strategy_type.value}")
            else:
                self.logger.warning(f"Strategy not found: {strategy_type.value}")
                return None
                
        return self._instances[strategy_type]
        
    def select_strategy(self, context: ExecutionContext) -> Optional[BaseStrategy]:
        """Select the best strategy for the given context."""
        # Check if strategy is explicitly specified
        if "strategy" in context.metadata:
            requested_strategy = context.metadata["strategy"]
            try:
                strategy_type = StrategyType(requested_strategy)
                strategy = self.get_strategy(strategy_type)
                if strategy and strategy.can_handle(context):
                    self.logger.info(f"Using explicitly requested strategy: {strategy_type.value}")
                    return strategy
                else:
                    self.logger.warning(f"Requested strategy {strategy_type.value} cannot handle context")
            except ValueError:
                self.logger.warning(f"Invalid strategy type requested: {requested_strategy}")
        
        # Find the best strategy based on capability
        candidates = []
        
        for strategy_type in self._strategies:
            strategy = self.get_strategy(strategy_type)
            if strategy and strategy.can_handle(context):
                # Calculate priority score
                priority = self._calculate_strategy_priority(strategy_type, context)
                candidates.append((priority, strategy))
                
        if candidates:
            # Sort by priority (higher is better)
            candidates.sort(key=lambda x: x[0], reverse=True)
            selected_strategy = candidates[0][1]
            self.logger.info(f"Selected strategy: {selected_strategy.strategy_type.value}")
            return selected_strategy
        else:
            # Fallback to single query strategy
            self.logger.warning("No suitable strategy found, falling back to single query")
            return self.get_strategy(StrategyType.SINGLE_QUERY)
            
    def _calculate_strategy_priority(self, strategy_type: StrategyType, context: ExecutionContext) -> int:
        """Calculate priority score for strategy selection."""
        base_priority = {
            StrategyType.INVESTIGATION: 100,  # Highest priority for deep analysis
            StrategyType.COLLABORATIVE: 75,   # High priority for complex queries
            StrategyType.SINGLE_QUERY: 50     # Default fallback
        }
        
        priority = base_priority.get(strategy_type, 0)
        
        # Adjust based on context metadata
        mode = context.metadata.get("mode", "")
        if mode == strategy_type.value:
            priority += 50  # Boost if explicitly requested
            
        # Adjust based on query characteristics
        query_lower = context.query.lower()
        
        if strategy_type == StrategyType.INVESTIGATION:
            investigation_keywords = ["investigate", "analyze", "deep dive", "research", "study"]
            if any(keyword in query_lower for keyword in investigation_keywords):
                priority += 25
                
        elif strategy_type == StrategyType.COLLABORATIVE:
            collaborative_keywords = ["compare", "review", "examine", "explore", "debug"]
            if any(keyword in query_lower for keyword in collaborative_keywords):
                priority += 25
                
        return priority
        
    def list_strategies(self) -> List[StrategyType]:
        """List all registered strategy types."""
        return list(self._strategies.keys())
        
    def get_strategy_info(self, strategy_type: StrategyType) -> Optional[Dict[str, str]]:
        """Get information about a strategy."""
        if strategy_type in self._strategies:
            strategy_class = self._strategies[strategy_type]
            return {
                "type": strategy_type.value,
                "class": strategy_class.__name__,
                "description": strategy_class.__doc__ or "No description available"
            }
        return None


# Global registry instance
_strategy_registry = None


def get_strategy_registry() -> StrategyRegistry:
    """Get the global strategy registry instance."""
    global _strategy_registry
    if _strategy_registry is None:
        _strategy_registry = StrategyRegistry()
    return _strategy_registry
