"""Core orchestration interfaces for strategy execution."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from ..strategies.base_strategy import BaseStrategy, ExecutionContext, StrategyResult


class StrategyRegistry(ABC):
    """Interface for managing and selecting strategies."""
    
    @abstractmethod
    def register_strategy(self, strategy: BaseStrategy) -> None:
        """Register a strategy."""
        pass
        
    @abstractmethod
    def get_strategy(self, strategy_type: str) -> Optional[BaseStrategy]:
        """Get strategy by type."""
        pass
        
    @abstractmethod
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy types."""
        pass
        
    @abstractmethod
    def select_strategy(self, context: ExecutionContext) -> Optional[BaseStrategy]:
        """Select best strategy for given context."""
        pass


class OrchestratorInterface(ABC):
    """Core orchestrator interface."""
    
    @abstractmethod
    def execute_query(self, query: str, **kwargs) -> StrategyResult:
        """
        Execute a query using appropriate strategy.
        
        Args:
            query: The query to execute
            **kwargs: Additional context parameters
            
        Returns:
            Strategy execution result
        """
        pass
        
    @abstractmethod
    def set_strategy_registry(self, registry: StrategyRegistry) -> None:
        """Set the strategy registry."""
        pass
        
    @abstractmethod
    def get_execution_context(self, query: str, **kwargs) -> ExecutionContext:
        """Build execution context for query."""
        pass


class ModelSelectorInterface(ABC):
    """Interface for model selection logic."""
    
    @abstractmethod
    def select_model(self, context: ExecutionContext) -> Optional[str]:
        """
        Select appropriate model for execution context.
        
        Args:
            context: Execution context
            
        Returns:
            Selected model name or None
        """
        pass
        
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        pass


class ContextBuilderInterface(ABC):
    """Interface for building execution contexts."""
    
    @abstractmethod
    def build_context(self, query: str, **kwargs) -> ExecutionContext:
        """
        Build execution context from query and parameters.
        
        Args:
            query: The query to execute
            **kwargs: Additional parameters
            
        Returns:
            Built execution context
        """
        pass
        
    @abstractmethod
    def enhance_context(self, context: ExecutionContext) -> ExecutionContext:
        """
        Enhance context with additional information.
        
        Args:
            context: Original context
            
        Returns:
            Enhanced context
        """
        pass


class ExecutionManagerInterface(ABC):
    """Interface for managing strategy execution."""
    
    @abstractmethod
    def execute_strategy(self, strategy: BaseStrategy, context: ExecutionContext) -> StrategyResult:
        """
        Execute strategy with context.
        
        Args:
            strategy: Strategy to execute
            context: Execution context
            
        Returns:
            Execution result
        """
        pass
        
    @abstractmethod
    def handle_execution_error(self, error: Exception, context: ExecutionContext) -> StrategyResult:
        """
        Handle execution errors.
        
        Args:
            error: The error that occurred
            context: Execution context
            
        Returns:
            Error result
        """
        pass
