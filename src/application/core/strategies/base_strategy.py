"""Base strategy interface for execution strategies."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from enum import Enum
import logging


class StrategyType(Enum):
    """Types of execution strategies."""
    SINGLE_QUERY = "single_query"
    COLLABORATIVE = "collaborative" 
    INVESTIGATION = "investigation"
    REFLECTION = "reflection"
    ENHANCED_INVESTIGATION = "enhanced_investigation"


class ExecutionContext:
    """Context object passed to strategies."""
    
    def __init__(self, query: str, session_id: str = None, **kwargs):
        self.query = query
        self.session_id = session_id
        self.metadata: Dict[str, Any] = kwargs
        self.intermediate_results: Dict[str, Any] = {}
        self.discovered_files: List[str] = []
        self.execution_history: List[Dict[str, Any]] = []
        
    def add_result(self, key: str, result: Any):
        """Add intermediate result."""
        self.intermediate_results[key] = result
        
    def get_result(self, key: str, default: Any = None) -> Any:
        """Get intermediate result."""
        return self.intermediate_results.get(key, default)
        
    def add_execution_step(self, step: Dict[str, Any]):
        """Add execution step to history."""
        self.execution_history.append(step)


class BaseStrategy(ABC):
    """Abstract base class for all execution strategies."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.strategy_type = self._get_strategy_type()
        
    @abstractmethod
    def _get_strategy_type(self) -> StrategyType:
        """Return the type of this strategy."""
        pass
        
    @abstractmethod
    def execute(self, context: ExecutionContext) -> Dict[str, Any]:
        """
        Execute the strategy with given context.
        
        Args:
            context: Execution context containing query and metadata
            
        Returns:
            Dictionary containing execution results
        """
        pass
        
    @abstractmethod
    def can_handle(self, context: ExecutionContext) -> bool:
        """
        Check if this strategy can handle the given context.
        
        Args:
            context: Execution context to evaluate
            
        Returns:
            True if strategy can handle the context
        """
        pass
        
    def prepare_context(self, context: ExecutionContext) -> ExecutionContext:
        """
        Prepare context before execution (hook for subclasses).
        
        Args:
            context: Original execution context
            
        Returns:
            Modified execution context
        """
        return context
        
    def cleanup_context(self, context: ExecutionContext) -> None:
        """
        Cleanup after execution (hook for subclasses).
        
        Args:
            context: Execution context to cleanup
        """
        pass
        
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about this strategy."""
        return {
            "name": self.__class__.__name__,
            "type": self.strategy_type.value,
            "description": self.__doc__ or "No description available"
        }
        
    def log_execution_start(self, context: ExecutionContext):
        """Log strategy execution start."""
        self.logger.info(f"Starting {self.strategy_type.value} strategy execution")
        self.logger.debug(f"Query: {context.query}")
        
    def log_execution_end(self, context: ExecutionContext, success: bool):
        """Log strategy execution end."""
        status = "completed" if success else "failed"
        self.logger.info(f"{self.strategy_type.value} strategy execution {status}")
        
    def handle_error(self, context: ExecutionContext, error: Exception) -> Dict[str, Any]:
        """
        Handle execution errors (can be overridden by subclasses).
        
        Args:
            context: Execution context
            error: The error that occurred
            
        Returns:
            Error result dictionary
        """
        self.logger.error(f"Strategy execution failed: {error}")
        return {
            "success": False,
            "error": str(error),
            "strategy": self.strategy_type.value
        }


class StrategyResult:
    """Container for strategy execution results."""
    
    def __init__(self, success: bool, data: Dict[str, Any] = None, error: str = None):
        self.success = success
        self.data = data or {}
        self.error = error
        self.strategy_type: Optional[StrategyType] = None
        
    def add_data(self, key: str, value: Any):
        """Add data to result."""
        self.data[key] = value
        
    def get_data(self, key: str, default: Any = None) -> Any:
        """Get data from result."""
        return self.data.get(key, default)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "strategy_type": self.strategy_type.value if self.strategy_type else None
        }
