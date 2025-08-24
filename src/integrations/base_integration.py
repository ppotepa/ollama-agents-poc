#!/usr/bin/env python3
"""
Base Integration Interface

Defines the contract for all external service integrations.
Follows interface segregation principle by defining minimal required methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseIntegration(ABC):
    """
    Abstract base class for external service integrations.
    
    Defines the minimum interface that all integrations must implement
    to ensure consistency and interoperability.
    """
    
    @abstractmethod
    def get_models(self) -> List[Dict[str, Any]]:
        """
        Fetch available models from the external service.
        
        Returns:
            List of model dictionaries in standardized format
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the external service is available and reachable.
        
        Returns:
            True if service is available, False otherwise
        """
        pass
    
    def get_version(self) -> Optional[str]:
        """
        Get version information from the external service.
        
        Returns:
            Version string if available, None otherwise
        """
        return None
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the external service.
        
        Returns:
            Dictionary with health status information
        """
        available = self.is_available()
        version = self.get_version()
        
        return {
            "available": available,
            "version": version,
            "status": "healthy" if available else "unavailable"
        }
