"""Repository operation interfaces."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class RepositoryInterface(ABC):
    """Interface for repository operations."""
    
    @abstractmethod
    def clone_repository(self, url: str, target_path: Path) -> bool:
        """Clone repository to target path."""
        pass
        
    @abstractmethod
    def get_repository_info(self, path: Path) -> Dict[str, Any]:
        """Get repository information."""
        pass
        
    @abstractmethod
    def list_files(self, path: Path, pattern: str = None) -> List[str]:
        """List files in repository."""
        pass


class VirtualRepositoryInterface(ABC):
    """Interface for virtual repository operations."""
    
    @abstractmethod
    def create_virtual_repository(self, url: str) -> bool:
        """Create virtual repository from URL."""
        pass
        
    @abstractmethod
    def get_virtual_file_content(self, url: str, file_path: str) -> Optional[str]:
        """Get file content from virtual repository."""
        pass
        
    @abstractmethod
    def list_virtual_files(self, url: str, directory: str = "") -> List[str]:
        """List files in virtual repository."""
        pass


class CacheInterface(ABC):
    """Interface for repository caching."""
    
    @abstractmethod
    def cache_repository(self, url: str, data: bytes) -> None:
        """Cache repository data."""
        pass
        
    @abstractmethod
    def get_cached_repository(self, url: str) -> Optional[bytes]:
        """Get cached repository data."""
        pass
        
    @abstractmethod
    def clear_cache(self) -> None:
        """Clear all cached data."""
        pass
        
    @abstractmethod
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information."""
        pass
