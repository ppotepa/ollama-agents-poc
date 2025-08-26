"""State management package for context and cache services."""

from .repository_cache import (
    RepositoryCache, 
    get_repository_cache, 
    get_virtual_repository, 
    cache_virtual_repository,
    get_cached_repositories,
    clear_repository_cache,
    get_virtual_directory
)
from .virtual_repository import VirtualRepository, VirtualDirectory

__all__ = [
    "RepositoryCache",
    "get_repository_cache", 
    "get_virtual_repository",
    "cache_virtual_repository",
    "get_cached_repositories",
    "clear_repository_cache", 
    "get_virtual_directory",
    "VirtualRepository",
    "VirtualDirectory"
]
