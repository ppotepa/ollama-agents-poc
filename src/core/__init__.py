"""Core architecture package - modular, clean separation of concerns."""

# Import key components from each layer
from .state import VirtualRepository, RepositoryCache
from .io import (
    parse_github_url, 
    clone_repository, 
    get_repository_info,
    generate_repo_hash
)
from .validation import (
    AgentCapability,
    validate_repository_requirement,
    get_agent_capabilities
)
from .factory import get_agent_instance

__all__ = [
    # State management
    'VirtualRepository',
    'RepositoryCache',
    
    # I/O operations
    'parse_github_url',
    'clone_repository', 
    'get_repository_info',
    'generate_repo_hash',
    
    # Validation
    'AgentCapability',
    'validate_repository_requirement',
    'get_agent_capabilities',
    
    # Factory
    'get_agent_instance'
]
