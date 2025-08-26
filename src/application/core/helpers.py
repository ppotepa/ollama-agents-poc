"""Legacy helpers module - provides backwards compatibility through imports from modular packages.

This module maintains the original interface while delegating to the new modular architecture.
All actual implementation has been moved to:
- src/core/state/ - State management and caching
- src/core/io/ - I/O operations and external integrations  
- src/core/validation/ - Agent validation and requirements
- src/core/factory/ - Agent creation and management
"""

# Re-export from state management layer
from .state import (
    VirtualRepository,
    VirtualDirectory, 
    get_virtual_repository,
    cache_virtual_repository,
    get_virtual_directory,
    get_cached_repositories,
    clear_repository_cache,
    RepositoryCache
)

# Re-export from I/O layer  
from .io import (
    parse_github_url,
    download_github_zip,
    download_github_zip_to_memory,
    generate_repo_hash,
    check_agent_supports_coding,
    get_clone_directory,
    clone_repository,
    is_valid_git_url,
    is_git_repository,
    check_remote_repository_exists,
    extract_repo_name_from_url,
    sanitize_project_name,
    create_new_repository,
    get_repository_info,
    list_existing_repositories,
    display_repository_selection,
    is_git_url
)

# Re-export from validation layer
from .validation import (
    AgentCapability,
    is_coding_agent,
    get_agent_capabilities,
    requires_repository,
    validate_repository_requirement
)

# Re-export from factory layer
from .factory import get_agent_instance


# Additional helper functions that need to stay in helpers for now
def _ensure_virtual_repository(git_url: str) -> bool:
    """Ensure virtual repository is available for lightweight mode."""
    try:
        virtual_repo = get_virtual_repository(git_url)
        if virtual_repo is None:
            # Try to load from GitHub directly into memory
            if download_github_zip_to_memory(git_url, None):
                return True
        return virtual_repo is not None
    except Exception as e:
        print(f"⚠️  Failed to ensure virtual repository: {e}")
        return False


def _extract_virtual_repo_to_disk(virtual_repo: VirtualRepository, target_dir):
    """Extract virtual repository to disk - delegates to VirtualRepository."""
    return virtual_repo.extract_to_disk(target_dir)


# Backwards compatibility - maintain all original function names and signatures
__all__ = [
    # State management
    'VirtualRepository',
    'VirtualDirectory',
    'get_virtual_repository', 
    'cache_virtual_repository',
    'get_virtual_directory',
    'get_cached_repositories',
    'clear_repository_cache',
    'RepositoryCache',
    
    # I/O operations
    'parse_github_url',
    'download_github_zip',
    'download_github_zip_to_memory', 
    'generate_repo_hash',
    'check_agent_supports_coding',
    'get_clone_directory',
    'clone_repository',
    'is_valid_git_url',
    'is_git_repository',
    'check_remote_repository_exists',
    'extract_repo_name_from_url',
    'sanitize_project_name', 
    'create_new_repository',
    'get_repository_info',
    'list_existing_repositories',
    'display_repository_selection',
    'is_git_url',
    
    # Validation
    'AgentCapability',
    'is_coding_agent',
    'get_agent_capabilities',
    'requires_repository', 
    'validate_repository_requirement',
    
    # Factory
    'get_agent_instance',
    
    # Legacy helpers
    '_ensure_virtual_repository',
    '_extract_virtual_repo_to_disk'
]
