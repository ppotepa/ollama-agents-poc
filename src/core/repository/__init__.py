"""Repository package for virtual repositories, caching, and Git operations."""

from .cache_manager import (
    cache_virtual_repository,
    clear_repository_cache,
    ensure_virtual_repository,
    get_cached_repositories,
    get_virtual_directory,
    get_virtual_repository,
)
from .git_operations import (
    clone_repository,
    create_new_repository,
    download_github_zip_to_memory,
    extract_repo_name_from_url,
    get_clone_directory,
    is_git_repository,
)
from .url_validator import (
    check_remote_repository_exists,
    generate_repo_hash,
    is_valid_git_url,
    parse_github_url,
    sanitize_project_name,
)
from .virtual_repo import VirtualDirectory, VirtualRepository

__all__ = [
    # Virtual repository classes
    "VirtualRepository",
    "VirtualDirectory",
    # Cache management
    "cache_virtual_repository",
    "clear_repository_cache", 
    "ensure_virtual_repository",
    "get_cached_repositories",
    "get_virtual_directory",
    "get_virtual_repository",
    # Git operations
    "clone_repository",
    "create_new_repository",
    "download_github_zip_to_memory",
    "extract_repo_name_from_url",
    "get_clone_directory",
    "is_git_repository",
    # URL validation and parsing
    "check_remote_repository_exists",
    "generate_repo_hash",
    "is_valid_git_url",
    "parse_github_url",
    "sanitize_project_name",
]
