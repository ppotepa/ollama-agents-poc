"""I/O adapters package for side-effects and external integrations."""

from .github_operations import (
    parse_github_url,
    download_github_zip,
    download_github_zip_to_memory,
    get_github_download_url
)
from .utility_functions import (
    generate_repo_hash,
    check_agent_supports_coding,
    validate_model_id,
    normalize_path,
    get_file_extension,
    is_code_file,
    is_documentation_file,
    get_directory_depth,
    format_file_size
)
from .clone_operations import (
    get_clone_directory,
    clone_repository,
    is_valid_git_url,
    is_git_repository,
    check_remote_repository_exists,
    extract_repo_name_from_url,
    sanitize_project_name,
    create_new_repository
)
from .repository_operations import (
    get_repository_info,
    list_existing_repositories,
    display_repository_selection,
    is_git_url
)

__all__ = [
    "parse_github_url",
    "download_github_zip", 
    "download_github_zip_to_memory",
    "get_github_download_url",
    "generate_repo_hash",
    "check_agent_supports_coding",
    "validate_model_id",
    "normalize_path",
    "get_file_extension",
    "is_code_file",
    "is_documentation_file",
    "get_directory_depth",
    "format_file_size",
    "get_clone_directory",
    "clone_repository",
    "is_valid_git_url",
    "is_git_repository",
    "check_remote_repository_exists",
    "extract_repo_name_from_url",
    "sanitize_project_name",
    "create_new_repository",
    "get_repository_info",
    "list_existing_repositories", 
    "display_repository_selection",
    "is_git_url"
]
