"""Utility functions for repository operations."""

import hashlib
import os
import re
from typing import Optional


def generate_repo_hash(git_url: str) -> str:
    """Generate a consistent 5-character hash from a git URL for directory naming."""
    # Normalize the URL (remove .git suffix, convert to lowercase)
    normalized_url = git_url.lower().rstrip('/')
    if normalized_url.endswith('.git'):
        normalized_url = normalized_url[:-4]

    # Create MD5 hash and take first 5 characters
    hash_object = hashlib.md5(normalized_url.encode())
    return hash_object.hexdigest()[:5]


def check_agent_supports_coding(agent_name: str) -> bool:
    """Check if the given agent supports coding by looking up its configuration."""
    try:
        from src.integrations.model_config_reader import ModelConfigReader

        # Load the model configuration
        config_reader = ModelConfigReader('src/config/models.yaml')

        # Look for the agent by short_name
        model_config = config_reader.get_model(agent_name)
        if model_config:
            return model_config.supports_coding

        # If not found by short name, try to find by model ID
        for model in config_reader.get_all_models():
            if agent_name in model.model_id or model.model_id.startswith(agent_name):
                return model.supports_coding

        # Default to False if agent not found in configuration
        return False

    except Exception as e:
        print(f"⚠️  Warning: Could not determine coding capability for agent '{agent_name}': {e}")
        # For safety, assume coding agents need repositories
        return True


def validate_model_id(model_id: str) -> bool:
    """Validate model ID format."""
    if not model_id or not isinstance(model_id, str):
        return False
    
    # Basic validation - model IDs should be alphanumeric with some allowed separators
    pattern = r'^[a-zA-Z0-9][a-zA-Z0-9\-_\.:]*[a-zA-Z0-9]$'
    return bool(re.match(pattern, model_id))


def normalize_path(path: str) -> str:
    """Normalize file paths for consistent handling."""
    # Convert backslashes to forward slashes
    normalized = path.replace('\\', '/')
    
    # Remove leading ./
    if normalized.startswith('./'):
        normalized = normalized[2:]
    
    # Remove leading /
    if normalized.startswith('/'):
        normalized = normalized[1:]
    
    return normalized


def get_file_extension(path: str) -> str:
    """Get file extension from path."""
    return os.path.splitext(path)[1].lower()


def is_code_file(path: str) -> bool:
    """Check if file is a code file based on extension."""
    code_extensions = {
        '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.php',
        '.rb', '.go', '.rs', '.swift', '.kt', '.scala', '.html', '.css',
        '.jsx', '.tsx', '.vue', '.sql', '.sh', '.bash', '.ps1', '.yml',
        '.yaml', '.json', '.xml', '.dockerfile', '.toml', '.ini', '.cfg'
    }
    return get_file_extension(path) in code_extensions


def is_documentation_file(path: str) -> bool:
    """Check if file is a documentation file."""
    doc_extensions = {'.md', '.rst', '.txt', '.doc', '.docx', '.pdf'}
    doc_names = {'readme', 'license', 'changelog', 'contributing', 'authors'}
    
    extension = get_file_extension(path)
    filename = os.path.basename(path).lower()
    
    return (extension in doc_extensions or 
            any(name in filename for name in doc_names))


def get_directory_depth(path: str) -> int:
    """Get the depth of a directory path."""
    normalized = normalize_path(path)
    if not normalized:
        return 0
    return len(normalized.split('/'))


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
