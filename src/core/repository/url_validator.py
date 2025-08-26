"""URL validation and parsing utilities for repository operations."""

import re
import subprocess
from typing import Optional


def parse_github_url(git_url: str) -> Optional[dict]:
    """
    Parse a GitHub URL and return repository information.

    Args:
        git_url: GitHub repository URL (https://github.com/owner/repo or git@github.com:owner/repo.git)

    Returns:
        Dict with 'owner', 'repo', and 'branch' keys, or None if not a valid GitHub URL
    """
    # Patterns for different GitHub URL formats
    patterns = [
        r'https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?(?:/tree/([^/]+))?$',
        r'git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$',
        r'github\.com/([^/]+)/([^/]+?)(?:\.git)?/?(?:/tree/([^/]+))?$'
    ]

    for pattern in patterns:
        match = re.match(pattern, git_url.strip())
        if match:
            owner, repo = match.groups()[:2]
            branch = match.groups()[2] if len(match.groups()) > 2 and match.groups()[2] else 'main'
            return {'owner': owner, 'repo': repo, 'branch': branch}

    return None


def is_valid_git_url(text: str) -> bool:
    """Check if the given text is a valid git repository URL."""
    if not text:
        return False

    # Common git URL patterns
    git_url_patterns = [
        r'^https?://[\w\.-]+/[\w\.-]+/[\w\.-]+\.git$',  # https://domain/user/repo.git
        r'^https?://[\w\.-]+/[\w\.-]+/[\w\.-]+/?$',     # https://domain/user/repo
        r'^git@[\w\.-]+:[\w\.-]+/[\w\.-]+\.git$',       # git@domain:user/repo.git
        r'^git://[\w\.-]+/[\w\.-]+/[\w\.-]+\.git$',     # git://domain/user/repo.git
    ]

    for pattern in git_url_patterns:
        if re.match(pattern, text):
            return True

    # Additional check for common git hosting domains
    common_hosts = ['github.com', 'gitlab.com', 'bitbucket.org', 'gitea.com', 'codeberg.org']
    return any(host in text and '/' in text for host in common_hosts)


def sanitize_project_name(name: str) -> str:
    """Sanitize a project name to be valid for directory and git repository names."""
    # Remove or replace invalid characters for file systems
    sanitized = re.sub(r'[<>:"/\\|?*]', '-', name)

    # Replace spaces with hyphens
    sanitized = re.sub(r'\s+', '-', sanitized)

    # Remove multiple consecutive hyphens
    sanitized = re.sub(r'-+', '-', sanitized)

    # Remove leading/trailing hyphens and dots
    sanitized = sanitized.strip('-.')

    # Ensure it's not empty and not too long
    if not sanitized:
        sanitized = "new-project"
    elif len(sanitized) > 50:
        sanitized = sanitized[:50].rstrip('-.')

    return sanitized


def check_remote_repository_exists(git_url: str) -> bool:
    """Check if a remote git repository exists and is accessible."""
    try:
        # Use git ls-remote to check if repository exists
        result = subprocess.run(
            ["git", "ls-remote", "--heads", git_url],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=30  # 30 second timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"⚠️  Timeout checking repository: {git_url}")
        return False
    except FileNotFoundError:
        print("⚠️  Git command not found. Please install Git.")
        return False
    except Exception as e:
        print(f"⚠️  Error checking repository {git_url}: {e}")
        return False


def generate_repo_hash(git_url: str) -> str:
    """Generate a short hash for repository URL identification."""
    import hashlib
    
    # Normalize URL for consistent hashing
    normalized_url = git_url.lower().strip()
    if normalized_url.endswith('.git'):
        normalized_url = normalized_url[:-4]

    # Generate SHA256 hash
    hash_object = hashlib.sha256(normalized_url.encode('utf-8'))
    return hash_object.hexdigest()[:5]  # Use first 5 characters
