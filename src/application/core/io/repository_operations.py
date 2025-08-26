"""Git and repository management operations."""

import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from .clone_operations import is_git_repository, get_clone_directory, clone_repository


def get_repository_info(path: str = ".") -> Optional[Dict]:
    """Get information about the git repository."""
    if not is_git_repository(path):
        return None

    try:
        # Get repository URL
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=path,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        remote_url = result.stdout.strip() if result.returncode == 0 else None

        # Get current branch
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=path,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        current_branch = result.stdout.strip() if result.returncode == 0 else None

        # Get commit count
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        commit_count = int(result.stdout.strip()) if result.returncode == 0 else 0

        return {
            "remote_url": remote_url,
            "current_branch": current_branch,
            "commit_count": commit_count,
            "is_git_repo": True
        }
    except Exception:
        return {"is_git_repo": True}


def list_existing_repositories(data_path: str = "./data") -> List[Dict]:
    """List all existing repositories in the data folder with their info."""
    repositories = []
    data_dir = Path(data_path)

    if not data_dir.exists():
        return repositories

    for item in data_dir.iterdir():
        if item.is_dir() and len(item.name) == 5:  # Check if it's a hash folder
            repo_info = get_repository_info(str(item))
            if repo_info and repo_info.get("is_git_repo", False):
                repo_name = "Unknown Repository"
                remote_url = repo_info.get("remote_url", "")

                # Extract repository name from URL
                if remote_url:
                    if remote_url.endswith('.git'):
                        remote_url = remote_url[:-4]
                    repo_name = remote_url.split('/')[-1] if '/' in remote_url else remote_url

                repositories.append({
                    "hash": item.name,
                    "name": repo_name,
                    "path": str(item),
                    "remote_url": repo_info.get("remote_url", ""),
                    "branch": repo_info.get("current_branch", ""),
                    "commits": repo_info.get("commit_count", 0)
                })

    return repositories


def display_repository_selection(data_path: str = "./data") -> Optional[str]:
    """Display available repositories and prompt for selection or new URL."""
    repositories = list_existing_repositories(data_path)

    print("\n" + "="*60)
    print("🗂️  REPOSITORY SELECTION")
    print("="*60)

    if repositories:
        print("📁 Available repositories:")
        for i, repo in enumerate(repositories, 1):
            print(f"  {i}. {repo['name']} ({repo['hash']})")
            if repo['remote_url']:
                print(f"     📍 {repo['remote_url']}")
            if repo['branch']:
                print(f"     🌿 Branch: {repo['branch']} ({repo['commits']} commits)")
            print()
    else:
        print("📂 No existing repositories found in data folder.")
        print()

    print("Options:")
    if repositories:
        print(f"  • Enter a number (1-{len(repositories)}) to select an existing repository")
    print("  • Paste a git URL to clone a new repository")
    print("  • Press Enter to continue without a repository")

    while True:
        user_input = input("\nYour choice: ").strip()

        # Empty input - continue without repository
        if not user_input:
            return None

        # Check if it's a number selection
        if user_input.isdigit():
            selection = int(user_input)
            if 1 <= selection <= len(repositories):
                selected_repo = repositories[selection - 1]
                print(f"✓ Selected: {selected_repo['name']} ({selected_repo['hash']})")
                return selected_repo['remote_url']
            else:
                print(f"❌ Invalid selection. Please enter a number between 1 and {len(repositories)}")
                continue

        # Check if it looks like a git URL
        if is_git_url(user_input):
            return user_input
        else:
            print("❌ Invalid input. Please enter a number, git URL, or press Enter to skip.")
            continue


def is_git_url(url: str) -> bool:
    """Check if the given string looks like a git URL."""
    git_patterns = [
        'github.com',
        'gitlab.com',
        'bitbucket.org',
        '.git',
        'git@',
        'https://',
        'http://',
        'ssh://'
    ]

    url_lower = url.lower()
    return any(pattern in url_lower for pattern in git_patterns)
