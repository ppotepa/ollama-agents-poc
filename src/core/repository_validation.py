"""Repository validation for coding agents."""

import os
import subprocess
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple
from .enums import AgentCapability, Domain


def generate_repo_hash(git_url: str) -> str:
    """Generate a consistent hash from a git URL for directory naming."""
    # Normalize the URL (remove .git suffix, convert to lowercase)
    normalized_url = git_url.lower().rstrip('/')
    if normalized_url.endswith('.git'):
        normalized_url = normalized_url[:-4]
    
    # Generate SHA256 hash
    hash_object = hashlib.sha256(normalized_url.encode('utf-8'))
    return hash_object.hexdigest()[:12]  # Use first 12 characters for readability


def get_clone_directory(git_url: str, base_path: str = None) -> Path:
    """Get the directory path where a repository should be cloned."""
    if base_path is None:
        # Always use data folder in the original working directory when the script was started
        base_path = "./data"
    
    repo_hash = generate_repo_hash(git_url)
    clone_path = Path(base_path) / repo_hash
    return clone_path.resolve()  # Return absolute path


def clone_repository(git_url: str, target_dir: Path) -> bool:
    """
    Clone a git repository to the specified directory.
    
    Args:
        git_url: URL of the git repository to clone
        target_dir: Directory where the repository should be cloned
        
    Returns:
        True if cloning was successful, False otherwise
    """
    try:
        # Create parent directory if it doesn't exist
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # If directory already exists and is a git repo, just pull
        if target_dir.exists() and is_git_repository(str(target_dir)):
            print(f"Repository already exists at {target_dir}, pulling latest changes...")
            result = subprocess.run(
                ["git", "pull", "origin", "main"],
                cwd=str(target_dir),
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                # Try pulling from master branch if main fails
                result = subprocess.run(
                    ["git", "pull", "origin", "master"],
                    cwd=str(target_dir),
                    capture_output=True,
                    text=True
                )
            return result.returncode == 0
        
        # If directory exists but is not a git repo, remove it
        if target_dir.exists():
            print(f"Removing existing non-git directory: {target_dir}")
            import shutil
            shutil.rmtree(target_dir)
        
        # Clone the repository
        print(f"Cloning {git_url} to {target_dir}...")
        result = subprocess.run(
            ["git", "clone", git_url, str(target_dir)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✓ Successfully cloned repository to {target_dir}")
            return True
        else:
            print(f"✗ Failed to clone repository: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Error cloning repository: {e}")
        return False


def is_git_repository(path: str = ".") -> bool:
    """Check if the current directory is a git repository."""
    try:
        # Check if .git directory exists
        git_dir = os.path.join(path, ".git")
        if os.path.exists(git_dir):
            return True
        
        # Alternative: use git command to check
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=path,
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def is_coding_agent(agent_name: str) -> bool:
    """Check if an agent is a coding agent that requires repository validation."""
    # List of known coding agents
    coding_agents = {
        "deepcoder", "coder", "code-assistant", "programming-assistant",
        "developer", "code-generator", "software-engineer"
    }
    return agent_name.lower() in coding_agents


def get_agent_capabilities(agent_name: str) -> List[AgentCapability]:
    """Get capabilities for a given agent."""
    # This could be enhanced to read from agent metadata
    # For now, using simple mapping
    capability_map = {
        "deepcoder": [AgentCapability.CODE, AgentCapability.REPO_ANALYSIS, AgentCapability.FILES],
        "coder": [AgentCapability.CODE, AgentCapability.REPO_ANALYSIS],
        "assistant": [AgentCapability.STREAMING, AgentCapability.FUNCTION_CALLS],
    }
    
    return capability_map.get(agent_name.lower(), [])


def requires_repository(capabilities: List[AgentCapability]) -> bool:
    """Check if the given capabilities require a git repository."""
    return AgentCapability.CODE in capabilities or AgentCapability.REPO_ANALYSIS in capabilities


def validate_repository_requirement(agent_name: str, path: str = ".", git_url: Optional[str] = None, data_path: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate that coding agents are run in a git repository or clone from URL.
    
    Args:
        agent_name: Name of the agent
        path: Path to check for git repository (default: current directory)
        git_url: Optional git URL to clone if not in a repository
        data_path: Optional custom data directory path for clones
        
    Returns:
        Tuple of (validation_passed, working_directory)
        
    Raises:
        ValueError: If a coding agent is used without a repository and no git_url provided
    """
    capabilities = get_agent_capabilities(agent_name)
    
    if requires_repository(capabilities):
        # Check if current directory is already a git repository
        if is_git_repository(path):
            return True, path
        
        # If git URL is provided, clone the repository
        if git_url:
            clone_dir = get_clone_directory(git_url, data_path)
            if clone_repository(git_url, clone_dir):
                return True, str(clone_dir)
            else:
                raise ValueError(
                    f"Failed to clone repository '{git_url}' for agent '{agent_name}'. "
                    f"Please check the URL and your network connection."
                )
        
        # No git repository and no URL provided
        raise ValueError(
            f"Agent '{agent_name}' requires a git repository. "
            f"Please run this command in a git repository directory, "
            f"initialize one with 'git init', or provide a git URL with --git-repo."
        )
    
    return True, path


def get_repository_info(path: str = ".") -> Optional[dict]:
    """Get information about the git repository."""
    if not is_git_repository(path):
        return None
    
    try:
        # Get repository URL
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=path,
            capture_output=True,
            text=True
        )
        remote_url = result.stdout.strip() if result.returncode == 0 else None
        
        # Get current branch
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=path,
            capture_output=True,
            text=True
        )
        current_branch = result.stdout.strip() if result.returncode == 0 else None
        
        # Get commit count
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True
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
