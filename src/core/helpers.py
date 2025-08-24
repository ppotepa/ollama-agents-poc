"""Repository validation for coding agents."""

import os
import subprocess
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple
from .enums import AgentCapability, Domain


def check_agent_supports_coding(agent_name: str) -> bool:
    """Check if the given agent supports coding by looking up its configuration."""
    try:
        from integrations.model_config_reader import ModelConfigReader
        
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


def generate_repo_hash(git_url: str) -> str:
    """Generate a consistent 5-character hash from a git URL for directory naming."""
    # Normalize the URL (remove .git suffix, convert to lowercase)
    normalized_url = git_url.lower().rstrip('/')
    if normalized_url.endswith('.git'):
        normalized_url = normalized_url[:-4]
    
    # Generate SHA256 hash
    hash_object = hashlib.sha256(normalized_url.encode('utf-8'))
    return hash_object.hexdigest()[:5]  # Use first 5 characters as requested


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
    """Get capabilities for a given agent based on YAML configuration."""
    capabilities = []
    
    # Check if agent supports coding from YAML configuration
    if check_agent_supports_coding(agent_name):
        capabilities.extend([AgentCapability.CODE, AgentCapability.REPO_ANALYSIS, AgentCapability.FILES])
    else:
        # Non-coding agents get basic capabilities
        capabilities.extend([AgentCapability.STREAMING, AgentCapability.FUNCTION_CALLS])
    
    return capabilities


def requires_repository(capabilities: List[AgentCapability]) -> bool:
    """Check if the given capabilities require a git repository."""
    return AgentCapability.CODE in capabilities or AgentCapability.REPO_ANALYSIS in capabilities


def validate_repository_requirement(agent_name: str, path: str = ".", git_url: Optional[str] = None, data_path: Optional[str] = None, optional: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Validate that coding agents are run in a git repository or clone from URL.
    
    Args:
        agent_name: Name of the agent
        path: Path to check for git repository (default: current directory)
        git_url: Optional git URL to clone if not in a repository
        data_path: Optional custom data directory path for clones
        optional: If True, allows coding agents to work without repositories
        
    Returns:
        Tuple of (validation_passed, working_directory)
        
    Raises:
        ValueError: If a coding agent is used without a repository and no git_url provided (only when optional=False)
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
        if not optional:
            raise ValueError(
                f"Agent '{agent_name}' requires a git repository. "
                f"Please run this command in a git repository directory, "
                f"initialize one with 'git init', or provide a git URL with --git-repo."
            )
        else:
            # Optional mode - proceed without repository
            return True, path
    
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


def list_existing_repositories(data_path: str = "./data") -> List[dict]:
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


def display_repository_selection(data_path: str = "./data") -> str:
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
        print("  • Enter a number (1-{}) to select an existing repository".format(len(repositories)))
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
