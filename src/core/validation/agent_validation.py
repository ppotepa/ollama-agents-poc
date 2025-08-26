"""Agent validation and repository requirements."""

from typing import List, Optional, Tuple
from enum import Enum

from ..io.clone_operations import get_clone_directory, clone_repository, is_git_repository
from ..io.utility_functions import check_agent_supports_coding


class AgentCapability(Enum):
    """Enum defining different agent capabilities."""
    CODE = "code"
    REPO_ANALYSIS = "repo_analysis"
    FILES = "files"
    STREAMING = "streaming"
    FUNCTION_CALLS = "function_calls"


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


def validate_repository_requirement(
    agent_name: str, 
    path: str = ".", 
    git_url: Optional[str] = None, 
    data_path: Optional[str] = None, 
    optional: bool = False, 
    interception_mode: str = "smart"
) -> Tuple[bool, Optional[str]]:
    """
    Validate that coding agents are run in a git repository or clone from URL.

    Args:
        agent_name: Name of the agent
        path: Path to check for git repository (default: current directory)
        git_url: Optional git URL to clone if not in a repository
        data_path: Optional custom data directory path for clones
        optional: If True, allows coding agents to work without repositories
        interception_mode: Mode for prompt interception (lightweight/full/smart)

    Returns:
        Tuple of (validation_passed, working_directory)

    Raises:
        ValueError: If a coding agent is used without a repository and no git_url provided (only when optional=False)
    """
    capabilities = get_agent_capabilities(agent_name)

    if requires_repository(capabilities):
        # If git URL is provided, handle based on interception mode
        if git_url:
            # For lightweight mode, use virtual repository instead of full cloning
            if interception_mode == "lightweight":
                print(f"⚡ Using lightweight mode - virtual repository for: {git_url}")
                try:
                    # Ensure virtual repository is available
                    _ensure_virtual_repository(git_url)
                    return True, path  # Stay in current directory
                except Exception as e:
                    print(f"⚠️  Virtual repository setup failed: {e}")
                    # Fallback to normal cloning if virtual fails

            # Set data path to default if not provided
            if data_path is None:
                data_path = "./data"
            
            print(f"🔄 Setting up repository: {git_url}")
            print(f"📂 Data path: {data_path}")
            
            # Normal cloning for full mode or fallback
            clone_dir = get_clone_directory(git_url, data_path)
            print(f"🎯 Target directory: {clone_dir}")
            
            if clone_repository(git_url, clone_dir):
                print(f"✅ Repository successfully set up at: {clone_dir}")
                return True, str(clone_dir)
            else:
                raise ValueError(
                    f"Failed to clone repository '{git_url}' for agent '{agent_name}'. "
                    f"Please check the URL and your network connection."
                )

        # If no git URL provided, check if current directory is already a git repository
        if is_git_repository(path):
            return True, path

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


def _ensure_virtual_repository(git_url: str):
    """Ensure virtual repository is available for lightweight mode."""
    try:
        from src.core.state import get_virtual_repository
        virtual_repo = get_virtual_repository()
        
        # Check if URL is already loaded
        if not virtual_repo.has_repository(git_url):
            # Load repository into virtual cache
            virtual_repo.load_from_url(git_url)
            
        print(f"✅ Virtual repository ready for: {git_url}")
        
    except ImportError:
        raise Exception("Virtual repository system not available")
    except Exception as e:
        raise Exception(f"Failed to setup virtual repository: {e}")
