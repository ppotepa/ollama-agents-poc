"""Repository cloning operations."""

import subprocess
from pathlib import Path
from typing import Optional

from .github_operations import parse_github_url, download_github_zip_to_memory
from .utility_functions import generate_repo_hash


def get_clone_directory(git_url_or_name: str, base_path: str = None) -> Path:
    """Get the directory path where a repository should be cloned or project created."""
    if base_path is None:
        # Always use data folder in the original working directory when the script was started
        base_path = "./data"

    # If it's a valid git URL, use hash-based naming
    if is_valid_git_url(git_url_or_name):
        repo_hash = generate_repo_hash(git_url_or_name)
        clone_path = Path(base_path) / repo_hash
    else:
        # For project names, use sanitized name directly
        sanitized_name = sanitize_project_name(git_url_or_name)
        clone_path = Path(base_path) / sanitized_name

    return clone_path.resolve()  # Return absolute path


def clone_repository(git_url: str, target_dir: Path) -> bool:
    """
    Clone a git repository to the specified directory.
    For GitHub repositories, uses ZIP download for faster performance.
    If input is not a valid git URL, creates a new local repository with that name.
    If repository doesn't exist remotely, creates a new local repository.

    Args:
        git_url: URL of the git repository to clone OR project name for new repository
        target_dir: Directory where the repository should be cloned

    Returns:
        True if cloning/creation was successful, False otherwise
    """
    try:
        # Check if directory already exists and is a git repo
        if target_dir.exists() and is_git_repository(str(target_dir)):
            print(f"📁 Repository already exists at {target_dir}, refreshing...")
            
            # Check if remote origin exists
            try:
                remote_check = subprocess.run(
                    ["git", "remote", "-v"],
                    cwd=str(target_dir),
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
                
                # If remote exists and this is a valid git URL, try pulling
                if remote_check.returncode == 0 and "origin" in remote_check.stdout and is_valid_git_url(git_url):
                    print("🔄 Pulling latest changes...")
                    result = subprocess.run(
                        ["git", "pull", "origin", "main"],
                        cwd=str(target_dir),
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace'
                    )
                    if result.returncode != 0:
                        # Try pulling from master branch if main fails
                        result = subprocess.run(
                            ["git", "pull", "origin", "master"],
                            cwd=str(target_dir),
                            capture_output=True,
                            text=True,
                            encoding='utf-8',
                            errors='replace'
                        )
                else:
                    print("✅ Using existing local repository (no remote to pull from)")
                    result = subprocess.CompletedProcess(args=[], returncode=0)
            except Exception:
                # If something goes wrong, just consider it successful and continue
                print("✅ Using existing repository as-is")
                result = subprocess.CompletedProcess(args=[], returncode=0)

            # Load repository context into memory
            try:
                from src.tools.context import load_repository_context_after_clone
                if load_repository_context_after_clone(str(target_dir), cache_content=True, quiet=True):
                    print("🧠 Repository context refreshed in memory")
            except Exception as e:
                print(f"⚠️  Could not refresh repository context: {e}")

            return True  # Consider it successful if the repo already exists

        # Check if input is a valid git URL
        if not is_valid_git_url(git_url):
            print(f"📝 Input '{git_url}' is not a valid git URL - treating as project name")

            # Sanitize the project name
            project_name = sanitize_project_name(git_url)
            print(f"🆕 Creating new local project: {project_name}")

            # Create new repository locally with the project name
            return create_new_repository(target_dir, project_name, None)

        # It's a valid git URL - check if remote repository exists
        print(f"🔍 Checking if remote repository exists: {git_url}")
        if not check_remote_repository_exists(git_url):
            print(f"❌ Remote repository does not exist or is not accessible: {git_url}")

            # Extract repository name from URL
            repo_name = extract_repo_name_from_url(git_url)
            print(f"🆕 Creating new local repository with name: {repo_name}")

            # Create new repository locally
            return create_new_repository(target_dir, repo_name, git_url)

        print("✅ Remote repository exists, proceeding with clone")

        # Try GitHub ZIP download first (much faster)
        if parse_github_url(git_url):
            print("🚀 Detected GitHub repository, using fast ZIP download...")
            if download_github_zip_to_memory(git_url, target_dir):
                # Set up virtual directory for context
                try:
                    from src.tools.context import set_virtual_directory_for_context
                    set_virtual_directory_for_context(git_url)
                except ImportError:
                    print("⚠️  Virtual directory context not available")
                return True
            else:
                print("⚠️  ZIP download failed, falling back to git clone...")

        # Fall back to git clone for non-GitHub URLs or if ZIP download failed
        # Create parent directory if it doesn't exist
        target_dir.parent.mkdir(parents=True, exist_ok=True)

        # If directory exists but is not a git repo, remove it
        if target_dir.exists():
            print(f"Removing existing non-git directory: {target_dir}")
            import shutil
            shutil.rmtree(target_dir)

        # Clone the repository using git
        print(f"Cloning {git_url} to {target_dir}...")
        result = subprocess.run(
            ["git", "clone", git_url, str(target_dir)],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        if result.returncode == 0:
            print(f"✅ Successfully cloned repository to {target_dir}")

            # Load repository context into memory for fast access
            try:
                from src.tools.context import load_repository_context_after_clone
                if load_repository_context_after_clone(str(target_dir), cache_content=True, quiet=True):
                    print("🧠 Repository context loaded into memory")
            except ImportError:
                print("⚠️  Repository context system not available")
            except Exception as e:
                print(f"⚠️  Could not load repository context: {e}")

            return True
        else:
            print(f"❌ Failed to clone repository:")
            print(f"   Error: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error cloning repository: {e}")
        return False


def is_valid_git_url(url: str) -> bool:
    """Check if the URL is a valid git repository URL."""
    if not url or not isinstance(url, str):
        return False

    # Common git URL patterns
    git_patterns = [
        r'^https?://github\.com/[\w\-\.]+/[\w\-\.]+/?$',
        r'^https?://gitlab\.com/[\w\-\.]+/[\w\-\.]+/?$',
        r'^https?://bitbucket\.org/[\w\-\.]+/[\w\-\.]+/?$',
        r'^git@github\.com:[\w\-\.]+/[\w\-\.]+\.git$',
        r'^git@gitlab\.com:[\w\-\.]+/[\w\-\.]+\.git$',
        r'^https?://.*\.git$'
    ]

    import re
    return any(re.match(pattern, url.strip()) for pattern in git_patterns)


def is_git_repository(directory: str) -> bool:
    """Check if directory is a git repository."""
    git_dir = Path(directory) / '.git'
    return git_dir.exists()


def check_remote_repository_exists(git_url: str) -> bool:
    """Check if remote repository exists and is accessible."""
    try:
        result = subprocess.run(
            ["git", "ls-remote", "--heads", git_url],
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
            encoding='utf-8',
            errors='replace'
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


def extract_repo_name_from_url(git_url: str) -> str:
    """Extract repository name from git URL."""
    # Remove common git URL parts and extensions
    name = git_url.rstrip('/')
    
    if name.endswith('.git'):
        name = name[:-4]
    
    # Extract the repository name (last part of the path)
    if '/' in name:
        name = name.split('/')[-1]
    
    return sanitize_project_name(name)


def sanitize_project_name(name: str) -> str:
    """Sanitize project name for use as directory name."""
    import re
    
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = "new_project"
    
    return sanitized


def create_new_repository(target_dir: Path, name: str, remote_url: Optional[str] = None) -> bool:
    """Create a new local git repository."""
    try:
        # Create directory
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize git repository
        result = subprocess.run(
            ["git", "init"],
            cwd=str(target_dir),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ Failed to initialize git repository: {result.stderr}")
            return False
        
        # Create README file
        readme_path = target_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(f"# {name}\n\nA new project created by the coding agent.\n")
        
        # Add and commit initial files
        subprocess.run(["git", "add", "README.md"], cwd=str(target_dir))
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(target_dir))
        
        # Add remote if provided
        if remote_url:
            subprocess.run(["git", "remote", "add", "origin", remote_url], cwd=str(target_dir))
        
        print(f"✅ Created new repository: {name}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating new repository: {e}")
        return False
