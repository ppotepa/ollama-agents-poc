"""Git operations for repository cloning, downloading, and management."""

import subprocess
import tempfile
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional

from .url_validator import is_valid_git_url, parse_github_url, sanitize_project_name, check_remote_repository_exists
from .virtual_repo import VirtualRepository, VirtualDirectory
from .cache_manager import cache_virtual_repository


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
            print(f"📁 Repository already exists at {target_dir}, pulling latest changes...")
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

            # Load repository context into memory if pull was successful
            if result.returncode == 0:
                try:
                    from src.tools.context import load_repository_context_after_clone
                    if load_repository_context_after_clone(str(target_dir), cache_content=True, quiet=True):
                        print("🧠 Repository context refreshed in memory")
                except Exception as e:
                    print(f"⚠️  Could not refresh repository context: {e}")

            return result.returncode == 0

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
            print(f"✓ Successfully cloned repository to {target_dir}")

            # Load repository context into memory after successful clone
            try:
                from src.tools.context import load_repository_context_after_clone
                if load_repository_context_after_clone(str(target_dir), cache_content=True, quiet=True):
                    print("🧠 Repository context loaded into memory")
            except Exception as e:
                print(f"⚠️  Could not load repository context: {e}")

            return True
        else:
            print(f"✗ Failed to clone repository: {result.stderr}")
            return False

    except Exception as e:
        print(f"✗ Error cloning repository: {e}")
        return False


def download_github_zip_to_memory(git_url: str, target_dir: Path) -> bool:
    """
    Download a GitHub repository as ZIP and extract it to disk while also creating virtual repository.

    Args:
        git_url: GitHub repository URL
        target_dir: Directory where the repository should be extracted

    Returns:
        True if download and extraction was successful, False otherwise
    """
    try:
        github_info = parse_github_url(git_url)
        if not github_info:
            print(f"❌ Invalid GitHub URL: {git_url}")
            return False

        owner = github_info['owner']
        repo = github_info['repo']
        branch = github_info['branch']

        # Construct ZIP download URL
        zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"

        print(f"⬇️  Downloading ZIP from: {zip_url}")

        # Download ZIP data
        with urllib.request.urlopen(zip_url) as response:
            zip_data = response.read()

        print(f"✅ Downloaded {len(zip_data)} bytes")

        # Create virtual repository in memory
        virtual_repo = VirtualRepository(git_url, zip_data)
        cache_virtual_repository(git_url, virtual_repo)

        # Extract to disk
        _extract_virtual_repo_to_disk(virtual_repo, target_dir)

        print(f"✅ Successfully extracted repository to {target_dir}")
        return True

    except Exception as e:
        print(f"❌ Error downloading GitHub ZIP: {e}")
        return False


def _extract_virtual_repo_to_disk(virtual_repo: VirtualRepository, target_dir: Path):
    """Extract virtual repository contents to disk."""
    try:
        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)

        # Extract ZIP file to disk
        with zipfile.ZipFile(BytesIO(virtual_repo.zip_data), 'r') as zip_ref:
            # GitHub ZIP files have a root folder named "repo-branch"
            # We need to extract and rename it to our target directory

            # First, extract to a temporary directory
            with tempfile.TemporaryDirectory() as temp_extract_dir:
                zip_ref.extractall(temp_extract_dir)

                # Find the extracted folder (should be only one)
                extracted_items = list(Path(temp_extract_dir).iterdir())
                if not extracted_items:
                    raise ValueError("No items found in extracted ZIP")

                source_dir = extracted_items[0]
                if not source_dir.is_dir():
                    raise ValueError("Expected directory in ZIP root")

                # Copy contents to target directory
                import shutil
                for item in source_dir.iterdir():
                    dest_path = target_dir / item.name
                    if item.is_file():
                        shutil.copy2(item, dest_path)
                    elif item.is_dir():
                        shutil.copytree(item, dest_path, dirs_exist_ok=True)

    except Exception as e:
        print(f"❌ Error extracting virtual repo to disk: {e}")
        raise


def is_git_repository(path: str) -> bool:
    """Check if a directory is a git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=path,
            capture_output=True,
            text=True
        )
        return result.returncode == 0 and result.stdout.strip() == "true"
    except Exception:
        return False


def create_new_repository(target_dir: Path, repo_name: str, origin_url: Optional[str] = None) -> bool:
    """Create a new git repository locally."""
    try:
        # Create directory structure
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

        # Create basic files
        readme_content = f"# {repo_name}\n\nA new project created by Ollama Agents.\n"
        (target_dir / "README.md").write_text(readme_content)

        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
        (target_dir / ".gitignore").write_text(gitignore_content)

        # Initial commit
        subprocess.run(["git", "add", "."], cwd=str(target_dir), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=str(target_dir),
            capture_output=True
        )

        # Add remote origin if provided
        if origin_url:
            subprocess.run(
                ["git", "remote", "add", "origin", origin_url],
                cwd=str(target_dir),
                capture_output=True
            )

        print(f"✅ Created new repository: {repo_name}")
        return True

    except Exception as e:
        print(f"❌ Error creating new repository: {e}")
        return False


def extract_repo_name_from_url(git_url: str) -> str:
    """Extract repository name from git URL."""
    import re
    
    # Try to extract from common patterns
    patterns = [
        r'https?://[^/]+/[^/]+/([^/]+?)(?:\.git)?/?$',  # https://host/user/repo.git
        r'git@[^:]+:([^/]+/)?([^/]+?)(?:\.git)?$',      # git@host:user/repo.git
    ]

    for pattern in patterns:
        match = re.search(pattern, git_url)
        if match:
            # Get the last group (repo name)
            groups = match.groups()
            repo_name = groups[-1] if groups else None
            if repo_name:
                return sanitize_project_name(repo_name)

    # Fallback: sanitize the entire URL
    return sanitize_project_name(git_url)


def get_clone_directory(git_url_or_name: str, base_path: str = None) -> Path:
    """Get the target directory for cloning a repository."""
    if base_path is None:
        base_path = "./repositories"

    base = Path(base_path)
    
    # If it's a URL, extract the repo name
    if is_valid_git_url(git_url_or_name):
        repo_name = extract_repo_name_from_url(git_url_or_name)
    else:
        # It's a project name, sanitize it
        repo_name = sanitize_project_name(git_url_or_name)

    return base / repo_name
