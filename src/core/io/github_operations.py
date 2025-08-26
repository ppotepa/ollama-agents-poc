"""GitHub operations for repository handling."""

import re
import shutil
import tempfile
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional

from ..state.virtual_repository import VirtualRepository, VirtualDirectory
from ..state.repository_cache import cache_virtual_repository


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


def download_github_zip(git_url: str, target_dir: Path) -> bool:
    """
    Download a GitHub repository as ZIP and extract it.
    This creates both disk files (for change tracking) and virtual directory (for fast context access).

    Args:
        git_url: GitHub repository URL
        target_dir: Directory where the repository should be extracted

    Returns:
        True if download and extraction was successful, False otherwise
    """
    try:
        github_info = parse_github_url(git_url)
        if not github_info:
            return False  # Not a GitHub URL, fall back to git clone

        owner = github_info['owner']
        repo = github_info['repo']
        branch = github_info['branch']

        # GitHub ZIP download URL
        zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"

        print(f"📦 Downloading {owner}/{repo} (branch: {branch}) as ZIP...")

        # Download ZIP data to memory
        try:
            response = urllib.request.urlopen(zip_url)
            zip_data = response.read()

            print(f"✅ Downloaded ZIP file ({len(zip_data)} bytes)")

        except urllib.error.HTTPError as e:
            if e.code == 404 and branch == 'main':
                # Try with 'master' branch
                print("🔄 Trying with 'master' branch...")
                master_zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/master.zip"

                try:
                    response = urllib.request.urlopen(master_zip_url)
                    zip_data = response.read()
                    print("✅ Downloaded ZIP file (master branch)")

                except Exception:
                    print("❌ Could not download repository with master branch either")
                    return False
            else:
                print(f"❌ Repository not found or access denied (HTTP {e.code})")
                return False

        # Create virtual repository from ZIP data
        virtual_repo = VirtualRepository(git_url, zip_data)
        cache_virtual_repository(git_url, virtual_repo)
        
        print(f"🧠 Created virtual repository: {virtual_repo.metadata['total_files']} files, "
              f"{virtual_repo.metadata['total_directories']} directories")

        # Extract to disk for file system access and change tracking
        # Create parent directory if it doesn't exist
        target_dir.parent.mkdir(parents=True, exist_ok=True)

        # If directory already exists, remove it
        if target_dir.exists():
            print(f"🗑️  Removing existing directory: {target_dir}")
            shutil.rmtree(target_dir)

        # Extract ZIP file to disk
        with zipfile.ZipFile(BytesIO(zip_data), 'r') as zip_ref:
            # GitHub ZIP files have a root folder named "repo-branch"
            # We need to extract and rename it to our target directory

            # First, extract to a temporary directory
            with tempfile.TemporaryDirectory() as temp_extract_dir:
                zip_ref.extractall(temp_extract_dir)

                # Find the extracted folder (should be repo-branch)
                extracted_folders = [f for f in os.listdir(temp_extract_dir)
                                   if os.path.isdir(os.path.join(temp_extract_dir, f))]

                if not extracted_folders:
                    print("❌ No folders found in ZIP file")
                    return False

                # Move the extracted content to target directory
                extracted_folder = os.path.join(temp_extract_dir, extracted_folders[0])
                shutil.move(extracted_folder, str(target_dir))

                print(f"✅ Successfully extracted repository to {target_dir}")

                # Load repository context into memory for fast access
                try:
                    from src.tools.context import load_repository_context_after_clone
                    if load_repository_context_after_clone(str(target_dir), cache_content=True, quiet=True):
                        print("🔍 Starting repository scan...")
                        print("🧠 Repository context loaded into memory")
                except ImportError:
                    print("⚠️  Repository context system not available")
                except Exception as e:
                    print(f"⚠️  Could not load repository context: {e}")

                return True

    except Exception as e:
        print(f"❌ Error downloading repository: {e}")
        return False


def download_github_zip_to_memory(git_url: str, target_dir: Path) -> bool:
    """
    Enhanced version: Download GitHub repository as ZIP and create virtual repository in memory.
    This provides instant access to all files without disk I/O.

    Args:
        git_url: GitHub repository URL
        target_dir: Directory where the repository should be extracted (for compatibility)

    Returns:
        True if download and virtual extraction was successful, False otherwise
    """
    try:
        github_info = parse_github_url(git_url)
        if not github_info:
            return False  # Not a GitHub URL, fall back to git clone

        owner = github_info['owner']
        repo = github_info['repo']
        branch = github_info['branch']

        # Check if we already have this repository in virtual cache
        from ..state.repository_cache import get_virtual_repository
        existing_virtual_repo = get_virtual_repository(git_url)
        if existing_virtual_repo:
            print(f"⚡ Using cached virtual repository: {owner}/{repo}")
            # Also extract to disk for compatibility
            if not target_dir.exists():
                _extract_virtual_repo_to_disk(existing_virtual_repo, target_dir)
            return True

        # GitHub ZIP download URL
        zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"

        print(f"📦 Downloading {owner}/{repo} (branch: {branch}) as ZIP...")

        # Download ZIP file to memory
        with urllib.request.urlopen(zip_url) as response:
            zip_data = response.read()

        print(f"✅ Downloaded ZIP file ({len(zip_data) / 1024 / 1024:.1f} MB)")

        # Create virtual repository in memory
        print("🧠 Creating virtual repository in memory...")
        virtual_repo = VirtualRepository(git_url, zip_data)
        cache_virtual_repository(git_url, virtual_repo)

        print(f"✅ Virtual repository ready: {virtual_repo.metadata['total_files']} files accessible instantly")

        # Also extract to disk for compatibility
        if not target_dir.exists():
            _extract_virtual_repo_to_disk(virtual_repo, target_dir)

        return True

    except Exception as e:
        print(f"❌ Error downloading repository to memory: {e}")
        return False


def _extract_virtual_repo_to_disk(virtual_repo: VirtualRepository, target_dir: Path):
    """Extract virtual repository to disk."""
    try:
        # Create parent directory if it doesn't exist
        target_dir.parent.mkdir(parents=True, exist_ok=True)

        # Extract ZIP file to disk
        with zipfile.ZipFile(BytesIO(virtual_repo.zip_data), 'r') as zip_ref:
            # First, extract to a temporary directory
            with tempfile.TemporaryDirectory() as temp_extract_dir:
                zip_ref.extractall(temp_extract_dir)

                # Find the extracted folder (should be repo-branch)
                extracted_folders = [f for f in os.listdir(temp_extract_dir)
                                   if os.path.isdir(os.path.join(temp_extract_dir, f))]

                if extracted_folders:
                    # Move the extracted content to target directory
                    extracted_folder = os.path.join(temp_extract_dir, extracted_folders[0])
                    shutil.move(extracted_folder, str(target_dir))
                    print(f"✅ Extracted virtual repository to disk: {target_dir}")

    except Exception as e:
        print(f"⚠️  Could not extract virtual repository to disk: {e}")


def get_github_download_url(git_url: str, branch: str = None) -> Optional[str]:
    """Get GitHub ZIP download URL."""
    github_info = parse_github_url(git_url)
    if not github_info:
        return None
        
    owner = github_info['owner']
    repo = github_info['repo']
    branch = branch or github_info['branch']
    
    return f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
