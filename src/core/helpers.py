"""Repository validation for coding agents."""

import os
import subprocess
import hashlib
import urllib.request
import zipfile
import shutil
import tempfile
import re
import json
import requests
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from io import BytesIO
from .enums import AgentCapability, Domain

# Global ZIP cache and virtual directory storage
_zip_cache = {}
_virtual_directories = {}


class VirtualRepository:
    """In-memory representation of a repository extracted from ZIP."""
    
    def __init__(self, repo_url: str, zip_data: bytes):
        self.repo_url = repo_url
        self.zip_data = zip_data
        self.files: Dict[str, bytes] = {}
        self.directories: Dict[str, List[str]] = {}
        self.metadata: Dict[str, Any] = {}
        self._extract_to_memory()
    
    def _extract_to_memory(self):
        """Extract ZIP contents to memory structures."""
        try:
            with zipfile.ZipFile(BytesIO(self.zip_data), 'r') as zip_ref:
                # Get all file and directory paths
                all_paths = zip_ref.namelist()
                
                # Process each path
                for path in all_paths:
                    if path.endswith('/'):
                        # It's a directory
                        dir_path = path.rstrip('/')
                        self.directories[dir_path] = []
                    else:
                        # It's a file
                        try:
                            file_content = zip_ref.read(path)
                            self.files[path] = file_content
                            
                            # Add to parent directory listing
                            parent_dir = str(Path(path).parent)
                            if parent_dir == '.':
                                parent_dir = ''
                            
                            if parent_dir not in self.directories:
                                self.directories[parent_dir] = []
                            
                            filename = Path(path).name
                            if filename not in self.directories[parent_dir]:
                                self.directories[parent_dir].append(filename)
                                
                        except Exception as e:
                            print(f"⚠️  Could not read file {path}: {e}")
                
                # Store metadata
                self.metadata = {
                    'total_files': len(self.files),
                    'total_directories': len(self.directories),
                    'repo_url': self.repo_url,
                    'zip_size': len(self.zip_data)
                }
                
        except Exception as e:
            print(f"❌ Error extracting ZIP to memory: {e}")
            raise
    
    def get_file_content(self, path: str) -> Optional[bytes]:
        """Get file content from memory."""
        return self.files.get(path)
    
    def get_file_content_text(self, path: str, encoding: str = 'utf-8') -> Optional[str]:
        """Get file content as text."""
        content = self.get_file_content(path)
        if content is None:
            return None
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            return None
    
    def list_directory(self, path: str = '') -> List[str]:
        """List contents of a directory."""
        return self.directories.get(path, [])
    
    def file_exists(self, path: str) -> bool:
        """Check if file exists in virtual repository."""
        return path in self.files
    
    def directory_exists(self, path: str) -> bool:
        """Check if directory exists in virtual repository."""
        return path in self.directories
    
    def get_all_files(self) -> List[str]:
        """Get list of all file paths."""
        return list(self.files.keys())
    
    def get_all_directories(self) -> List[str]:
        """Get list of all directory paths."""
        return list(self.directories.keys())


# Global cache for virtual repositories and persistent storage
_virtual_repo_cache: Dict[str, VirtualRepository] = {}
_zip_cache_dir = Path("./data/zip_cache")
_zip_cache_dir.mkdir(exist_ok=True)


def get_virtual_repository(repo_url: str) -> Optional[VirtualRepository]:
    """Get cached virtual repository, loading from disk if needed."""
    global _virtual_repo_cache
    
    # First check in-memory cache
    if repo_url in _virtual_repo_cache:
        return _virtual_repo_cache[repo_url]
    
    # Try loading from disk cache
    github_info = parse_github_url(repo_url)
    if github_info:
        owner = github_info['owner']
        repo = github_info['repo']
        branch = github_info['branch']
        cache_key = f"{owner}_{repo}_{branch}"
        
        zip_file_path = _zip_cache_dir / f"{cache_key}.zip"
        if zip_file_path.exists():
            try:
                with open(zip_file_path, 'rb') as f:
                    zip_data = f.read()
                
                # Create virtual repository from cached ZIP
                virtual_repo = VirtualRepository(repo_url, zip_data)
                _virtual_repo_cache[repo_url] = virtual_repo
                
                return virtual_repo
            except Exception as e:
                print(f"⚠️  Could not load cached ZIP {zip_file_path}: {e}")
    
    return None


def cache_virtual_repository(repo_url: str, virtual_repo: VirtualRepository):
    """Cache virtual repository both in memory and on disk."""
    global _virtual_repo_cache
    
    # Cache in memory
    _virtual_repo_cache[repo_url] = virtual_repo
    
    # Cache ZIP data to disk for persistence
    github_info = parse_github_url(repo_url)
    if github_info:
        owner = github_info['owner']
        repo = github_info['repo']
        branch = github_info['branch']
        cache_key = f"{owner}_{repo}_{branch}"
        
        zip_file_path = _zip_cache_dir / f"{cache_key}.zip"
        try:
            with open(zip_file_path, 'wb') as f:
                f.write(virtual_repo.zip_data)
            print(f"💾 ZIP cached to disk: {zip_file_path}")
        except Exception as e:
            print(f"⚠️  Could not cache ZIP to disk: {e}")

class VirtualDirectory:
    """In-memory representation of a ZIP-based repository for fast access."""
    
    def __init__(self, repo_url: str, zip_data: bytes):
        self.repo_url = repo_url
        self.zip_data = zip_data
        self.files = {}  # path -> file_info dict
        self.directories = set()
        self._build_virtual_structure()
    
    def _build_virtual_structure(self):
        """Build the virtual directory structure from ZIP data."""
        with zipfile.ZipFile(BytesIO(self.zip_data), 'r') as zip_ref:
            for info in zip_ref.infolist():
                # Skip directories and extract file info
                if not info.is_dir():
                    # Remove the top-level directory from path (e.g., "vscode-main/")
                    path_parts = info.filename.split('/', 1)
                    if len(path_parts) > 1:
                        clean_path = path_parts[1]
                    else:
                        clean_path = info.filename
                    
                    # Store file information
                    self.files[clean_path] = {
                        'size': info.file_size,
                        'compressed_size': info.compress_size,
                        'modified': info.date_time,
                        'zip_info': info
                    }
                    
                    # Track directories
                    dir_path = os.path.dirname(clean_path)
                    while dir_path and dir_path != '.':
                        self.directories.add(dir_path)
                        dir_path = os.path.dirname(dir_path)
    
    def get_file_content(self, file_path: str) -> Optional[bytes]:
        """Get file content directly from ZIP."""
        if file_path not in self.files:
            return None
        
        try:
            with zipfile.ZipFile(BytesIO(self.zip_data), 'r') as zip_ref:
                zip_info = self.files[file_path]['zip_info']
                return zip_ref.read(zip_info)
        except Exception:
            return None
    
    def list_files(self, directory: str = "") -> List[str]:
        """List files in a directory."""
        if directory and not directory.endswith('/'):
            directory += '/'
        
        files = []
        for path in self.files.keys():
            if directory == "" or path.startswith(directory):
                # Get relative path from directory
                relative_path = path[len(directory):] if directory else path
                # Only include direct children (no subdirectories)
                if '/' not in relative_path:
                    files.append(relative_path)
        return files
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the virtual directory for context."""
        total_files = len(self.files)
        total_dirs = len(self.directories)
        total_size = sum(info['size'] for info in self.files.values())
        
        # Get file extensions
        extensions = {}
        for path in self.files.keys():
            ext = os.path.splitext(path)[1].lower()
            if ext:
                extensions[ext] = extensions.get(ext, 0) + 1
        
        # Get top directories by file count
        dir_counts = {}
        for path in self.files.keys():
            dir_path = os.path.dirname(path)
            if dir_path:
                top_dir = dir_path.split('/')[0]
                dir_counts[top_dir] = dir_counts.get(top_dir, 0) + 1
        
        return {
            'repo_url': self.repo_url,
            'total_files': total_files,
            'total_directories': total_dirs,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'file_extensions': dict(sorted(extensions.items(), key=lambda x: x[1], reverse=True)[:10]),
            'top_directories': dict(sorted(dir_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        }


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
    global _zip_cache, _virtual_directories
    
    try:
        github_info = parse_github_url(git_url)
        if not github_info:
            return False  # Not a GitHub URL, fall back to git clone
        
        owner = github_info['owner']
        repo = github_info['repo']
        branch = github_info['branch']
        
        # Create cache key for this repository
        cache_key = f"{owner}/{repo}@{branch}"
        
        # Check if we have this repository cached
        zip_data = None
        if cache_key in _zip_cache:
            print(f"💾 Using cached ZIP for {cache_key}")
            zip_data = _zip_cache[cache_key]
        else:
            # GitHub ZIP download URL
            zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
            
            print(f"📦 Downloading {owner}/{repo} (branch: {branch}) as ZIP...")
            
            # Download ZIP data to memory
            try:
                response = urllib.request.urlopen(zip_url)
                zip_data = response.read()
                
                # Cache the ZIP data
                _zip_cache[cache_key] = zip_data
                print(f"✅ Downloaded and cached ZIP file ({len(zip_data)} bytes)")
                
            except urllib.error.HTTPError as e:
                if e.code == 404 and branch == 'main':
                    # Try with 'master' branch
                    print(f"� Trying with 'master' branch...")
                    master_zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/master.zip"
                    
                    try:
                        response = urllib.request.urlopen(master_zip_url)
                        zip_data = response.read()
                        
                        # Update cache key for master branch
                        cache_key = f"{owner}/{repo}@master"
                        _zip_cache[cache_key] = zip_data
                        print(f"✅ Downloaded and cached ZIP file (master branch)")
                        
                    except Exception:
                        print(f"❌ Could not download repository with master branch either")
                        return False
                else:
                    print(f"❌ Repository not found or access denied (HTTP {e.code})")
                    return False
        
        # Create virtual directory from ZIP data
        virtual_dir = VirtualDirectory(git_url, zip_data)
        _virtual_directories[cache_key] = virtual_dir
        print(f"🧠 Created virtual directory: {virtual_dir.get_context_summary()['total_files']} files, "
              f"{virtual_dir.get_context_summary()['total_directories']} directories")
        
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
                    print(f"❌ No folders found in ZIP file")
                    return False
                
                # Move the extracted content to target directory
                extracted_folder = os.path.join(temp_extract_dir, extracted_folders[0])
                shutil.move(extracted_folder, str(target_dir))
                
                print(f"✅ Successfully extracted repository to {target_dir}")
                
                # Load repository context into memory for fast access
                try:
                    from src.tools.context import load_repository_context_after_clone
                    if load_repository_context_after_clone(str(target_dir), cache_content=True, quiet=True):
                        print(f"🔍 Starting repository scan...")
                        print(f"🧠 Repository context loaded into memory")
                except ImportError:
                    print(f"⚠️  Repository context system not available")
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
        print(f"🧠 Creating virtual repository in memory...")
        virtual_repo = VirtualRepository(git_url, zip_data)
        
        # Cache the virtual repository
        cache_virtual_repository(git_url, virtual_repo)
        
        print(f"✅ Virtual repository created: {virtual_repo.metadata['total_files']} files, {virtual_repo.metadata['total_directories']} directories")
        
        # Also extract to disk for compatibility with existing tools
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if target_dir.exists():
            print(f"🗑️  Removing existing directory: {target_dir}")
            shutil.rmtree(target_dir)
        
        _extract_virtual_repo_to_disk(virtual_repo, target_dir)
        print(f"✅ Successfully extracted repository to {target_dir}")
        
        # Load repository context into memory for fast access
        try:
            from src.tools.context import load_repository_context_after_clone
            if load_repository_context_after_clone(str(target_dir), cache_content=True, quiet=True):
                print(f"🧠 Repository context loaded into memory")
        except ImportError:
            print(f"⚠️  Repository context system not available")
        except Exception as e:
            print(f"⚠️  Could not load repository context: {e}")
        
        return True
                    
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"❌ Repository not found or branch '{branch}' doesn't exist")
            # Try with 'master' branch if 'main' failed
            if branch == 'main':
                print(f"🔄 Trying with 'master' branch...")
                
                # Try downloading with master branch
                master_zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/master.zip"
                
                try:
                    with urllib.request.urlopen(master_zip_url) as response:
                        zip_data = response.read()
                    
                    print(f"✅ Downloaded ZIP file (master branch, {len(zip_data) / 1024 / 1024:.1f} MB)")
                    
                    # Create virtual repository with master branch
                    virtual_repo = VirtualRepository(git_url, zip_data)
                    cache_key_master = f"{owner}/{repo}:master"
                    cache_virtual_repository(cache_key_master, virtual_repo)
                    
                    print(f"✅ Virtual repository created: {virtual_repo.metadata['total_files']} files, {virtual_repo.metadata['total_directories']} directories")
                    
                    # Extract to disk for compatibility
                    target_dir.parent.mkdir(parents=True, exist_ok=True)
                    if target_dir.exists():
                        shutil.rmtree(target_dir)
                    
                    _extract_virtual_repo_to_disk(virtual_repo, target_dir)
                    print(f"✅ Successfully extracted repository to {target_dir}")
                    
                    return True
                    
                except Exception as master_e:
                    print(f"❌ Master branch download also failed: {master_e}")
                    return False
        else:
            print(f"❌ HTTP error downloading repository: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading repository: {e}")
        return False


def _extract_virtual_repo_to_disk(virtual_repo: VirtualRepository, target_dir: Path):
    """Extract virtual repository to disk for compatibility."""
    try:
        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract all files
        for file_path, content in virtual_repo.files.items():
            # Remove the top-level directory from GitHub ZIP structure
            path_parts = Path(file_path).parts
            if len(path_parts) > 1:
                # Skip the first part (repo-branch folder)
                relative_path = Path(*path_parts[1:])
            else:
                relative_path = Path(file_path)
            
            full_path = target_dir / relative_path
            
            # Create parent directory if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file content
            try:
                with open(full_path, 'wb') as f:
                    f.write(content)
            except Exception as e:
                print(f"⚠️  Could not write file {relative_path}: {e}")
                
    except Exception as e:
        print(f"❌ Error extracting virtual repository to disk: {e}")


def get_virtual_directory(git_url: str) -> Optional['VirtualDirectory']:
    """Get the virtual directory for a GitHub repository if it exists."""
    global _virtual_directories
    
    github_info = parse_github_url(git_url)
    if not github_info:
        return None
    
    # Try both main and master branch keys
    cache_key_main = f"{github_info['owner']}/{github_info['repo']}@{github_info['branch']}"
    cache_key_master = f"{github_info['owner']}/{github_info['repo']}@master"
    
    return _virtual_directories.get(cache_key_main) or _virtual_directories.get(cache_key_master)


def get_cached_repositories() -> Dict[str, Dict[str, Any]]:
    """Get information about all cached repositories."""
    global _virtual_directories
    
    result = {}
    for cache_key, virtual_dir in _virtual_directories.items():
        result[cache_key] = virtual_dir.get_context_summary()
    
    return result


def clear_repository_cache():
    """Clear all cached ZIP files and virtual directories."""
    global _zip_cache, _virtual_directories
    
    _zip_cache.clear()
    _virtual_directories.clear()
    print("🧹 Repository cache cleared")


def _ensure_virtual_repository(git_url: str) -> bool:
    """Ensure a virtual repository is available for the given URL."""
    try:
        # Check if virtual repository already exists
        virtual_repo = get_virtual_repository(git_url)
        if virtual_repo:
            print(f"⚡ Virtual repository already cached for: {git_url}")
            return True
        
        # Try to download and create virtual repository
        github_info = parse_github_url(git_url)
        if not github_info:
            print(f"⚠️  Not a GitHub URL, cannot create virtual repository: {git_url}")
            return False
        
        owner = github_info['owner']
        repo = github_info['repo'] 
        branch = github_info['branch']
        
        print(f"⚡ Creating virtual repository for: {owner}/{repo}@{branch}")
        
        # Download ZIP to memory and create virtual repository
        zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
        
        with urllib.request.urlopen(zip_url) as response:
            zip_data = response.read()
        
        print(f"✅ Downloaded ZIP ({len(zip_data) / 1024 / 1024:.1f} MB)")
        
        # Create and cache virtual repository
        virtual_repo = VirtualRepository(git_url, zip_data)
        cache_virtual_repository(git_url, virtual_repo)
        
        print(f"✅ Virtual repository created: {virtual_repo.metadata['total_files']} files")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to ensure virtual repository: {e}")
        return False


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
    For GitHub repositories, uses ZIP download for faster performance.
    
    Args:
        git_url: URL of the git repository to clone
        target_dir: Directory where the repository should be cloned
        
    Returns:
        True if cloning was successful, False otherwise
    """
    try:
        # Try GitHub ZIP download first (much faster)
        if parse_github_url(git_url):
            print(f"🚀 Detected GitHub repository, using fast ZIP download...")
            if download_github_zip_to_memory(git_url, target_dir):
                # Set up virtual directory for context
                try:
                    from src.tools.context import set_virtual_directory_for_context
                    set_virtual_directory_for_context(git_url)
                except ImportError:
                    print(f"⚠️  Virtual directory context not available")
                return True
            else:
                print(f"⚠️  ZIP download failed, falling back to git clone...")
        
        # Fall back to git clone for non-GitHub URLs or if ZIP download failed
        # Create parent directory if it doesn't exist
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # If directory already exists and is a git repo, just pull
        if target_dir.exists() and is_git_repository(str(target_dir)):
            print(f"Repository already exists at {target_dir}, pulling latest changes...")
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
                        print(f"🧠 Repository context refreshed in memory")
                except Exception as e:
                    print(f"⚠️  Could not refresh repository context: {e}")
            
            return result.returncode == 0
        
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
                    print(f"🧠 Repository context loaded into memory")
            except Exception as e:
                print(f"⚠️  Could not load repository context: {e}")
            
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
            text=True,
            encoding='utf-8',
            errors='replace'
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


def validate_repository_requirement(agent_name: str, path: str = ".", git_url: Optional[str] = None, data_path: Optional[str] = None, optional: bool = False, interception_mode: str = "smart") -> Tuple[bool, Optional[str]]:
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
                    pass
            
            # Normal cloning for full mode or fallback
            clone_dir = get_clone_directory(git_url, data_path)
            if clone_repository(git_url, clone_dir):
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


def get_agent_instance(agent_name: str, streaming: bool = True, with_tools: bool = True):
    """
    Get the appropriate agent instance based on the agent name.
    
    Args:
        agent_name: The name of the agent to create
        streaming: Whether to enable streaming responses
        with_tools: Whether to enable tools for the agent
    """
    try:
        # Use the new DynamicAgentFactory for creating agents
        from src.core.agent_factory import get_agent_factory
        
        factory = get_agent_factory(streaming=streaming)
        agent = factory.get_or_create_agent(agent_name)
        
        if agent:
            print(f"✅ Created agent using DynamicAgentFactory: {agent_name} (streaming: {streaming})")
            
            # If tools are explicitly disabled, update the agent
            if not with_tools and hasattr(agent, '_tools'):
                agent._tools = []
                print(f"🛑 Tools explicitly disabled for {agent_name}")
            
            return agent
        
        # Fallback to legacy system if DynamicAgentFactory fails
        print(f"⚠️ DynamicAgentFactory failed, falling back to legacy system")
        
    except Exception as e:
        print(f"⚠️ Error with DynamicAgentFactory: {e}")
        # Continue to legacy fallback
    
    # Legacy fallback system
    import sys
    try:
        # Add parent directory to path for imports
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        
        from integrations.model_config_reader import ModelConfigReader
        
        # Load the model configuration to get agent details
        config_reader = ModelConfigReader('src/config/models.yaml')
        model_config = config_reader.get_model(agent_name)
        
        if not model_config:
            # Try to find by model ID or partial match
            for model in config_reader.get_all_models():
                if (agent_name in model.model_id or 
                    model.model_id.startswith(agent_name) or
                    model.model_id == agent_name or  # Exact model ID match
                    agent_name in model.short_name):
                    model_config = model
                    break
        
        if not model_config:
            raise ValueError(f"No configuration found for agent '{agent_name}'")
        
        # Convert model config to agent config format
        agent_config = {
            "name": model_config.name,
            "model_id": model_config.model_id,
            "backend_image": model_config.model_id,
            "parameters": model_config.parameters,
            "tools": model_config.tools,
            "system_message": model_config.system_message,
            "supports_coding": model_config.supports_coding
        }
        
        # Use UniversalAgent for all agent types
        try:
            from src.agents.universal.agent import create_universal_agent
            return create_universal_agent(agent_name, agent_config, model_config.model_id, streaming)
        except ImportError:
            print(f"⚠️  Warning: UniversalAgent not available, falling back to simple agent")
            # Import SimpleQueryAgent from single_query_mode
            from .single_query_mode import SimpleQueryAgent
            return SimpleQueryAgent(agent_name, agent_config)
            
    except Exception as e:
        print(f"⚠️  Warning: Could not load proper agent, using fallback: {e}")
        # Create a basic fallback config with proper model mapping
        model_mapping = {
            'phi3_mini': 'phi3:mini',
            'deepcoder': 'deepcoder:14b',
            'deepcoder:14b': 'deepcoder:14b',  # Allow direct model ID
            'qwen2.5-coder:7b': 'qwen2.5-coder:7b',
            'qwen2.5:7b-instruct-q4_K_M': 'qwen2.5:7b-instruct-q4_K_M',
            'qwen2.5:3b-instruct-q4_K_M': 'qwen2.5:3b-instruct-q4_K_M',
            'gemma:7b-instruct-q4_K_M': 'gemma:7b-instruct-q4_K_M',
            'codellama:13b-instruct': 'codellama:13b-instruct',
            'mistral:7b-instruct': 'mistral:7b-instruct',
            'assistant': 'qwen2.5:3b-instruct-q4_K_M',  # Use available model
            'coder': 'qwen2.5-coder:7b'
        }
        
        fallback_config = {
            'name': agent_name.title(),
            'model_id': model_mapping.get(agent_name, agent_name),
            'backend_image': model_mapping.get(agent_name, agent_name),
            'parameters': {'temperature': 0.7, 'num_ctx': 8192},
            'tools': [],
            'system_message': "You are an AI assistant.",
            'supports_coding': True
        }
        
        # Try UniversalAgent first, then fallback to SimpleQueryAgent
        try:
            from src.agents.universal.agent import create_universal_agent
            return create_universal_agent(agent_name, fallback_config, fallback_config['model_id'], streaming)
        except ImportError:
            from .single_query_mode import SimpleQueryAgent
            return SimpleQueryAgent(agent_name, fallback_config)
