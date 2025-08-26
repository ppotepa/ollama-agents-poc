"""Repository cache management for virtual repositories."""

from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .virtual_repo import VirtualDirectory

from .virtual_repo import VirtualRepository
from .url_validator import parse_github_url

# Global caches
_virtual_repo_cache: Dict[str, VirtualRepository] = {}
_virtual_directories = {}
_zip_cache = {}

# Cache directory setup
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


def get_virtual_directory(git_url: str) -> Optional['VirtualDirectory']:
    """Get cached virtual directory."""
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
    global _zip_cache, _virtual_directories, _virtual_repo_cache

    _zip_cache.clear()
    _virtual_directories.clear()
    _virtual_repo_cache.clear()
    print("🧹 Repository cache cleared")


def ensure_virtual_repository(git_url: str) -> bool:
    """Ensure a virtual repository is available for the given URL."""
    try:
        # Check if virtual repository already exists
        virtual_repo = get_virtual_repository(git_url)
        if virtual_repo:
            print(f"⚡ Virtual repository already cached for: {git_url}")
            return True

        # Try to download and create virtual repository
        # This would need to be implemented to create a virtual repo
        # For now, return False to indicate it needs implementation
        print(f"📝 Virtual repository creation needed for: {git_url}")
        return False

    except Exception as e:
        print(f"❌ Error ensuring virtual repository for {git_url}: {e}")
        return False
