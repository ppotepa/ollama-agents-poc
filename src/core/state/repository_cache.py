"""Repository caching system for virtual repositories."""

from pathlib import Path
from typing import Optional, Dict, Any

from .virtual_repository import VirtualRepository


class RepositoryCache:
    """Cache manager for virtual repositories."""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path("./data/zip_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self._memory_cache: dict[str, VirtualRepository] = {}
        
    def get(self, repo_url: str) -> Optional[VirtualRepository]:
        """Get cached virtual repository, loading from disk if needed."""
        # First check in-memory cache
        if repo_url in self._memory_cache:
            return self._memory_cache[repo_url]

        # Try loading from disk cache
        cache_key = self._get_cache_key(repo_url)
        if cache_key:
            zip_file_path = self.cache_dir / f"{cache_key}.zip"
            if zip_file_path.exists():
                try:
                    with open(zip_file_path, 'rb') as f:
                        zip_data = f.read()

                    # Create virtual repository from cached ZIP
                    virtual_repo = VirtualRepository(repo_url, zip_data)
                    self._memory_cache[repo_url] = virtual_repo

                    return virtual_repo
                except Exception as e:
                    print(f"⚠️  Could not load cached ZIP {zip_file_path}: {e}")

        return None

    def set(self, repo_url: str, virtual_repo: VirtualRepository):
        """Cache virtual repository both in memory and on disk."""
        # Cache in memory
        self._memory_cache[repo_url] = virtual_repo

        # Cache ZIP data to disk for persistence
        cache_key = self._get_cache_key(repo_url)
        if cache_key:
            zip_file_path = self.cache_dir / f"{cache_key}.zip"
            try:
                with open(zip_file_path, 'wb') as f:
                    f.write(virtual_repo.zip_data)
                print(f"💾 ZIP cached to disk: {zip_file_path}")
            except Exception as e:
                print(f"⚠️  Could not cache ZIP to disk: {e}")

    def _get_cache_key(self, repo_url: str) -> Optional[str]:
        """Generate cache key from repository URL."""
        from ..repository.url_validator import parse_github_url
        
        github_info = parse_github_url(repo_url)
        if github_info:
            owner = github_info['owner']
            repo = github_info['repo']
            branch = github_info['branch']
            return f"{owner}_{repo}_{branch}"
        
        return None

    def clear_memory(self):
        """Clear in-memory cache."""
        self._memory_cache.clear()

    def clear_disk(self):
        """Clear disk cache."""
        for cache_file in self.cache_dir.glob("*.zip"):
            try:
                cache_file.unlink()
            except Exception as e:
                print(f"⚠️  Could not remove cache file {cache_file}: {e}")

    def clear_all(self):
        """Clear both memory and disk cache."""
        self.clear_memory()
        self.clear_disk()


# Global cache instance
_global_cache = None


def get_repository_cache() -> RepositoryCache:
    """Get the global repository cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = RepositoryCache()
    return _global_cache


def get_virtual_repository(repo_url: str) -> Optional[VirtualRepository]:
    """Get cached virtual repository."""
    return get_repository_cache().get(repo_url)


def cache_virtual_repository(repo_url: str, virtual_repo: VirtualRepository):
    """Cache virtual repository."""
    get_repository_cache().set(repo_url, virtual_repo)


def get_cached_repositories() -> Dict[str, Dict[str, Any]]:
    """Get information about all cached repositories."""
    cache = get_repository_cache()
    result = {}
    
    # Get from memory cache
    for cache_key, virtual_repo in cache._memory_cache.items():
        if hasattr(virtual_repo, 'get_context_summary'):
            result[cache_key] = virtual_repo.get_context_summary()
        else:
            result[cache_key] = {"type": "VirtualRepository", "files": len(virtual_repo.files)}
    
    return result


def clear_repository_cache():
    """Clear all cached ZIP files and virtual directories."""
    cache = get_repository_cache()
    cache.clear_all()
    print("🧹 Repository cache cleared")


def get_virtual_directory(git_url: str) -> Optional['VirtualDirectory']:
    """Get the virtual directory for a GitHub repository if it exists."""
    # For backwards compatibility - try to get VirtualRepository and check if it has directory structure
    virtual_repo = get_virtual_repository(git_url)
    if virtual_repo and hasattr(virtual_repo, 'get_directory_structure'):
        # This would need to be implemented if VirtualDirectory is different from VirtualRepository
        # For now, return None since VirtualDirectory might be a separate concept
        pass
    return None
