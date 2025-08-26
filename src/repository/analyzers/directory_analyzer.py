"""Directory analysis service for repository scanning.

Single Responsibility: Analyze directory structure and organization.
"""
from __future__ import annotations

from pathlib import Path

from ..models.repository_context import AnalysisConfig, DirectoryInfo


class DirectoryAnalyzer:
    """Analyzes directory structure and organization.

    Responsibility: Extract directory information and calculate directory statistics.
    """

    def __init__(self, config: AnalysisConfig):
        self.config = config

    def should_ignore(self, path: Path) -> bool:
        """Check if a directory should be ignored based on configuration.

        Args:
            path: Directory path to check

        Returns:
            True if directory should be ignored
        """
        # Check each part of the path
        for part in path.parts:
            if part in self.config.ignore_patterns:
                return True

        # Check glob patterns
        return any('*' in pattern and path.match(pattern) for pattern in self.config.ignore_patterns)

    def analyze_directory(self, dir_path: Path, root_path: Path) -> DirectoryInfo | None:
        """Analyze a single directory and extract metadata.

        Args:
            dir_path: Path to the directory to analyze
            root_path: Repository root path for relative path calculation

        Returns:
            DirectoryInfo object or None if directory should be skipped
        """
        try:
            # Check if directory should be ignored
            if self.should_ignore(dir_path):
                return None

            file_count = 0
            subdirectory_count = 0
            total_size = 0

            # Iterate through directory contents
            for item in dir_path.iterdir():
                if self.should_ignore(item):
                    continue

                if item.is_file():
                    file_count += 1
                    try:
                        total_size += item.stat().st_size
                    except Exception:
                        pass  # Skip if can't get size
                elif item.is_dir():
                    subdirectory_count += 1

            return DirectoryInfo(
                path=str(dir_path.relative_to(root_path)),
                file_count=file_count,
                subdirectory_count=subdirectory_count,
                total_size=total_size
            )

        except Exception:
            return None  # Skip problematic directories

    def get_directory_depth(self, dir_path: Path, root_path: Path) -> int:
        """Calculate the depth of a directory relative to root.

        Args:
            dir_path: Directory path
            root_path: Repository root path

        Returns:
            Directory depth (0 for root)
        """
        try:
            relative_path = dir_path.relative_to(root_path)
            return len(relative_path.parts)
        except ValueError:
            return 0

    def get_directory_tree(self, root_path: Path, max_depth: int | None = None) -> dict:
        """Build a hierarchical tree structure of directories.

        Args:
            root_path: Repository root path
            max_depth: Maximum depth to traverse (None for unlimited)

        Returns:
            Nested dictionary representing directory tree
        """
        tree = {}

        try:
            for item in root_path.iterdir():
                if self.should_ignore(item):
                    continue

                if item.is_dir():
                    depth = self.get_directory_depth(item, root_path)

                    # Check max depth
                    if max_depth is not None and depth >= max_depth:
                        continue

                    # Recursively build subtree
                    subtree = self.get_directory_tree(item, max_depth)
                    tree[item.name] = {
                        'type': 'directory',
                        'path': str(item.relative_to(root_path)),
                        'children': subtree
                    }
                elif item.is_file():
                    tree[item.name] = {
                        'type': 'file',
                        'path': str(item.relative_to(root_path))
                    }
        except Exception:
            pass

        return tree

    def calculate_directory_statistics(self, directories: list[DirectoryInfo]) -> dict:
        """Calculate statistics from directory information.

        Args:
            directories: List of DirectoryInfo objects

        Returns:
            Dictionary with directory statistics
        """
        total_dirs = len(directories)
        total_files_in_dirs = sum(d.file_count for d in directories)
        total_subdirs = sum(d.subdirectory_count for d in directories)
        total_size_in_dirs = sum(d.total_size for d in directories)

        # Find largest directories by file count
        largest_by_files = sorted(directories, key=lambda d: d.file_count, reverse=True)[:10]

        # Find largest directories by size
        largest_by_size = sorted(directories, key=lambda d: d.total_size, reverse=True)[:10]

        # Calculate average files per directory
        avg_files_per_dir = total_files_in_dirs / total_dirs if total_dirs > 0 else 0

        return {
            'total_directories': total_dirs,
            'total_files_in_dirs': total_files_in_dirs,
            'total_subdirectories': total_subdirs,
            'total_size_in_dirs': total_size_in_dirs,
            'largest_by_files': largest_by_files,
            'largest_by_size': largest_by_size,
            'avg_files_per_directory': avg_files_per_dir,
        }
