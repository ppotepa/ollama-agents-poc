"""Main repository analyzer that orchestrates all analysis components.

Single Responsibility: Coordinate different analyzers and build complete repository context.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..integrations.git_analyzer import GitAnalyzer
from ..models.repository_context import AnalysisConfig, RepositoryContext
from .directory_analyzer import DirectoryAnalyzer
from .file_analyzer import FileAnalyzer


class RepositoryAnalyzer:
    """Main repository analyzer that coordinates all analysis components.

    Responsibility: Orchestrate file, directory, and Git analysis to build complete context.
    """

    def __init__(self, root_path: str, config: AnalysisConfig | None = None):
        self.root_path = Path(root_path).resolve()
        self.config = config or AnalysisConfig.default()

        # Initialize component analyzers
        self.file_analyzer = FileAnalyzer(self.config)
        self.directory_analyzer = DirectoryAnalyzer(self.config)
        self.git_analyzer = GitAnalyzer() if self.config.include_git_info else None

    def analyze_repository(self) -> RepositoryContext:
        """Perform complete repository analysis.

        Returns:
            RepositoryContext with all analysis results
        """
        files = []
        directories = []

        # Walk through repository and analyze files/directories
        for root, dirs, filenames in os.walk(self.root_path):
            root_path = Path(root)

            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not self.directory_analyzer.should_ignore(root_path / d)]

            # Skip ignored root directories
            if self.directory_analyzer.should_ignore(root_path):
                continue

            # Analyze directory (skip root directory itself)
            if root_path != self.root_path:
                dir_info = self.directory_analyzer.analyze_directory(root_path, self.root_path)
                if dir_info:
                    directories.append(dir_info)

            # Analyze files in directory
            for filename in filenames:
                file_path = root_path / filename
                file_info = self.file_analyzer.analyze_file(file_path, self.root_path)
                if file_info:
                    files.append(file_info)

        # Calculate file statistics
        file_stats = self.file_analyzer.get_file_statistics(files)

        # Get Git information
        git_info = None
        if self.git_analyzer:
            git_info = self.git_analyzer.analyze_git_repository(self.root_path)

        # Build repository context
        return RepositoryContext(
            root_path=str(self.root_path),
            total_files=len(files),
            total_directories=len(directories),
            total_size=file_stats['total_size'],
            total_lines=file_stats['total_lines'],
            languages=file_stats['languages'],
            file_types=file_stats['file_types'],
            largest_files=file_stats['largest_files'],
            directory_structure=directories,
            files=files,
            git_info=git_info
        )

    def analyze_specific_path(self, relative_path: str) -> dict | None:
        """Analyze a specific file or directory within the repository.

        Args:
            relative_path: Path relative to repository root

        Returns:
            Analysis results for the specific path
        """
        target_path = self.root_path / relative_path

        if not target_path.exists():
            return None

        if target_path.is_file():
            file_info = self.file_analyzer.analyze_file(target_path, self.root_path)
            return {'type': 'file', 'info': file_info}

        elif target_path.is_dir():
            dir_info = self.directory_analyzer.analyze_directory(target_path, self.root_path)

            # Also get directory tree for this path
            tree = self.directory_analyzer.get_directory_tree(target_path, max_depth=2)

            return {
                'type': 'directory',
                'info': dir_info,
                'tree': tree
            }

        return None

    def get_files_by_language(self, language: str) -> list:
        """Get all files of a specific programming language.

        Args:
            language: Programming language name

        Returns:
            List of FileInfo objects for the specified language
        """
        context = self.analyze_repository()
        return [f for f in context.files if f.language == language]

    def get_files_by_extension(self, extension: str) -> list:
        """Get all files with a specific extension.

        Args:
            extension: File extension (e.g., '.py', '.js')

        Returns:
            List of FileInfo objects with the specified extension
        """
        context = self.analyze_repository()
        return [f for f in context.files if Path(f.path).suffix.lower() == extension.lower()]

    def search_files_by_pattern(self, pattern: str) -> list:
        """Search for files matching a pattern.

        Args:
            pattern: Glob pattern to match

        Returns:
            List of FileInfo objects matching the pattern
        """
        context = self.analyze_repository()
        matching_files = []

        for file_info in context.files:
            file_path = Path(file_info.path)
            if file_path.match(pattern):
                matching_files.append(file_info)

        return matching_files
