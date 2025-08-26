"""Data models for repository context analysis.

Single Responsibility: Define data structures and types for repository analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional


@dataclass
class FileInfo:
    """Information about a single file in the repository.
    
    Responsibility: Hold file metadata and properties.
    """
    path: str
    size: int
    lines: int
    language: Optional[str]
    mime_type: Optional[str]
    is_binary: bool
    last_modified: float
    
    
@dataclass
class DirectoryInfo:
    """Information about a directory in the repository.
    
    Responsibility: Hold directory metadata and statistics.
    """
    path: str
    file_count: int
    subdirectory_count: int
    total_size: int
    

@dataclass
class RepositoryContext:
    """Complete repository context information.
    
    Responsibility: Aggregate all repository analysis results.
    """
    root_path: str
    total_files: int
    total_directories: int
    total_size: int
    total_lines: int
    languages: Dict[str, int]  # language -> file count
    file_types: Dict[str, int]  # extension -> file count
    largest_files: List[FileInfo]
    directory_structure: List[DirectoryInfo]
    files: List[FileInfo]
    git_info: Optional[Dict[str, Any]] = None
    
    
@dataclass
class AnalysisConfig:
    """Configuration for repository analysis.
    
    Responsibility: Hold analysis configuration and options.
    """
    ignore_patterns: set[str]
    max_largest_files: int = 10
    include_git_info: bool = True
    include_binary_files: bool = True
    max_file_size_mb: Optional[int] = None  # Skip files larger than this
    
    @classmethod
    def default(cls) -> 'AnalysisConfig':
        """Create default analysis configuration."""
        return cls(
            ignore_patterns={
                '__pycache__', '.git', '.venv', 'venv', 'node_modules', 
                '.pytest_cache', '.mypy_cache', 'dist', 'build',
                '.tox', 'htmlcov', '.coverage', '*.pyc', '*.pyo'
            }
        )
