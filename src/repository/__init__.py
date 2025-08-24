"""Repository analysis package following SOLID principles.

This package provides modular repository analysis capabilities with clear separation of concerns:
- models: Data structures and types
- analyzers: Core analysis logic split by responsibility  
- formatters: Output formatting and presentation
- integrations: External system integrations (git, tools)
"""

__version__ = "1.0.0"
__all__ = [
    "RepositoryAnalyzer",
    "RepositoryContext", 
    "FileInfo",
    "DirectoryInfo",
    "AnalysisConfig",
    "analyze_repository_context",
    "analyze_repository_detailed",
    "get_language_breakdown",
    "get_directory_structure"
]

from .analyzers.repository_analyzer import RepositoryAnalyzer
from .models.repository_context import RepositoryContext, FileInfo, DirectoryInfo, AnalysisConfig
from .main import analyze_repository_context, analyze_repository_detailed, get_language_breakdown, get_directory_structure
