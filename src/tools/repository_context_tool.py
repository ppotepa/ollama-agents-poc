"""Repository context tool integration following SOLID principles.

Single Responsibility: Bridge between repository analysis package and agent tool system.
"""
from __future__ import annotations

from src.tools.repo_context import analyze_repo_structure, RepositoryAnalyzer, RepositoryContextFormatter

try:
    from langchain.tools import StructuredTool
    from src.tools.registry import register_tool
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False
    from src.tools.registry import register_tool


def analyze_repository_context(repository_path: str = ".") -> str:
    """Analyze repository structure and provide comprehensive context including file counts, languages, sizes, and structure."""
    return analyze_repo_structure(repository_path)


def analyze_repo_languages(repository_path: str = ".") -> str:
    """Analyze programming languages used in repository with detailed breakdown and percentages."""
    try:
        analyzer = RepositoryAnalyzer(repository_path)
        context = analyzer.analyze_repository()
        
        if not context.languages:
            return "No programming languages detected in repository."
        
        lines = ["💻 Programming Languages:"]
        
        # Sort by count, descending
        sorted_langs = sorted(
            context.language_percentages.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for lang, percentage in sorted_langs:
            count = context.languages[lang]
            lines.append(f"   • {lang}: {count} files ({percentage:.1f}%)")
        
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Error analyzing languages: {str(e)}"


def analyze_repo_directories(repository_path: str = ".", max_depth: int = 3) -> str:
    """Analyze repository directory structure with file counts and sizes."""
    try:
        analyzer = RepositoryAnalyzer(repository_path)
        context = analyzer.analyze_repository()
        
        if not context.directories:
            return "No directories found in repository."
        
        lines = ["📁 Directory Structure:"]
        
        # Sort directories by depth and name
        sorted_dirs = sorted(context.directories, key=lambda d: (d.path.count('/'), d.path))
        
        for dir_info in sorted_dirs:
            depth = dir_info.path.count('/')
            if depth <= max_depth:
                indent = "   " * (depth + 1)
                lines.append(f"{indent}📂 {dir_info.path.split('/')[-1]}/")
                lines.append(f"{indent}   • {dir_info.file_count} files, {dir_info.subdirectory_count} subdirs")
        
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Error analyzing directories: {str(e)}"


def get_repository_state() -> str:
    """Get current repository state as JSON for agent context management."""
    try:
        analyzer = RepositoryAnalyzer(".")
        context = analyzer.analyze_repository()
        return RepositoryContextFormatter.to_json(context)
    except Exception as e:
        return f"❌ Error getting repository state: {str(e)}"


# Register tools with the system
if LANGCHAIN_AVAILABLE:
    for fn in [analyze_repository_context, analyze_repo_languages, analyze_repo_directories, get_repository_state]:
        register_tool(StructuredTool.from_function(fn, name=fn.__name__, description=fn.__doc__ or fn.__name__))
else:
    class _Wrap:
        def __init__(self, fn):
            self._fn = fn
            self.name = fn.__name__
            self.description = fn.__doc__ or fn.__name__
        def __call__(self, *a, **kw):
            return self._fn(*a, **kw)
    for _f in [analyze_repository_context, analyze_repo_languages, analyze_repo_directories, get_repository_state]:
        register_tool(_Wrap(_f))


__all__ = [
    "analyze_repository_context",
    "analyze_repo_languages", 
    "analyze_repo_directories",
    "get_repository_state"
]
