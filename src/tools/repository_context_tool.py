"""Repository context tool integration following SOLID principles.

Single Responsibility: Bridge between repository analysis package and agent tool system.
"""
from __future__ import annotations

from src.repository import analyze_repository_context, get_language_breakdown, get_directory_structure

try:
    from langchain.tools import StructuredTool
    from src.tools.registry import register_tool
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False
    from src.tools.registry import register_tool


def analyze_repo_structure(repository_path: str = ".") -> str:
    """Analyze repository structure and provide comprehensive context including file counts, languages, sizes, and structure."""
    return analyze_repository_context(repository_path)


def analyze_repo_languages(repository_path: str = ".") -> str:
    """Analyze programming languages used in repository with detailed breakdown and percentages."""
    return get_language_breakdown(repository_path)


def analyze_repo_directories(repository_path: str = ".", max_depth: int = 3) -> str:
    """Analyze repository directory structure with file counts and sizes."""
    return get_directory_structure(repository_path, max_depth)


# Register tools with the system
if LANGCHAIN_AVAILABLE:
    for fn in [analyze_repo_structure, analyze_repo_languages, analyze_repo_directories]:
        register_tool(StructuredTool.from_function(fn, name=fn.__name__, description=fn.__doc__ or fn.__name__))
else:
    class _Wrap:
        def __init__(self, fn):
            self._fn = fn
            self.name = fn.__name__
            self.description = fn.__doc__ or fn.__name__
        def __call__(self, *a, **kw):
            return self._fn(*a, **kw)
    for _f in [analyze_repo_structure, analyze_repo_languages, analyze_repo_directories]:
        register_tool(_Wrap(_f))


__all__ = [
    "analyze_repo_structure",
    "analyze_repo_languages", 
    "analyze_repo_directories"
]
