"""Core architecture package - modular, clean separation of concerns."""

# Import key components from each layer
from .state import VirtualRepository, RepositoryCache
from .io import (
    parse_github_url, 
    clone_repository, 
    get_repository_info,
    generate_repo_hash
)
from .validation import (
    AgentCapability,
    validate_repository_requirement,
    get_agent_capabilities
)
from .factory import get_agent_instance

__all__ = [
    # State management
    'VirtualRepository',
    'RepositoryCache',
    
    # I/O operations
    'parse_github_url',
    'clone_repository', 
    'get_repository_info',
    'generate_repo_hash',
    
    # Validation
    'AgentCapability',
    'validate_repository_requirement',
    'get_agent_capabilities',
    
    # Factory
    'get_agent_instance'
]

# ---------------------------------------------------------------------------
# Backwards compatibility support
#
# Some legacy modules import ``TaskAnalyzer`` directly from ``src.core``.
# While this class is not part of the modern public API, we provide a minimal
# stub here to avoid import errors.  The real implementation lives in
# ``src.core.investigation_strategies`` and may be enhanced there.  See
# ``improved_investigation_strategies.py`` for a more feature-rich analyzer.

try:
    # Attempt to re-export TaskAnalyzer from our investigation strategies module
    from .investigation_strategies import TaskAnalyzer  # type: ignore[attr-defined]
except Exception:
    # Define a basic stub if the class is unavailable
    class TaskAnalyzer:
        """Fallback TaskAnalyzer for backward compatibility.

        This stub provides a minimal ``analyze_task`` method returning basic
        information about the query.  Subclasses or other modules may
        override this behaviour to provide richer analysis.
        """

        def analyze_task(self, query: str) -> dict[str, object]:
            return {
                "original_query": query,
                "tools_required": False,
                "reason": "No specific analysis available (TaskAnalyzer stub)",
            }

__all__.append('TaskAnalyzer')
