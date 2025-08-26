"""Domain layer package for Generic Ollama Agent.

This package exposes the core domain entities and interfaces used across the
application. It contains abstractions for agents, repository context, enums,
investigation types, and interfaces that describe contracts for orchestration
and repository access. By isolating these definitions in the domain layer we
promote separation of concerns and make it clear which pieces of the system
represent fundamental business concepts as opposed to application logic or
infrastructure details.
"""

# Re-export key domain components for convenient access from a single import.
from .agents.base import AbstractAgent  # noqa: F401
from .agents.descriptors import *  # noqa: F401,F403
from .repository.repository_context import RepositoryContext  # noqa: F401
from .interfaces.orchestrator_interface import OrchestratorInterface  # noqa: F401
from .interfaces.repository_interface import RepositoryInterface  # noqa: F401
from .enums import *  # noqa: F401,F403
from .investigation.types import *  # noqa: F401,F403

__all__ = [
    "AbstractAgent",
    "RepositoryContext",
    "OrchestratorInterface",
    "RepositoryInterface",
]