"""Domain objects related to repositories.

The domain layer defines the RepositoryContext which encapsulates all
information about a code repository required by the rest of the system. The
context aggregates data produced by repository analyzers but remains free of
any IO operations or external dependencies.
"""

from .repository_context import RepositoryContext  # noqa: F401

__all__ = ["RepositoryContext"]