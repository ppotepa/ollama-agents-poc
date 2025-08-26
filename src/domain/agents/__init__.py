"""Domain agent abstractions.

This subpackage defines the core agent abstractions for the system. In the
domain layer we only describe what an agent is and what behaviour it must
provide via the AbstractAgent class. Any concrete agent implementations live
in the application layer. By separating the abstract concept of an agent here
we can decouple our business logic from particular model backends.
"""

from .base.base_agent import AbstractAgent  # noqa: F401
from .descriptors import *  # noqa: F401,F403

__all__ = ["AbstractAgent"]