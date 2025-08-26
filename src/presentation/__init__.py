"""Presentation layer package for Generic Ollama Agent.

The presentation layer provides user interfaces for interacting with the
underlying system. This could include command line interfaces, GUI
applications, web services or any other means of delivering functionality
to end users. It should rely on the application layer to perform actions
and return data.
"""

from .interface import *  # noqa: F401,F403

__all__ = []