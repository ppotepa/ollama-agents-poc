"""Tool registration system."""
from __future__ import annotations

from typing import List, Any

_TOOLS: List[Any] = []


def register_tool(obj: Any):
    _TOOLS.append(obj)
    return obj


def get_registered_tools() -> List[Any]:
    return list(_TOOLS)


__all__ = ["register_tool", "get_registered_tools"]
