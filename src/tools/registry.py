"""Tool registration system."""
from __future__ import annotations

from typing import List, Any, Callable, Dict, Optional

_TOOLS: List[Any] = []


class ToolWrapper:
    """Standard wrapper for tools to make them compatible with LangChain."""
    def __init__(self, fn: Callable, name: Optional[str] = None, description: Optional[str] = None):
        """Initialize a tool wrapper.
        
        Args:
            fn: The function to wrap
            name: The name of the tool (defaults to function name)
            description: The description of the tool (defaults to function docstring)
        """
        self._fn = fn
        self.name = name or fn.__name__
        self.description = description or fn.__doc__ or fn.__name__
        self.args_schema = None
        
    def __call__(self, *args, **kwargs):
        """Call the wrapped function."""
        return self._fn(*args, **kwargs)
        
    # Make this object dict-like for compatibility with LangChain
    def get(self, key: str, default: Any = None) -> Any:
        """Get an attribute with dict-like syntax."""
        return getattr(self, key, default)


def register_tool(obj: Any):
    """Register a tool with the global registry.
    
    If the tool is a function, it will be wrapped with ToolWrapper.
    """
    # If it's a plain function, wrap it first
    if callable(obj) and not hasattr(obj, 'name'):
        obj = ToolWrapper(obj)
        
    _TOOLS.append(obj)
    return obj


def get_registered_tools() -> List[Any]:
    """Get all registered tools."""
    return list(_TOOLS)


__all__ = ["register_tool", "get_registered_tools", "ToolWrapper"]
