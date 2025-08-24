"""Decorator-based agent descriptor system."""

from typing import List, Optional, Type, Dict, Any, Callable, Set, TypeVar, TYPE_CHECKING
from functools import wraps

from src.core.enums import ModelFamily, Domain, AgentCapability, ToolType

if TYPE_CHECKING:
    from src.agents.base.base_agent import BaseAgent

# Base Agent Type
T = TypeVar('T')

# Global registry
_registry = {}

def AgentDescriptor(
    name: str,
    backend_image: str,
    description: Optional[str] = None,
    family: Optional[ModelFamily] = None,
    domain: Optional[Domain] = None,
) -> Callable[[Type[T]], Type[T]]:
    """Main agent decorator defining basic properties."""
    def decorator(cls: Type[T]) -> Type[T]:
        cls._agent_name = name
        cls._backend_image = backend_image
        cls._description = description
        cls._family = family or ModelFamily.from_name(backend_image)
        cls._domain = domain
        
        # Generate ID from name
        agent_id = name.lower().replace(" ", "_").replace(":", "_")
        _registry[agent_id] = cls
        
        # Add descriptor accessor methods
        @property
        def requires_repository(self):
            return (self._domain == Domain.CODING or 
                   (hasattr(self, '_capabilities') and AgentCapability.CODE in self._capabilities))
        
        cls.requires_repository = requires_repository
        return cls
    return decorator

def Capabilities(*capability_flags: AgentCapability):
    """Define agent capabilities using enum flags."""
    def decorator(cls):
        # Combine all capability flags
        combined = AgentCapability.NONE
        for cap in capability_flags:
            combined |= cap
        cls._capabilities = combined
        return cls
    return decorator

def Parameters(**params):
    """Define model parameters."""
    def decorator(cls):
        cls._parameters = params
        return cls
    return decorator

def Tools(*tool_names: str):
    """Define tools this agent can use."""
    def decorator(cls):
        cls._tools = list(tool_names)
        cls._tool_types = {ToolType.for_tool(tool) for tool in tool_names}
        return cls
    return decorator

def discover_agents():
    """Auto-discover and load agent modules with descriptors."""
    return _registry

def get_agent_class(agent_id):
    """Get agent class by ID."""
    return _registry.get(agent_id)

def list_agents():
    """Get all registered agents with metadata."""
    result = {}
    
    for agent_id, agent_cls in _registry.items():
        result[agent_id] = {
            "name": getattr(agent_cls, "_agent_name", agent_id),
            "backend_image": getattr(agent_cls, "_backend_image", ""),
            "description": getattr(agent_cls, "_description", ""),
            "family": agent_cls._family.name if hasattr(agent_cls, "_family") and agent_cls._family else None,
            "domain": agent_cls._domain.name if hasattr(agent_cls, "_domain") and agent_cls._domain else None,
            "requires_repo": getattr(agent_cls, "requires_repository", False),
            "capabilities": getattr(agent_cls, "_capabilities", AgentCapability.NONE).to_strings(),
            "tools": getattr(agent_cls, "_tools", [])
        }
    return result
