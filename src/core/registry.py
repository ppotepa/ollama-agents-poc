"""Agent registry & discovery."""
from __future__ import annotations

from importlib import import_module
from typing import Dict, Any, Callable

_FACTORIES: Dict[str, Callable[[str, Dict[str, Any]], Any]] = {}
_ALIASES: Dict[str, str] = {}


def register(agent_key: str, factory: Callable[[str, Dict[str, Any]], Any]):
    _FACTORIES[agent_key] = factory


def resolve_key(key: str) -> str:
    return _ALIASES.get(key, key)


def create(agent_key: str, config: Dict[str, Any]):
    key = resolve_key(agent_key)
    if key not in _FACTORIES:
        if key == "deepcoder":
            mod = import_module("src.agents.deepcoder.agent")
            register("deepcoder", mod.create_agent)  # type: ignore[attr-defined]
        else:
            raise KeyError(f"Unknown agent '{agent_key}'")
    return _FACTORIES[key](key, config)


def list_registered() -> Dict[str, Callable]:
    return dict(_FACTORIES)


__all__ = ["create", "register", "list_registered", "resolve_key"]
