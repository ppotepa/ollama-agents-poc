"""Base agent abstraction for Generic Ollama Agent platform.

Defines the common interface every concrete agent must implement. This keeps
`main.py` and the runtime decoupled from any specific model backend.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

try:  # Optional import; we only need the flag helper
    from src.utils.animations import is_fast_all  # type: ignore
except Exception:  # pragma: no cover
    def is_fast_all():  # fallback
        return False


class AbstractAgent(ABC):
    """Abstract base class for all agents.

    Lifecycle:
      1. instantiate(agent_id, config)
      2. call load() to build underlying LLM + tool wiring
      3. call run()/stream() for interactions
    """

    agent_id: str
    display_name: str

    def __init__(self, agent_id: str, config: dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.display_name = config.get("name", agent_id)
        self._loaded = False
        self._tools: list[Any] = []  # LangChain tools or plain callables
        self._llm: Any = None
        self.capabilities = config.get("capabilities", {})

    def load(self) -> None:
        if self._loaded:
            return
        self._llm = self._build_llm()
        self._tools = self._build_tools()
        self._loaded = True

    @abstractmethod
    def _build_llm(self) -> Any:
        """Return underlying LLM client (or None if unavailable)."""

    @abstractmethod
    def _build_tools(self) -> list[Any]:
        """Return list of tool objects (may be empty)."""

    def run(self, prompt: str) -> str:
        if not self._loaded:
            self.load()
        if self._llm is None:
            return "⚠️ LLM backend not available. Install dependencies."
        try:
            if hasattr(self._llm, "invoke"):
                result = self._llm.invoke(prompt)
                if isinstance(result, dict) and "content" in result:
                    return str(result["content"])
                return str(result)
            return str(self._llm(prompt))
        except Exception as e:
            return f"❌ Error running agent: {e}"

    def stream(self, prompt: str, on_token) -> str:
        if not self._loaded:
            self.load()
        if self._llm is None:
            return "⚠️ LLM backend not available."
        text = self.run(prompt)
        if is_fast_all():  # no per-char emission
            on_token(text)
            return text
        for ch in text:
            on_token(ch)
        return text

    @property
    def tools(self) -> list[Any]:
        if not self._loaded:
            self.load()
        return self._tools

    def tool_names(self) -> list[str]:
        names = []
        for t in self.tools:
            name = getattr(t, "name", None) or getattr(t, "__name__", None)
            if name:
                names.append(name)
        return names

    def supports(self, capability: str) -> bool:
        return bool(self.capabilities.get(capability))


__all__ = ["AbstractAgent"]
