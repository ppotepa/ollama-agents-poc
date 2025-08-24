"""DeepCoderAgent - concrete implementation using Ollama backend."""
from __future__ import annotations

from typing import Any, Dict, List

from src.agents.base.base_agent import AbstractAgent
from src.utils.animations import stream_text

try:  # LangChain optional
    from langchain_ollama import ChatOllama
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

from src.tools.registry import get_registered_tools  # registry list
import src.tools  # noqa: F401  # trigger side-effect imports & registration


class DeepCoderAgent(AbstractAgent):
    def _build_llm(self) -> Any:  # noqa: D401
        if not LANGCHAIN_AVAILABLE:
            return None
        model_id = self.config.get("backend_image") or self.config.get("model_id")
        params = self.config.get("parameters", {})
        llm = ChatOllama(
            model=model_id,
            streaming=True,
            **{k: v for k, v in params.items() if isinstance(k, str)}
        )
        return llm

    def _build_tools(self) -> List[Any]:  # noqa: D401
        desired = set(self.config.get("tools", []))
        selected: List[Any] = []
        for t in get_registered_tools():
            name = getattr(t, "name", None)
            if name and name in desired:
                selected.append(t)
        if desired and not selected:
            stream_text("⚠️ No matching tools found for this agent configuration")
        return selected

    @property
    def system_message(self) -> str:
        return self.config.get("system_message", "You are an AI coding assistant.")

    def stream(self, prompt: str, on_token):  # type: ignore[override]
        if not self._loaded:
            self.load()
        if not LANGCHAIN_AVAILABLE or self._llm is None:
            return super().stream(prompt, on_token)
        final_tokens: List[str] = []
        try:
            for chunk in self._llm.stream(prompt):  # type: ignore[attr-defined]
                text = getattr(chunk, "content", None) or str(chunk)
                if text:
                    on_token(text)
                    final_tokens.append(text)
        except Exception as e:
            stream_text(f"❌ Streaming error: {e}")
            return ""
        return "".join(final_tokens)


def create_agent(agent_id: str, config: Dict[str, Any]) -> DeepCoderAgent:
    return DeepCoderAgent(agent_id, config)


__all__ = ["DeepCoderAgent", "create_agent"]
