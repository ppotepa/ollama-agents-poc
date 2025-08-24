"""Streaming support utilities (callbacks, real-time capture)."""
from __future__ import annotations

from typing import List
from src.utils.animations import stream_text


class StreamingCallbackHandler:
    def __init__(self):
        self.tokens: List[str] = []
        self._thinking = False

    def on_llm_start(self, *_, **__):  # pragma: no cover
        self._thinking = True
        stream_text("🧠 AI is thinking...", delay=0.005)

    def on_llm_new_token(self, token: str, **_):  # pragma: no cover
        if self._thinking:
            self._thinking = False
            stream_text("🤖 AI Response: ", delay=0.001, newline=False)
        print(token, end="", flush=True)
        self.tokens.append(token)

    def on_llm_end(self, *_):  # pragma: no cover
        print()


class RealTimeCapture:
    def __init__(self):
        self.buffer: List[str] = []
        self.is_streaming = False

    def start(self):
        self.buffer.clear()
        self.is_streaming = True

    def add(self, text: str):
        if self.is_streaming:
            self.buffer.append(text)

    def stop(self) -> str:
        self.is_streaming = False
        return "".join(self.buffer)


__all__ = ["StreamingCallbackHandler", "RealTimeCapture"]
