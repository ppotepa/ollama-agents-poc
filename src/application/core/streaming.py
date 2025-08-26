"""Streaming support utilities (callbacks, real-time capture)."""
from __future__ import annotations

import threading

from src.utils.animations import stream_text


class StreamingCallbackHandler:
    def __init__(self):
        self.tokens: list[str] = []
        self._thinking = False
        self._lock = threading.Lock()
        self._printed_tokens: set[str] = set()

    def on_llm_start(self, *_, **__):  # pragma: no cover
        self._thinking = True
        stream_text("🧠 AI is thinking...", delay=0.005)

    def on_llm_new_token(self, token: str, **_):  # pragma: no cover
        with self._lock:
            if self._thinking:
                self._thinking = False
                stream_text("🤖 AI Response: ", delay=0.001, newline=False)

            # Only print if this token hasn't been printed yet
            token_id = f"{len(self.tokens)}_{token}"
            if token_id not in self._printed_tokens:
                print(token, end="", flush=True)
                self._printed_tokens.add(token_id)

            self.tokens.append(token)

    def on_llm_end(self, *_):  # pragma: no cover
        print()


class RealTimeCapture:
    def __init__(self):
        self.buffer: list[str] = []
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
