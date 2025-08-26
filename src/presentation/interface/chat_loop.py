"""Interactive chat loop for Generic Ollama Agent."""
from __future__ import annotations

import re
import sys
import threading

try:
    from src.utils.animations import is_fast_all
except Exception:  # pragma: no cover
    def is_fast_all():  # type: ignore
        return False

EXIT_ALIASES = {"/exit", "/quit", ":q", ":quit", "q", "quit", "exit", "/q", "/bye", "bye"}

class BusyIndicator:
    """Animated busy indicator with color + dot cycles."""
    def __init__(self, text: str = "Generating response", enabled: bool = True):
        self.text = text
        self.enabled = enabled and getattr(sys.stdout, 'isatty', lambda: False)()
        self._stop = threading.Event()
        self.colors = ["\033[97;100m", "\033[37;100m", "\033[2;37;100m", "\033[90;100m", "\033[2;90;100m"]
        self.dots = ["", ".", "..", "..."]
        self.interval = 0.35
        self.thread: threading.Thread | None = None

    def start(self):
        if not self.enabled:
            print(self.text, flush=True)
            return
        def run():
            i = 0
            while not self._stop.is_set():
                color = self.colors[i % len(self.colors)]
                dots = self.dots[i % len(self.dots)]
                base = f"{self.text}{dots}".ljust(len(self.text) + 3)
                print(f"\r{color} {base} \033[0m", end="", flush=True)
                i += 1
                for _ in range(10):
                    if self._stop.is_set():
                        break
                    self._stop.wait(self.interval / 10)
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.enabled:
            print("\r" + " " * (len(self.text) + 6) + "\r", end="", flush=True)
            return
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=0.5)
        print("\r" + " " * (len(self.text) + 8) + "\r", end="", flush=True)


def parse_output_redirection(prompt: str):
    m = re.search(r"\s>\s([^\s]+)$", prompt)
    if not m:
        return prompt, None
    return prompt[:m.start()].rstrip(), m.group(1)


def filter_stream_tokens():
    in_think = {"flag": False}
    def _filter(tok: str) -> str:
        if "<think>" in tok:
            in_think["flag"] = True
            tok = tok.split("<think>")[0]
        if "</think>" in tok:
            parts = tok.split("</think>")
            in_think["flag"] = False
            tok = parts[-1]
        return "" if in_think["flag"] else tok
    return _filter


def interactive_chat(agent, emit_func=print, initial_query: str | None = None):
    """
    Start interactive chat session with agent.

    Args:
        agent: The agent instance
        emit_func: Function to use for output
        initial_query: Optional first query to process automatically
    """
    emit_func("\n💬 Enter chat mode. Type /help for commands. /exit to quit.\n")

    def show_help():
        emit_func("""Commands:
  /help            Show this help
  /tools           List loaded tools
  /system          Show system message (if any)
  /clear           Clear screen
  /exit, /quit     Exit chat
""")

    show_help()
    interrupt_count = 0

    # Process initial query if provided
    if initial_query:
        emit_func(f"\n🧠 You > {initial_query}")
        process_user_input(initial_query)

    # Continue with regular chat loop
    while True:
        try:
            emit_func("\n🧠 You > ", newline=False)
            user_input = input().strip()

            # Process chat commands
            if user_input.startswith('/'):
                if process_command(user_input):
                    continue  # Command was processed
                else:
                    break  # Exit command

            # Skip empty inputs
            if not user_input:
                continue

            # Process normal input
            process_user_input(user_input)

        except KeyboardInterrupt:
            if interrupt_count > 0:
                # Second interrupt - exit
                emit_func("\n👋 Exiting on double interrupt.")
                break
            interrupt_count += 1
            emit_func("\n⚠️ Interrupted. Type /exit to quit.")
            continue

    emit_func("\n👋 Goodbye!")
    return


def process_user_input(user_input: str):
    """Process user input - placeholder function."""
    # TODO: Implement actual user input processing
    print(f"Processing: {user_input}")


def process_command(command: str) -> bool:
    """Process chat commands - placeholder function."""
    # TODO: Implement actual command processing
    if command in EXIT_ALIASES:
        return False
    print(f"Processing command: {command}")
    return True
