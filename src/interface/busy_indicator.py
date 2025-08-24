"""Animated busy indicator for chat interface."""

import threading
import time
import sys


class BusyIndicator:
    """Animated busy indicator with color transitions and dot cycling."""
    
    def __init__(self, text: str = "Generating response", enabled: bool = True):
        self.text = text
        self.enabled = enabled and getattr(sys.stdout, 'isatty', lambda: False)()
        self._stop = threading.Event()
        # Color fade sequence + dot cycle
        self.colors = [
            "\033[97;100m",  # bright white on gray
            "\033[37;100m",  # white
            "\033[2;37;100m", # dim white
            "\033[90;100m",  # gray
            "\033[2;90;100m"  # dim gray
        ]
        self.dots_cycle = ["", ".", "..", "..."]  # classic dot progression
        self.interval = 0.35
        self.thread: threading.Thread | None = None
        self._frame = 0

    def start(self):
        """Start the animated indicator."""
        if not self.enabled:
            print(self.text, flush=True)
            return
        
        def run():
            i = 0
            while not self._stop.is_set():
                color = self.colors[i % len(self.colors)]
                dots = self.dots_cycle[i % len(self.dots_cycle)]
                base = f"{self.text}{dots}".ljust(len(self.text) + 3)
                print(f"\r{color} {base} \033[0m", end="", flush=True)
                i += 1
                self._frame = i
                # Wait or break early if stop requested
                for _ in range(10):
                    if self._stop.is_set():
                        break
                    self._stop.wait(self.interval / 10)
        
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the animated indicator and clear the line."""
        if not self.enabled:
            # overwrite plain line
            print("\r" + " " * (len(self.text) + 6) + "\r", end="", flush=True)
            return
        
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=0.5)
        # Clear line
        print("\r" + " " * (len(self.text) + 8) + "\r", end="", flush=True)
