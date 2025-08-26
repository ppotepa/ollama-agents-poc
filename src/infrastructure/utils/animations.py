"""
DeepCoder Streaming Utilities
Extracted from deepcoder_interactive_stream.py for modular architecture
"""

import time

# Fast mode flags (set via main CLI)
FAST_ALL = False  # When True: disable character tokenization globally

def set_fast_all(value: bool):  # pragma: no cover - simple setter
    global FAST_ALL
    FAST_ALL = value

def is_fast_all() -> bool:
    return FAST_ALL


def stream_text(text: str, delay: float = 0.03, newline: bool = True):
    """Stream text character by character to simulate real-time generation.

    Honors FAST_ALL flag to bypass delays for faster debug cycles.
    """
    if FAST_ALL or delay <= 0:
        print(text, end='' if not newline else '\n', flush=True)
        return
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    if newline:
        print()


def stream_multiline(text: str, line_delay: float = 0.1, char_delay: float = 0.02):
    """Stream multiline text with delays between lines and characters."""
    lines = text.split('\n')
    for i, line in enumerate(lines):
        stream_text(line, delay=char_delay, newline=True)
        if i < len(lines) - 1:  # Don't delay after the last line
            time.sleep(line_delay)


def show_thinking_animation(duration: float = 2.0, message: str = "🔄 Thinking"):
    """Show a thinking animation for a specified duration."""
    animation = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    end_time = time.time() + duration
    i = 0

    while time.time() < end_time:
        print(f"\r{message} {animation[i % len(animation)]}", end='', flush=True)
        time.sleep(0.1)
        i += 1

    print(f"\r{' ' * (len(message) + 5)}\r", end='', flush=True)  # Clear the line


def progressive_reveal(sections: list[tuple], section_delay: float = 0.5):
    """Progressively reveal sections of text with animations."""
    for title, content in sections:
        stream_text(f"\n{title}", delay=0.05)
        time.sleep(section_delay)
        stream_multiline(content, line_delay=0.1, char_delay=0.01)
        time.sleep(section_delay)
