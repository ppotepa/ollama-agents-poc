#!/usr/bin/env python3
"""Test script to verify the file redirection feature works."""

import sys
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from src.interface.chat_loop import parse_output_redirection

def test_redirection():
    test_cases = [
        "generate me 200 random words and output them to file > test_words.txt",
        "create some data > output.txt",
        "hello world > simple.txt",
        "no redirection here",
        "multiple > symbols > file.txt",
    ]
    
    for test in test_cases:
        prompt, file_target = parse_output_redirection(test)
        print(f"Input: '{test}'")
        print(f"  Prompt: '{prompt}'")
        print(f"  File: {file_target}")
        print()

if __name__ == "__main__":
    test_redirection()
