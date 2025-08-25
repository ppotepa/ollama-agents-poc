"""Test script to verify the encoding fix for model pulling."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.model_discovery import OllamaModelDiscovery

def test_encoding_fix():
    """Test the encoding fix directly."""
    print("Testing encoding fix for model pulling...")
    
    discovery = OllamaModelDiscovery()
    
    # Test with a model that typically has encoding issues
    result = discovery.pull_model('qwen2.5:7b-instruct-q4_K_M')
    print(f"Pull result: {result}")
    
    return result

if __name__ == "__main__":
    test_encoding_fix()
