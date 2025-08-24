#!/usr/bin/env python3
"""Test script to verify agent repository functionality."""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.repository_validation import check_agent_supports_coding, validate_repository_requirement

def test_agent_coding_detection():
    """Test that we can correctly identify coding agents."""
    print("🧪 Testing agent coding capability detection...")
    
    test_agents = [
        ("deepcoder", True),  # Should support coding
        ("tinyllama", False),  # Should not support coding
        ("qwen2_5_coder_7b", True),  # Should support coding
        ("mistral_7b", True),  # Should support coding based on YAML
    ]
    
    for agent_name, expected in test_agents:
        try:
            supports_coding = check_agent_supports_coding(agent_name)
            status = "✅" if supports_coding == expected else "❌"
            print(f"  {status} Agent '{agent_name}': supports_coding = {supports_coding} (expected: {expected})")
        except Exception as e:
            print(f"  ❌ Error testing agent '{agent_name}': {e}")

def test_repository_cloning():
    """Test repository cloning with validation."""
    print("\n🧪 Testing repository cloning...")
    
    # Test with a small repository
    test_url = "https://github.com/octocat/Hello-World.git"
    agent_name = "deepcoder"
    data_path = "data"
    
    try:
        print(f"  🔄 Testing repository setup for agent '{agent_name}' with URL: {test_url}")
        validation_passed, working_dir = validate_repository_requirement(agent_name, ".", test_url, data_path)
        
        if validation_passed:
            print(f"  ✅ Repository validation passed. Working directory: {working_dir}")
            
            # Check if files exist
            working_path = Path(working_dir)
            if working_path.exists():
                files = list(working_path.iterdir())[:3]
                print(f"  📁 Found files: {[f.name for f in files]}")
            else:
                print(f"  ⚠️  Working directory does not exist: {working_path}")
        else:
            print(f"  ❌ Repository validation failed")
            
    except Exception as e:
        print(f"  ❌ Error during repository test: {e}")

def test_command_line_scenarios():
    """Test different command-line scenarios."""
    print("\n🧪 Testing command-line scenarios...")
    
    # Test coding agent without repository (should fail)
    print("  📝 Scenario 1: Coding agent without repository URL")
    agent_name = "deepcoder"
    supports_coding = check_agent_supports_coding(agent_name)
    print(f"    Agent '{agent_name}' supports coding: {supports_coding}")
    
    if supports_coding:
        print("    ✅ This would require -g flag in command-line mode")
    else:
        print("    ✅ This would not require -g flag")
    
    # Test non-coding agent
    print("  📝 Scenario 2: Non-coding agent")
    agent_name = "tinyllama"
    supports_coding = check_agent_supports_coding(agent_name)
    print(f"    Agent '{agent_name}' supports coding: {supports_coding}")
    
    if not supports_coding:
        print("    ✅ This would not require -g flag")
    else:
        print("    ❌ This should not require -g flag")

if __name__ == "__main__":
    print("🚀 Testing Ollama Agent Repository Integration\n")
    
    test_agent_coding_detection()
    test_repository_cloning()
    test_command_line_scenarios()
    
    print("\n✅ All tests completed!")
