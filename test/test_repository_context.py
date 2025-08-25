#!/usr/bin/env python3
"""Test script to verify repository context system is working."""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_repository_context():
    """Test that repository context tools work correctly."""
    print("🧪 Testing repository context system...")
    
    try:
        from tools.context import get_repository_state, analyze_repository_context, build_repository_context
        
        # Test from current directory
        print(f"📍 Current directory: {os.getcwd()}")
        
        # Try to get repository state
        print("🔍 Testing get_repository_state()...")
        repo_state = get_repository_state()
        print(f"Repository state result: {repo_state[:200]}...")
        
        # If no context loaded, try to build it
        if "No repository context" in repo_state or repo_state.startswith("❌"):
            print("🔄 No context found, trying to build repository context...")
            
            # Try building context for current directory
            context_result = build_repository_context(".", force_rebuild=True, cache_content=False)
            print(f"Build context result: {context_result[:200]}...")
            
            # Try again
            repo_state = get_repository_state()
            print(f"Repository state after build: {repo_state[:200]}...")
        
        # Test analyze_repository_context
        print("🔍 Testing analyze_repository_context()...")
        analysis = analyze_repository_context(".")
        print(f"Analysis result: {analysis[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing repository context: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_repository_context()
