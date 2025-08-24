#!/usr/bin/env python3
"""Test script to validate repository context tools integration."""

import sys
from pathlib import Path

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Test imports
print("🔍 Testing repository context tools integration...")

try:
    from src.tools.repository_context_tool import (
        analyze_repo_structure, 
        analyze_repo_languages, 
        analyze_repo_directories
    )
    print("✅ Repository tools imported successfully")
except Exception as e:
    print(f"❌ Failed to import repository tools: {e}")
    sys.exit(1)

# Test tool registry
try:
    from src.tools.registry import get_registered_tools
    tools = get_registered_tools()
    repo_tools = [tool for tool in tools if 'repo' in tool.name]
    print(f"✅ Found {len(repo_tools)} repository tools in registry:")
    for tool in repo_tools:
        print(f"   - {tool.name}: {tool.description}")
except Exception as e:
    print(f"❌ Failed to check tool registry: {e}")
    sys.exit(1)

# Test actual functionality
print("\n🧪 Testing tool functionality...")

try:
    # Test basic structure analysis
    result = analyze_repo_structure(".")
    print(f"✅ analyze_repo_structure: {len(result)} characters")
    
    # Test language analysis
    result = analyze_repo_languages(".")
    print(f"✅ analyze_repo_languages: {len(result)} characters")
    
    # Test directory analysis
    result = analyze_repo_directories(".", max_depth=2)
    print(f"✅ analyze_repo_directories: {len(result)} characters")
    
except Exception as e:
    print(f"❌ Tool functionality test failed: {e}")
    sys.exit(1)

print("\n🎉 All tests passed! Repository context tools are fully integrated.")
print("\nℹ️  Tools available for agents:")
print("   - analyze_repo_structure: Full repository analysis")
print("   - analyze_repo_languages: Programming language breakdown")
print("   - analyze_repo_directories: Directory structure overview")
