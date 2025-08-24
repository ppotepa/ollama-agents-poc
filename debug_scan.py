#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append('src')

from src.tools.context import RepositoryContextBuilder

def debug_scan():
    """Debug the repository scanning for VS Code."""
    repo_path = "./data/0d8bc"
    print(f"🔍 Debugging repository scan for: {repo_path}")
    
    # Check if directory exists
    path = Path(repo_path)
    if not path.exists():
        print(f"❌ Directory does not exist: {path.absolute()}")
        return
    
    print(f"✅ Directory exists: {path.absolute()}")
    
    # List directory contents
    try:
        items = list(path.iterdir())
        print(f"📁 Directory contains {len(items)} items:")
        for item in items[:10]:  # Show first 10 items
            print(f"   {'📁' if item.is_dir() else '📄'} {item.name}")
        if len(items) > 10:
            print(f"   ... and {len(items) - 10} more items")
    except Exception as e:
        print(f"❌ Error listing directory: {e}")
        return
    
    # Test the builder
    print(f"\n🔧 Testing RepositoryContextBuilder...")
    try:
        builder = RepositoryContextBuilder(repo_path)
        print(f"✅ Builder created successfully")
        print(f"   Repository path: {builder.repository_path}")
        
        # Test ignore logic on some items
        print(f"\n🚫 Testing ignore logic:")
        for item in items[:5]:
            should_ignore = builder._should_ignore_path(item)
            print(f"   {'🚫' if should_ignore else '✅'} {item.name} -> {'IGNORE' if should_ignore else 'INCLUDE'}")
        
        # Try building context
        print(f"\n🏗️  Building context...")
        context = builder.build_context_map(cache_content=False)
        print(f"✅ Context built: {context.total_files} files, {context.total_directories} directories")
        
    except Exception as e:
        print(f"❌ Error with builder: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_scan()
