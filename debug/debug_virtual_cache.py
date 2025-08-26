#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append('../src')

def debug_virtual_cache():
    """Debug what's in the virtual repository cache."""
    print("🔍 Debugging Virtual Repository Cache")
    print("=" * 50)
    
    try:
        from src.core.helpers import _virtual_repo_cache
        
        print(f"📊 Cache contains {len(_virtual_repo_cache)} entries:")
        for key, virtual_repo in _virtual_repo_cache.items():
            print(f"  🔑 Key: {key}")
            print(f"     📊 Files: {len(virtual_repo.files)}")
            print(f"     📂 Directories: {len(virtual_repo.directories)}")
            print(f"     🌐 URL: {virtual_repo.repo_url}")
            print()
            
        if not _virtual_repo_cache:
            print("❌ Virtual repository cache is empty")
            print("💡 This might mean:")
            print("   - No repository was downloaded with ZIP method")
            print("   - Repository was downloaded but virtual cache wasn't created")
            print("   - Different process/session was used")
            
    except Exception as e:
        print(f"❌ Error accessing virtual repository cache: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_virtual_cache()
