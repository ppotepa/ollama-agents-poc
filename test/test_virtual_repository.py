#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append('src')

def test_virtual_repository():
    """Test virtual repository functionality."""
    print("🧪 Testing Virtual Repository System")
    print("=" * 50)
    
    # Test 1: Check if virtual repository exists
    try:
        from src.core.helpers import get_virtual_repository, parse_github_url
        
        repo_url = "https://github.com/microsoft/vscode"
        github_info = parse_github_url(repo_url)
        if github_info:
            owner = github_info['owner']
            repo = github_info['repo']
            branch = github_info['branch']
            cache_key = f"{owner}/{repo}:{branch}"
            
            print(f"🔍 Looking for virtual repository: {cache_key}")
            
            virtual_repo = get_virtual_repository(cache_key)
            if virtual_repo:
                print(f"✅ Virtual repository found!")
                print(f"   📊 Files: {len(virtual_repo.files)}")
                print(f"   📂 Directories: {len(virtual_repo.directories)}")
                print(f"   💾 ZIP Size: {virtual_repo.metadata.get('zip_size', 0) / 1024 / 1024:.1f} MB")
                
                # Test file access
                print(f"\n📄 Testing file access:")
                files = list(virtual_repo.files.keys())[:5]  # First 5 files
                for file_path in files:
                    content = virtual_repo.get_file_content_text(file_path)
                    if content:
                        lines = content.split('\n')
                        print(f"   ✅ {file_path}: {len(lines)} lines")
                    else:
                        print(f"   📦 {file_path}: Binary file")
                
                # Test directory access
                print(f"\n📁 Testing directory access:")
                root_contents = virtual_repo.list_directory('')
                print(f"   Root directory contains: {len(root_contents)} items")
                for item in root_contents[:10]:  # First 10 items
                    print(f"     - {item}")
                
            else:
                print(f"❌ No virtual repository found for {cache_key}")
        else:
            print(f"❌ Could not parse GitHub URL: {repo_url}")
            
    except Exception as e:
        print(f"❌ Error testing virtual repository: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Check virtual repository context
    print(f"\n🔍 Testing virtual repository context...")
    try:
        from src.tools.context import get_virtual_repository_context
        
        context = get_virtual_repository_context("https://github.com/microsoft/vscode")
        if context:
            print(f"✅ Virtual repository context generated:")
            print(context)
        else:
            print(f"❌ No virtual repository context available")
            
    except Exception as e:
        print(f"❌ Error testing virtual repository context: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_virtual_repository()
