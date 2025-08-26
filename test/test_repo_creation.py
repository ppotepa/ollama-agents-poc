#!/usr/bin/env python3
"""Test the new repository creation functionality."""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from core.helpers import check_remote_repository_exists, extract_repo_name_from_url, create_new_repository, clone_repository


def test_repo_name_extraction():
    """Test repository name extraction from various URL formats."""
    print("🧪 Testing repository name extraction...")
    
    test_cases = [
        ("https://github.com/user/my-repo.git", "my-repo"),
        ("https://github.com/user/my-repo", "my-repo"),
        ("git@github.com:user/my-repo.git", "my-repo"),
        ("https://gitlab.com/user/my-project", "my-project"),
        ("invalid-url", None)  # Should use fallback
    ]
    
    for url, expected in test_cases:
        result = extract_repo_name_from_url(url)
        print(f"  URL: {url}")
        print(f"  Expected: {expected}, Got: {result}")
        if expected and result != expected:
            print(f"  ❌ FAIL")
        else:
            print(f"  ✅ PASS")
        print()


def test_remote_check():
    """Test checking if remote repositories exist."""
    print("🧪 Testing remote repository existence check...")
    
    test_cases = [
        ("https://github.com/octocat/Hello-World", True),  # Should exist
        ("https://github.com/nonexistent/nonexistent-repo-12345", False),  # Should not exist
    ]
    
    for url, expected in test_cases:
        print(f"  Checking: {url}")
        result = check_remote_repository_exists(url)
        print(f"  Expected: {expected}, Got: {result}")
        if result == expected:
            print(f"  ✅ PASS")
        else:
            print(f"  ❌ FAIL")
        print()


def safe_rmtree(path):
    """Safely remove directory tree, handling Windows permission issues with git files."""
    import stat
    
    def handle_remove_readonly(func, path, exc):
        """Error handler for removing read-only files on Windows."""
        if os.path.exists(path):
            os.chmod(path, stat.S_IWRITE)
            func(path)
    
    if path.exists():
        import shutil
        shutil.rmtree(path, onerror=handle_remove_readonly)


def test_repo_creation():
    """Test creating a new local repository."""
    print("🧪 Testing repository creation...")
    
    test_dir = Path("./test_repo_temp")
    
    # Clean up if exists
    safe_rmtree(test_dir)
    
    try:
        repo_name = "test-repo"
        git_url = "https://github.com/user/test-repo"
        
        result = create_new_repository(test_dir, repo_name, git_url)
        
        if result:
            print(f"  ✅ Repository created successfully")
            
            # Check if it's a valid git repo
            from core.helpers import is_git_repository
            if is_git_repository(str(test_dir)):
                print(f"  ✅ Git repository initialized correctly")
            else:
                print(f"  ❌ Git repository not properly initialized")
            
            # Check if README exists
            readme_path = test_dir / "README.md"
            if readme_path.exists():
                print(f"  ✅ README.md created")
                with open(readme_path, 'r') as f:
                    content = f.read()
                    if repo_name in content:
                        print(f"  ✅ README contains repository name")
                    else:
                        print(f"  ❌ README doesn't contain repository name")
            else:
                print(f"  ❌ README.md not created")
        else:
            print(f"  ❌ Repository creation failed")
            
    finally:
        # Clean up
        safe_rmtree(test_dir)


def test_full_clone_with_nonexistent_repo():
    """Test the full clone_repository function with a non-existent repository."""
    print("🧪 Testing full clone with non-existent repository...")
    
    test_dir = Path("./test_clone_temp")
    
    # Clean up if exists
    safe_rmtree(test_dir)
    
    try:
        # Use a non-existent repository URL
        git_url = "https://github.com/nonexistent/nonexistent-repo-12345"
        
        result = clone_repository(git_url, test_dir)
        
        if result:
            print(f"  ✅ Clone function handled non-existent repo correctly")
            
            # Check if directory was created
            if test_dir.exists():
                print(f"  ✅ Directory created")
                
                # Check if it's a valid git repo
                from core.helpers import is_git_repository
                if is_git_repository(str(test_dir)):
                    print(f"  ✅ Git repository initialized correctly")
                else:
                    print(f"  ❌ Git repository not properly initialized")
            else:
                print(f"  ❌ Directory not created")
        else:
            print(f"  ❌ Clone function failed")
            
    finally:
        # Clean up
        safe_rmtree(test_dir)


if __name__ == "__main__":
    print("🚀 Testing new repository creation functionality")
    print("=" * 60)
    
    test_repo_name_extraction()
    print("-" * 60)
    
    test_remote_check()
    print("-" * 60)
    
    test_repo_creation()
    print("-" * 60)
    
    test_full_clone_with_nonexistent_repo()
    print("-" * 60)
    
    print("✅ All tests completed!")
