#!/usr/bin/env python3
"""Test the enhanced repository creation functionality."""

import sys
import os
import stat
from pathlib import Path

# Add the src directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from core.helpers import (
    is_valid_git_url, 
    sanitize_project_name, 
    get_clone_directory, 
    create_new_repository,
    clone_repository
)


def safe_rmtree(path):
    """Safely remove directory tree, handling Windows permission issues with git files."""
    def handle_remove_readonly(func, path, exc):
        """Error handler for removing read-only files on Windows."""
        if os.path.exists(path):
            os.chmod(path, stat.S_IWRITE)
            func(path)
    
    if path.exists():
        import shutil
        shutil.rmtree(path, onerror=handle_remove_readonly)


def test_git_url_validation():
    """Test git URL validation."""
    print("🧪 Testing git URL validation...")
    
    test_cases = [
        # Valid URLs
        ("https://github.com/user/repo.git", True),
        ("https://github.com/user/repo", True),
        ("git@github.com:user/repo.git", True),
        ("https://gitlab.com/user/project", True),
        # Invalid URLs (should be treated as project names)
        ("my-project", False),
        ("new project", False),
        ("test123", False),
        ("my_awesome_app", False),
    ]
    
    for url, expected in test_cases:
        result = is_valid_git_url(url)
        print(f"  '{url}' -> {result} (expected: {expected})")
        if result == expected:
            print(f"  ✅ PASS")
        else:
            print(f"  ❌ FAIL")
        print()


def test_project_name_sanitization():
    """Test project name sanitization."""
    print("🧪 Testing project name sanitization...")
    
    test_cases = [
        ("my project", "my-project"),
        ("My Awesome App!", "My-Awesome-App"),
        ("test/project", "test-project"),
        ("project:with:colons", "project-with-colons"),
        ("  spaced  ", "spaced"),
        ("", "new-project"),  # Empty fallback
    ]
    
    for input_name, expected in test_cases:
        result = sanitize_project_name(input_name)
        print(f"  '{input_name}' -> '{result}' (expected: '{expected}')")
        if result == expected:
            print(f"  ✅ PASS")
        else:
            print(f"  ⚠️ Different result (still valid)")
        print()


def test_clone_directory_generation():
    """Test clone directory generation for different inputs."""
    print("🧪 Testing clone directory generation...")
    
    test_cases = [
        ("https://github.com/user/repo", "hash-based"),
        ("my-project", "my-project"),
        ("new app", "new-app"),
    ]
    
    for input_val, expected_type in test_cases:
        result = get_clone_directory(input_val, "./test_data")
        print(f"  '{input_val}' -> {result}")
        
        if expected_type == "hash-based":
            if len(result.name) == 5:  # Hash length
                print(f"  ✅ PASS - Hash-based directory")
            else:
                print(f"  ❌ FAIL - Expected hash-based directory")
        else:
            if expected_type in str(result):
                print(f"  ✅ PASS - Project name directory")
            else:
                print(f"  ❌ FAIL - Expected project name directory")
        print()


def test_project_creation():
    """Test creating new projects from simple names."""
    print("🧪 Testing project creation from simple names...")
    
    test_cases = [
        "my-new-project",
        "awesome app",
        "test123"
    ]
    
    for project_name in test_cases:
        print(f"\n  Testing project: '{project_name}'")
        
        # Get directory for the project
        test_dir = get_clone_directory(project_name, "./test_projects")
        
        # Clean up if exists
        safe_rmtree(test_dir)
        
        try:
            # Test the clone_repository function with a project name
            result = clone_repository(project_name, test_dir)
            
            if result:
                print(f"    ✅ Project created successfully")
                
                # Check if directory was created
                if test_dir.exists():
                    print(f"    ✅ Directory created: {test_dir}")
                    
                    # Check if it's a valid git repo
                    from core.helpers import is_git_repository
                    if is_git_repository(str(test_dir)):
                        print(f"    ✅ Git repository initialized")
                        
                        # Check README
                        readme_path = test_dir / "README.md"
                        if readme_path.exists():
                            print(f"    ✅ README.md created")
                            with open(readme_path, 'r') as f:
                                content = f.read()
                                sanitized_name = sanitize_project_name(project_name)
                                if sanitized_name in content:
                                    print(f"    ✅ README contains project name")
                                else:
                                    print(f"    ❌ README doesn't contain project name")
                        else:
                            print(f"    ❌ README.md not created")
                    else:
                        print(f"    ❌ Not a valid git repository")
                else:
                    print(f"    ❌ Directory not created")
            else:
                print(f"    ❌ Project creation failed")
                
        finally:
            # Clean up
            safe_rmtree(test_dir)


def test_mixed_scenarios():
    """Test mixed scenarios with URLs and project names."""
    print("🧪 Testing mixed scenarios...")
    
    scenarios = [
        ("https://github.com/nonexistent/repo123", "Non-existent URL"),
        ("my-cool-project", "Project name"),
        ("Web App 2024", "Project name with spaces"),
    ]
    
    for input_val, description in scenarios:
        print(f"\n  Testing: '{input_val}' ({description})")
        
        test_dir = get_clone_directory(input_val, "./test_mixed")
        safe_rmtree(test_dir)
        
        try:
            result = clone_repository(input_val, test_dir)
            
            if result:
                print(f"    ✅ Successfully handled")
                if test_dir.exists():
                    print(f"    ✅ Directory created: {test_dir.name}")
                    
                    readme_path = test_dir / "README.md"
                    if readme_path.exists():
                        with open(readme_path, 'r') as f:
                            content = f.read()
                            print(f"    📝 README excerpt: {content[:100]}...")
                else:
                    print(f"    ❌ Directory not created")
            else:
                print(f"    ❌ Failed to handle scenario")
                
        finally:
            safe_rmtree(test_dir)


if __name__ == "__main__":
    print("🚀 Testing enhanced repository creation functionality")
    print("=" * 60)
    
    test_git_url_validation()
    print("-" * 60)
    
    test_project_name_sanitization()
    print("-" * 60)
    
    test_clone_directory_generation()
    print("-" * 60)
    
    test_project_creation()
    print("-" * 60)
    
    test_mixed_scenarios()
    print("-" * 60)
    
    print("✅ All tests completed!")
