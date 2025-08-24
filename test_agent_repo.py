#!/usr/bin/env python3
"""Test script to verify agent repository functionality."""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.helpers import check_agent_supports_coding, validate_repository_requirement, download_github_zip, parse_github_url
from tools.context import build_repository_context, load_repository_context_after_clone
import tempfile
import shutil

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

def test_vscode_repository_zip_download():
    """Test VS Code repository ZIP download and context loading."""
    print("\n🧪 Testing VS Code repository ZIP download...")
    
    # Use VS Code repository as it's a well-known, large repository
    test_url = "https://github.com/microsoft/vscode"
    branch = "main"
    
    try:
        print(f"  🔄 Testing ZIP download for VS Code repository...")
        
        # Parse GitHub URL
        github_info = parse_github_url(test_url)
        if github_info:
            owner = github_info['owner']
            repo = github_info['repo']
            branch = github_info['branch']
            print(f"  📊 Parsed URL: owner={owner}, repo={repo}, branch={branch}")
        else:
            print(f"  ❌ Could not parse GitHub URL: {test_url}")
            return
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"  📁 Using temporary directory: {temp_dir}")
            
            # Download and extract repository
            success = download_github_zip(test_url, Path(temp_dir))
            
            if success:
                print("  ✅ ZIP download and extraction successful")
                
                # Check if extracted directory exists - it should be extracted directly to temp_dir
                extracted_path = Path(temp_dir)
                if extracted_path.exists() and any(extracted_path.iterdir()):
                    print(f"  📁 Repository extracted successfully to: {extracted_path}")
                    
                    # Count files in root directory
                    root_files = list(extracted_path.iterdir())
                    print(f"  📄 Root directory contains {len(root_files)} items")
                    
                    # Show some key VS Code files we expect
                    expected_files = ['package.json', 'src', 'extensions', 'build', 'resources']
                    found_files = []
                    
                    for expected in expected_files:
                        file_path = extracted_path / expected
                        if file_path.exists():
                            found_files.append(expected)
                    
                    print(f"  ✅ Found expected VS Code files: {found_files}")
                    
                    # Test repository context loading
                    print("  🔄 Testing repository context loading...")
                    
                    # Change to the extracted directory for context analysis
                    original_cwd = os.getcwd()
                    try:
                        os.chdir(str(extracted_path))
                        
                        # Load repository context
                        load_success = load_repository_context_after_clone(
                            str(extracted_path), 
                            cache_content=False,  # Don't cache content for large repos
                            quiet=True
                        )
                        
                        if load_success:
                            print("  ✅ Repository context loaded successfully")
                            
                            # Build basic context report
                            context_report = build_repository_context(
                                str(extracted_path), 
                                force_rebuild=False, 
                                cache_content=False
                            )
                            
                            # Extract some key statistics from the report
                            lines = context_report.split('\n')
                            for line in lines[:10]:  # Show first 10 lines of report
                                if any(keyword in line.lower() for keyword in ['files', 'directories', 'languages', 'size']):
                                    print(f"    📊 {line.strip()}")
                            
                        else:
                            print("  ⚠️  Repository context loading failed")
                            
                    finally:
                        os.chdir(original_cwd)
                    
                else:
                    print(f"  ❌ Repository extraction failed or directory is empty: {extracted_path}")
                    
            else:
                print("  ❌ ZIP download failed")
                
    except Exception as e:
        print(f"  ❌ Error during VS Code repository test: {e}")
        import traceback
        traceback.print_exc()

def test_github_url_parsing():
    """Test GitHub URL parsing functionality."""
    print("\n🧪 Testing GitHub URL parsing...")
    
    test_cases = [
        ("https://github.com/microsoft/vscode", {"owner": "microsoft", "repo": "vscode", "branch": "main"}),
        ("https://github.com/microsoft/vscode.git", {"owner": "microsoft", "repo": "vscode", "branch": "main"}),
        ("https://github.com/microsoft/vscode/tree/main", {"owner": "microsoft", "repo": "vscode", "branch": "main"}),
        ("https://github.com/microsoft/vscode/tree/release/1.80", {"owner": "microsoft", "repo": "vscode", "branch": "release/1.80"}),
        ("https://github.com/octocat/Hello-World", {"owner": "octocat", "repo": "Hello-World", "branch": "main"}),
    ]
    
    for url, expected in test_cases:
        try:
            result = parse_github_url(url)
            status = "✅" if result == expected else "❌"
            print(f"  {status} URL: {url}")
            if result != expected:
                print(f"    Expected: {expected}")
                print(f"    Got:      {result}")
        except Exception as e:
            print(f"  ❌ Error parsing URL {url}: {e}")

def test_small_repository_full_workflow():
    """Test the complete workflow with a small repository."""
    print("\n🧪 Testing complete workflow with small repository...")
    
    # Use a small, well-known repository
    test_url = "https://github.com/octocat/Hello-World"
    agent_name = "deepcoder"
    
    try:
        print(f"  🔄 Testing complete workflow with {test_url}")
        
        # Create temporary data directory
        with tempfile.TemporaryDirectory() as temp_data_dir:
            print(f"  📁 Using temporary data directory: {temp_data_dir}")
            
            # Test the full validation and cloning process
            validation_passed, working_dir = validate_repository_requirement(
                agent_name, 
                ".", 
                test_url, 
                temp_data_dir
            )
            
            if validation_passed:
                print(f"  ✅ Repository validation passed. Working directory: {working_dir}")
                
                # Verify the repository was downloaded correctly
                working_path = Path(working_dir)
                if working_path.exists():
                    files = list(working_path.iterdir())
                    print(f"  📁 Repository contains {len(files)} items")
                    print(f"  📄 Items: {[f.name for f in files[:5]]}")  # Show first 5 items
                    
                    # Test that we can read a common file
                    readme_path = working_path / "README"
                    if readme_path.exists():
                        print("  ✅ Found README file")
                        try:
                            with open(readme_path, 'r', encoding='utf-8') as f:
                                content = f.read()[:100]  # First 100 chars
                            print(f"  📖 README preview: {content}...")
                        except Exception as e:
                            print(f"  ⚠️  Could not read README: {e}")
                    
                else:
                    print(f"  ❌ Working directory not found: {working_path}")
            else:
                print("  ❌ Repository validation failed")
                
    except Exception as e:
        print(f"  ❌ Error during complete workflow test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Testing Ollama Agent Repository Integration\n")
    
    test_agent_coding_detection()
    test_github_url_parsing()
    test_repository_cloning()
    test_small_repository_full_workflow()
    test_vscode_repository_zip_download()
    test_command_line_scenarios()
    
    print("\n✅ All tests completed!")
