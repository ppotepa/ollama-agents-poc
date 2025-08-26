#!/usr/bin/env python3
"""Test script to verify repository handling."""

import os
import sys
import argparse
from pathlib import Path

# Add src to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.core.io.clone_operations import get_clone_directory, clone_repository, is_git_repository


def main():
    """Test repository handling."""
    parser = argparse.ArgumentParser(description='Test repository handling')
    parser.add_argument('-g', '--git-repo', dest='git_repo', help='Git repository URL or project name')
    args = parser.parse_args()

    if not args.git_repo:
        print("Please provide a repository URL or project name with -g/--git-repo")
        return

    # Get clone directory
    clone_dir = get_clone_directory(args.git_repo, "./data")
    print(f"🔍 Clone directory would be: {clone_dir}")
    
    # Check if directory already exists and is a git repo
    if clone_dir.exists():
        print(f"📁 Directory exists: {clone_dir}")
        if is_git_repository(str(clone_dir)):
            print(f"✅ Directory is already a git repository")
        else:
            print(f"❌ Directory exists but is not a git repository")

    # Ask for confirmation before proceeding
    response = input(f"Proceed with cloning/creating repository at {clone_dir}? (y/n): ")
    if response.lower() != 'y':
        print("Operation cancelled")
        return

    # Clone or create repository
    if clone_repository(args.git_repo, clone_dir):
        print(f"✅ Repository operation completed successfully")
    else:
        print(f"❌ Repository operation failed")


if __name__ == "__main__":
    main()
