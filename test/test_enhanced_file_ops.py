#!/usr/bin/env python3
"""Test script to demonstrate enhanced file operations."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.tools.file_ops import list_files, get_file_info, write_file, read_file

def main():
    """Test the enhanced file operations."""
    print("🧪 Testing Enhanced File Operations")
    print("=" * 60)
    
    print("\n1. 📁 DETAILED DIRECTORY LISTING (like dir command):")
    print("-" * 50)
    result = list_files(".", "*", detailed=True)
    print(result)
    
    print("\n\n2. 📄 DETAILED FILE INFORMATION:")
    print("-" * 50)
    # Test with main.py
    if os.path.exists("main.py"):
        info = get_file_info("main.py")
        print(info)
    else:
        print("main.py not found")
    
    print("\n\n3. 📝 ENHANCED FILE CREATION:")
    print("-" * 50)
    test_content = """# Test File
This is a test file created by the enhanced file operations.
It demonstrates the detailed feedback provided by the new system.

Features:
- File size information
- Creation timestamps
- Detailed feedback
"""
    result = write_file("test_enhanced_file.txt", test_content)
    print(result)
    
    print("\n\n4. 📖 ENHANCED FILE READING:")
    print("-" * 50)
    if os.path.exists("test_enhanced_file.txt"):
        content = read_file("test_enhanced_file.txt", show_info=True)
        print(content)
    
    print("\n\n5. 🎯 SIMPLE LISTING (original behavior):")
    print("-" * 50)
    simple_result = list_files(".", "*", detailed=False)
    print(simple_result)
    
    # Cleanup
    if os.path.exists("test_enhanced_file.txt"):
        os.remove("test_enhanced_file.txt")
        print("\n🧹 Cleaned up test file")

if __name__ == "__main__":
    main()
