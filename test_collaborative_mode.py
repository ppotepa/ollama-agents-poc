#!/usr/bin/env python3
"""Test script for collaborative mode functionality."""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_collaborative_system():
    """Test the collaborative system creation and basic functionality."""
    try:
        from src.core.collaborative_system import create_collaborative_system, CollaborativeAgentSystem
        from src.agents.base.mock_agent import MockAgent
        
        print("🧪 Testing collaborative system creation...")
        
        # Create a mock agent for testing
        mock_agent = MockAgent("test-agent")
        
        # Create collaborative system
        system = create_collaborative_system(mock_agent, max_iterations=3)
        print(f"✅ Collaborative system created: {type(system).__name__}")
        
        # Test basic functionality
        if hasattr(system, 'collaborative_execution'):
            print("✅ Collaborative execution method exists")
        else:
            print("❌ Collaborative execution method missing")
            
        print("🧪 Testing collaborative execution...")
        results = system.collaborative_execution(
            query="List files in current directory",
            working_directory=os.getcwd(),
            max_steps=2
        )
        
        print(f"📊 Results: {results}")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_operations():
    """Test the enhanced file operations."""
    try:
        from src.tools.file_ops import list_files
        
        print("🧪 Testing enhanced file operations...")
        
        # Test basic file listing
        files = list_files(".", detailed=False)
        print(f"✅ Basic file listing: {len(files)} files")
        
        # Test detailed file listing
        detailed_files = list_files(".", detailed=True)
        print(f"✅ Detailed file listing: {len(detailed_files)} files with metadata")
        
        if len(detailed_files) > 0:
            print(f"📁 Sample detailed entry: {detailed_files[0][:100]}...")
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_integration():
    """Test CLI argument parsing for collaborative mode."""
    try:
        from src.config.arguments import get_parser
        
        print("🧪 Testing CLI argument parsing...")
        
        parser = get_parser()
        
        # Test collaborative mode arguments
        args = parser.parse_args(['--collaborative', '--max-iterations', '10', '--query', 'test query'])
        
        if hasattr(args, 'collaborative') and args.collaborative:
            print("✅ Collaborative mode argument parsed correctly")
        else:
            print("❌ Collaborative mode argument not found")
            return False
            
        if hasattr(args, 'max_iterations') and args.max_iterations == 10:
            print("✅ Max iterations argument parsed correctly")
        else:
            print("❌ Max iterations argument not found or incorrect")
            return False
            
        print(f"📋 Parsed args: collaborative={args.collaborative}, max_iterations={args.max_iterations}")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting collaborative mode tests...")
    print("=" * 60)
    
    tests = [
        ("File Operations", test_file_operations),
        ("CLI Integration", test_cli_integration),
        ("Collaborative System", test_collaborative_system),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        if test_func():
            print(f"✅ {test_name} test PASSED")
            passed += 1
        else:
            print(f"❌ {test_name} test FAILED")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    if passed == total:
        print("🎉 All tests passed! Collaborative mode is ready.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
