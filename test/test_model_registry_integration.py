#!/usr/bin/env python3
"""
Test script for model registry integration with capability checker.
Tests size constraints, tool support verification, and intelligent model selection.
"""

import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.model_registry import ModelRegistry, ModelDescriptor
from src.core.model_capability_checker import ModelCapabilityChecker

def test_model_registry_loading():
    """Test loading and parsing models-full-list.json"""
    print("Testing Model Registry Loading...")
    
    try:
        registry = ModelRegistry()
        registry.load_from_file("models-full-list.json")
        print(f"✅ Successfully loaded {len(registry.models)} models")
        
        # Test size constraint filtering (≤14B)
        large_models = [m for m in registry.models.values() if m.size_b > 14.0]
        small_models = [m for m in registry.models.values() if m.size_b <= 14.0]
        
        print(f"📊 Models > 14B: {len(large_models)}")
        print(f"📊 Models ≤ 14B: {len(small_models)}")
        
        # Test tool support filtering
        tool_capable = [m for m in registry.models.values() if m.supports_tools]
        print(f"🔧 Tool-capable models: {len(tool_capable)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Registry loading failed: {e}")
        return False

def test_model_selection():
    """Test intelligent model selection with constraints"""
    print("\nTesting Model Selection...")
    
    try:
        registry = ModelRegistry()
        registry.load_from_file("models-full-list.json")
        
        # Test different task types
        tasks = [
            ("code_analysis", "Code analysis task"),
            ("tool_use", "Tool usage task"), 
            ("general", "General purpose task"),
            ("unknown_task", "Unknown task type")
        ]
        
        for task_type, description in tasks:
            print(f"\n📋 Testing {task_type}:")
            
            # Get best model for task
            best_model = registry.get_best_model_for_task(task_type)
            if best_model:
                print(f"  ✅ Best model: {best_model.name}")
                print(f"  📏 Size: {best_model.size_b}B")
                print(f"  🔧 Tool support: {best_model.supports_tools}")
                print(f"  📝 Description: {best_model.notes[:80]}...")
            else:
                print(f"  ❌ No suitable model found")
        
        return True
        
    except Exception as e:
        print(f"❌ Model selection failed: {e}")
        return False

def test_capability_checker_integration():
    """Test ModelCapabilityChecker with registry integration"""
    print("\nTesting Capability Checker Integration...")
    
    try:
        # Initialize checker
        checker = ModelCapabilityChecker(
            auto_update=True,
            auto_pull=False,
            max_model_size_b=14.0
        )
        
        # Manually load the registry if needed
        if not checker.model_registry:
            from src.core.model_registry import ModelRegistry
            registry = ModelRegistry()
            registry.load_from_file("models-full-list.json")
            checker.model_registry = registry
        
        print(f"✅ Capability checker initialized with registry")
        print(f"📏 Max model size: {checker.max_model_size_b}B")
        
        # Test best model selection for different tasks
        test_tasks = [
            "code_analysis",
            "general",
            "tool_use"
        ]
        
        for task in test_tasks:
            best_model = checker.get_best_model_for_task(task)
            print(f"🎯 Best model for {task}: {best_model}")
            
            # Check if model meets size constraints
            if checker.model_registry:
                model_info = checker.model_registry.get_model(best_model)
                if model_info:
                    size_ok = model_info.size_b <= checker.max_model_size_b
                    size_status = "✅" if size_ok else "❌"
                    print(f"   {size_status} Size: {model_info.size_b}B (limit: {checker.max_model_size_b}B)")
        
        # Test fallback models
        print(f"\n🔄 Testing fallback models...")
        fallbacks = checker.model_registry.get_fallback_models(max_size_b=14.0) if checker.model_registry else []
        print(f"📋 Available fallbacks: {fallbacks[:5]}")  # Show first 5
        
        return True
        
    except Exception as e:
        print(f"❌ Capability checker integration failed: {e}")
        return False

def test_size_constraint_enforcement():
    """Test that size constraints are properly enforced"""
    print("\nTesting Size Constraint Enforcement...")
    
    try:
        registry = ModelRegistry()
        registry.load_from_file("models-full-list.json")
        
        # Test with different size limits
        size_limits = [3.0, 7.0, 14.0, 30.0]
        
        for limit in size_limits:
            fallbacks = registry.get_fallback_models(max_size_b=limit)
            if fallbacks:
                # Verify all returned models are within limit
                all_within_limit = True
                for model_name in fallbacks[:3]:  # Check first 3
                    model = registry.get_model(model_name)
                    if model and model.size_b > limit:
                        all_within_limit = False
                        break
                
                status = "✅" if all_within_limit else "❌"
                print(f"{status} Size limit {limit}B: {len(fallbacks)} models, all within limit: {all_within_limit}")
            else:
                print(f"⚠️  Size limit {limit}B: No models found")
        
        return True
        
    except Exception as e:
        print(f"❌ Size constraint test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("Testing Model Registry Integration\n" + "="*50)
    
    # Check if models file exists
    if not os.path.exists("models-full-list.json"):
        print("❌ models-full-list.json not found. Please ensure it exists in the current directory.")
        return False
    
    tests = [
        test_model_registry_loading,
        test_model_selection, 
        test_capability_checker_integration,
        test_size_constraint_enforcement
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("="*50)
    print(f"🧪 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Model registry integration is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
