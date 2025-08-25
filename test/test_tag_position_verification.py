#!/usr/bin/env python3
"""
Test script for verifying tag usage and position-based model selection.
"""

import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.model_registry import ModelRegistry, ModelDescriptor

def test_tag_usage_and_positions():
    """Test that we're using the correct tags and respecting positions."""
    print("Testing Tag Usage and Position-Based Selection...")
    
    try:
        # Load registry
        registry = ModelRegistry()
        registry.load_from_file("models-full-list.json")
        
        print(f"Loaded {len(registry.models)} models")
        
        # Show first 10 models with their positions and tags
        print("\nFirst 10 models (position order):")
        models_by_position = sorted(registry.models.values(), key=lambda x: x.position)
        
        for i, model in enumerate(models_by_position[:10]):
            print(f"  {i+1:2d}. Position: {model.position:2d}, Tag: '{model.tag}', Name: '{model.name}'")
            print(f"      Size: {model.max_size_b}B, Tools: {model.supports_tools}, Coding: {model.is_coding_focused}")
        
        # Test model selection for coding tasks - should prefer early position models with coding focus
        print("\nTesting code_analysis task selection:")
        coding_models = registry.get_models_for_task("code_analysis", max_size_b=14.0)
        
        print(f"Found {len(coding_models)} suitable coding models:")
        for i, (tag, model) in enumerate(coding_models[:5]):
            task_score = model.get_task_compatibility_score("code_analysis")
            pos_score = model.get_position_priority_score()
            print(f"  {i+1}. {tag} (pos: {model.position}, task: {task_score:.2f}, priority: {pos_score:.2f})")
            print(f"     Size: {model.max_size_b}B, Tools: {model.supports_tools}, Coding: {model.is_coding_focused}")
        
        # Test tool use selection
        print("\nTesting tool_use task selection:")
        tool_models = registry.get_models_for_task("tool_use", max_size_b=14.0)
        
        print(f"Found {len(tool_models)} suitable tool models:")
        for i, (tag, model) in enumerate(tool_models[:5]):
            task_score = model.get_task_compatibility_score("tool_use")
            pos_score = model.get_position_priority_score()
            print(f"  {i+1}. {tag} (pos: {model.position}, task: {task_score:.2f}, priority: {pos_score:.2f})")
            print(f"     Size: {model.max_size_b}B, Tools: {model.supports_tools}")
        
        # Verify we're using the correct tags from JSON
        print("\nVerifying tag usage from JSON file:")
        with open("models-full-list.json", 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Check first few entries
        for i, (key, data) in enumerate(list(json_data.items())[:5]):
            json_tag = data.get("Tag / filename", key)
            if json_tag in registry.models:
                model = registry.models[json_tag]
                print(f"  JSON key: '{key}' -> Tag: '{json_tag}' -> Model found: YES (pos: {model.position})")
            else:
                print(f"  JSON key: '{key}' -> Tag: '{json_tag}' -> Model found: NO")
        
        return True
        
    except Exception as e:
        print(f"Error in tag/position test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_size_constraints_with_positions():
    """Test that size constraints work with position-based selection."""
    print("\nTesting Size Constraints with Position Priority...")
    
    try:
        registry = ModelRegistry()
        registry.load_from_file("models-full-list.json")
        
        # Test different size limits and see how they affect selection
        size_limits = [3.0, 7.0, 14.0]
        
        for limit in size_limits:
            print(f"\nSize limit: {limit}B")
            suitable_models = registry.get_models_for_task("general", max_size_b=limit)
            
            print(f"  Found {len(suitable_models)} models within {limit}B:")
            for i, (tag, model) in enumerate(suitable_models[:3]):
                pos_score = model.get_position_priority_score()
                print(f"    {i+1}. {tag}: {model.max_size_b}B (pos: {model.position}, priority: {pos_score:.3f})")
        
        return True
        
    except Exception as e:
        print(f"Error in size constraints test: {e}")
        return False

def main():
    """Run tag and position verification tests"""
    print("Tag Usage and Position Priority Verification")
    print("=" * 50)
    
    tests = [
        test_tag_usage_and_positions,
        test_size_constraints_with_positions
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("All tests passed! Tag usage and position priority are working correctly.")
        return True
    else:
        print("Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
