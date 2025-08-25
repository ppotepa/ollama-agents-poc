"""Verify that the fix for tool support verification works correctly."""

import os
import sys
import time
import json
from typing import Dict, Any, List

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.model_capability_checker import ModelCapabilityChecker
from src.core.model_registry import ModelRegistry, ModelDescriptor
from src.core.model_tool_support import test_model_tool_support


def main():
    """Run verification of the tool support fix."""
    print("Starting tool support verification...")
    
    # Create the model capability checker
    checker = ModelCapabilityChecker(auto_update=True, max_model_size_b=14.0)
    
    # Test getting models that support tools
    print("\n=== Models that claim to support tools ===")
    claimed_tool_models = []
    for model_name, config in checker.models_config.items():
        if config.get('supports_tools', False):
            claimed_tool_models.append(model_name)
    print(f"Found {len(claimed_tool_models)} models that claim tool support in metadata")
    
    # Test verifying tool support for a sample of models
    print("\n=== Tool support verification ===")
    sample_size = min(5, len(claimed_tool_models))
    sample_models = claimed_tool_models[:sample_size]
    
    verified_tool_models = []
    for model_name in sample_models:
        print(f"Testing {model_name}...")
        supports_tools = test_model_tool_support(model_name)
        print(f"  - {model_name}: {'✓ Supports tools' if supports_tools else '✗ Does NOT support tools'}")
        if supports_tools:
            verified_tool_models.append(model_name)
    
    print(f"\nOut of {sample_size} models tested, {len(verified_tool_models)} actually support tools")
    
    # Test getting best model for coding task (requires tools)
    print("\n=== Best model for coding (requires tools) ===")
    best_model = checker.get_best_model_for_task("coding", requires_tools=True)
    print(f"Best model selected: {best_model}")
    
    if best_model:
        # Verify the selected model actually supports tools
        supports_tools = test_model_tool_support(best_model)
        print(f"Selected model tool support verification: {'✓ Supports tools' if supports_tools else '✗ Does NOT support tools'}")
        
        # This should be true if our fix works correctly
        assert supports_tools, f"Selected model {best_model} should support tools!"
    
    print("\nTool support verification completed successfully!")


if __name__ == "__main__":
    main()
