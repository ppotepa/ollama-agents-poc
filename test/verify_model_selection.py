"""Verify that our model selection properly identifies models that support tools."""

import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.model_capability_checker import ModelCapabilityChecker
from src.core.model_registry import ModelRegistry, ModelDescriptor
from src.core.model_tool_support import test_model_tool_support, _test_model_with_ollama_api
from src.utils.enhanced_logging import get_logger

logger = get_logger()

def test_tool_support_for_common_models():
    """Test tool support for common models that may be used in the system."""
    print("\n=== Testing tool support for common models ===")
    
    # Models that are typically used for coding/tools
    models_to_test = [
        "qwen2.5-coder:7b",
        "phi3:small",
        "llama3:8b",
        "mistral:7b",
        "phi3:mini",
        "codestral:latest",
        "deepseek-coder:6.7b",
        "deepcoder:14b"
    ]
    
    results = {}
    supports_tools = []
    
    for model in models_to_test:
        print(f"Testing {model}...")
        try:
            supports = test_model_tool_support(model)
            results[model] = supports
            if supports:
                supports_tools.append(model)
            print(f"  - {model}: {'✓ Supports tools' if supports else '✗ Does NOT support tools'}")
        except Exception as e:
            print(f"  - {model}: Error testing - {str(e)}")
            results[model] = False
    
    print("\n=== Summary of tool support ===")
    print(f"Models that support tools ({len(supports_tools)}/{len(models_to_test)}):")
    for model in supports_tools:
        print(f"  - {model}")
    
    return supports_tools

def test_model_selection_with_tools_requirement():
    """Test that model selection correctly handles tool requirements."""
    print("\n=== Testing model selection with tool requirements ===")
    
    # Create the capability checker
    checker = ModelCapabilityChecker(auto_update=True, max_model_size_b=14.0)
    
    # Test getting the best model for a task that requires tools
    print("\nSelecting best model for coding task (requires tools)...")
    coding_model = checker.get_best_model_for_task("coding", requires_tools=True)
    
    if coding_model:
        print(f"Selected model: {coding_model}")
        
        # Verify the model supports tools
        supports_tools = test_model_tool_support(coding_model)
        print(f"Tool support verified: {'✓ YES' if supports_tools else '✗ NO'}")
        
        # This should be true if our fix works correctly
        if not supports_tools:
            print("ERROR: Selected model does not support tools!")
    else:
        print("No suitable model found for coding task.")
    
    return coding_model

def main():
    """Run the verification tests."""
    print("Starting model selection verification...")
    
    # First test which models actually support tools
    tool_supporting_models = test_tool_support_for_common_models()
    
    # Then test that model selection respects tool requirements
    selected_model = test_model_selection_with_tools_requirement()
    
    # Verify that the selected model is in our list of confirmed tool-supporting models
    if selected_model in tool_supporting_models:
        print("\n✅ SUCCESS: Model selection is working correctly!")
    else:
        print("\n❌ FAILURE: Model selection is not choosing a verified tool-supporting model")

if __name__ == "__main__":
    main()
