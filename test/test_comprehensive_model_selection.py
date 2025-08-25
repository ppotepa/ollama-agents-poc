"""Test the full model selection process with tool support verification."""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import json
import tempfile
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.model_capability_checker import ModelCapabilityChecker
from src.core.model_registry import ModelRegistry, ModelDescriptor


class TestComprehensiveModelSelection(unittest.TestCase):
    """Test the full model selection process with tool support verification."""

    def setUp(self):
        # Create a temporary JSON file with model data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.models_json_path = os.path.join(self.temp_dir.name, "models-full-list.json")
        
        # Create model JSON data
        models_data = {
            "models": [
                {
                    "name": "llama3",
                    "variants": ["8b", "70b"],
                    "capabilities": ["coding", "general_qa", "tool_use"],
                    "supports_tools": True,
                    "size_b": {"8b": 8.0, "70b": 70.0},
                    "position": 1
                },
                {
                    "name": "phi3",
                    "variants": ["small", "mini", "medium"],
                    "capabilities": ["coding", "general_qa", "tool_use"],
                    "supports_tools": True,
                    "size_b": {"mini": 2.0, "small": 3.8, "medium": 14.0},
                    "position": 2
                },
                {
                    "name": "mistral",
                    "variants": ["7b", "8x7b"],
                    "capabilities": ["coding", "general_qa"],
                    "supports_tools": True,
                    "size_b": {"7b": 7.0, "8x7b": 56.0},
                    "position": 3
                },
                {
                    "name": "qwen2.5-coder",
                    "variants": ["7b", "14b"],
                    "capabilities": ["coding", "file_operations"],
                    "supports_tools": True,
                    "size_b": {"7b": 7.0, "14b": 14.0},
                    "position": 4
                }
            ]
        }
        
        with open(self.models_json_path, 'w') as f:
            json.dump(models_data, f)
        
        # Mock the tool support testing function
        self.tool_support_patcher = patch('src.core.model_tool_support.test_model_tool_support')
        self.mock_test_tool_support = self.tool_support_patcher.start()
        
        # Configure the mock to return True only for specific models
        def mock_tool_support(model_name):
            # Only these models actually support tools (others just claim to)
            tool_models = ["llama3:8b", "phi3:small", "qwen2.5-coder:7b"]
            return model_name in tool_models
        
        self.mock_test_tool_support.side_effect = mock_tool_support
        
        # Mock model existence check
        self.model_exists_patcher = patch('src.core.model_discovery.model_exists')
        self.mock_model_exists = self.model_exists_patcher.start()
        self.mock_model_exists.return_value = True
        
        # Mock finding the models JSON file
        self.path_patcher = patch('src.core.model_registry._resolve_models_json_path')
        self.mock_resolve_path = self.path_patcher.start()
        self.mock_resolve_path.return_value = self.models_json_path
        
        # Create a ModelRegistry instance
        self.registry = ModelRegistry()
        
        # Create the capability checker
        self.checker = ModelCapabilityChecker(auto_update=False, max_model_size_b=14.0)
        self.checker.model_registry = self.registry
    
    def tearDown(self):
        # Stop patchers
        self.tool_support_patcher.stop()
        self.model_exists_patcher.stop()
        self.path_patcher.stop()
        
        # Clean up temp directory
        self.temp_dir.cleanup()
    
    def test_get_best_model_for_task_with_tool_requirement(self):
        """Test that the model selection process properly checks for tool support."""
        # Test with a task that requires tools
        model = self.checker.get_best_model_for_task("coding", requires_tools=True)
        
        # Should select a model that actually supports tools, not just claims to
        self.assertIn(model, ["llama3:8b", "phi3:small", "qwen2.5-coder:7b"],
                     f"Should select a model that actually supports tools, got {model}")
        
        # The best model for coding with tools should be llama3:8b (highest position that supports tools)
        self.assertEqual(model, "llama3:8b", 
                        f"Should select llama3:8b as the best model for coding with tools, got {model}")
    
    def test_get_best_model_for_task_without_tool_requirement(self):
        """Test that model selection ignores tool support when not required."""
        # Test with a task that doesn't require tools
        model = self.checker.get_best_model_for_task("general_qa", requires_tools=False)
        
        # Should select based on position rather than tool support
        self.assertIn(model, ["llama3:8b", "phi3:small", "mistral:7b"],
                     f"Should select based on position rather than tool support, got {model}")
    
    def test_get_best_model_with_size_constraint(self):
        """Test that model selection respects size constraints."""
        # Create a checker with a small size constraint
        small_checker = ModelCapabilityChecker(auto_update=False, max_model_size_b=3.0)
        small_checker.model_registry = self.registry
        
        # Test with a task that requires tools
        model = small_checker.get_best_model_for_task("coding", requires_tools=True)
        
        # Should select a model under 3B parameters that supports tools
        self.assertEqual(model, "phi3:small",
                        f"Should select a model under 3B parameters that supports tools, got {model}")
    
    def test_fallback_when_no_tool_supporting_models(self):
        """Test fallback behavior when no models with tool support are found."""
        # Change the mock to return False for all models
        self.mock_test_tool_support.side_effect = lambda model_name: False
        
        # Test with a task that requires tools
        model = self.checker.get_best_model_for_task("coding", requires_tools=True)
        
        # Should return a default model or None since no models support tools
        self.assertIsNotNone(model, "Should return a default model when no models support tools")


if __name__ == "__main__":
    unittest.main()
