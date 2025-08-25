"""Test that model selection properly filters models without tool support."""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import json

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.model_capability_checker import ModelCapabilityChecker
from src.core.model_registry import ModelRegistry, ModelDescriptor


class TestModelToolSelection(unittest.TestCase):
    """Test that model selection properly filters models that don't support tools."""

    def setUp(self):
        # Mock the tool support testing function
        self.tool_support_patcher = patch('src.core.model_tool_support.test_model_tool_support')
        self.mock_test_tool_support = self.tool_support_patcher.start()
        
        # Configure the mock to return True only for specific models
        def mock_tool_support(model_name):
            # Only these models actually support tools
            tool_models = ["llama3:8b", "phi3:small", "qwen2.5-coder:7b"]
            return model_name in tool_models
        
        self.mock_test_tool_support.side_effect = mock_tool_support
        
        # Mock model existence check
        self.model_exists_patcher = patch('src.core.model_discovery.model_exists')
        self.mock_model_exists = self.model_exists_patcher.start()
        self.mock_model_exists.return_value = True
    
    def tearDown(self):
        # Stop patchers
        self.tool_support_patcher.stop()
        self.model_exists_patcher.stop()
    
    def test_get_best_model_fallback_with_tools(self):
        """Test that _get_best_model_fallback correctly filters models by tool support."""
        # Create a test configuration with various models
        test_config = {
            "llama3:8b": {"supports_tools": True, "capabilities": ["coding"], "size_b": 8.0},
            "phi3:small": {"supports_tools": True, "capabilities": ["coding"], "size_b": 3.8},
            "phi3:mini": {"supports_tools": True, "capabilities": ["coding"], "size_b": 2.0},
            "qwen2.5-coder:7b": {"supports_tools": True, "capabilities": ["coding"], "size_b": 7.0},
            "mistral:7b": {"supports_tools": True, "capabilities": ["coding"], "size_b": 7.0}
        }
        
        # Create the capability checker with our test config
        checker = ModelCapabilityChecker(auto_update=False, max_model_size_b=14.0)
        checker.models_config = test_config
        
        # Configure the model_exists mock to return True for these specific models
        discovery_patcher = patch('src.core.model_discovery.OllamaModelDiscovery')
        mock_discovery_class = discovery_patcher.start()
        mock_discovery = MagicMock()
        mock_discovery_class.return_value = mock_discovery
        mock_discovery.model_exists.return_value = True
        
        try:
            # Test with tool requirements
            model = checker._get_best_model_fallback("coding", requires_tools=True)
            
            # Should select the largest model that actually supports tools
            self.assertEqual(model, "llama3:8b", 
                            "Should select the largest model that actually supports tools")
        finally:
            discovery_patcher.stop()
    
    def test_get_tool_supporting_models(self):
        """Test that get_tool_supporting_models only returns models with verified tool support."""
        # Create a test configuration with various models
        test_config = {
            "llama3:8b": {"supports_tools": True, "capabilities": ["coding"], "size_b": 8.0},
            "phi3:small": {"supports_tools": True, "capabilities": ["coding"], "size_b": 3.8},
            "phi3:mini": {"supports_tools": True, "capabilities": ["coding"], "size_b": 2.0},
            "qwen2.5-coder:7b": {"supports_tools": True, "capabilities": ["coding"], "size_b": 7.0},
            "mistral:7b": {"supports_tools": True, "capabilities": ["coding"], "size_b": 7.0}
        }
        
        # Create the capability checker with our test config
        checker = ModelCapabilityChecker(auto_update=False, max_model_size_b=14.0)
        checker.models_config = test_config
        
        # Get models that support tools
        tool_models = checker.get_tool_supporting_models()
        
        # Should only include models that actually support tools
        expected_models = ["llama3:8b", "phi3:small", "qwen2.5-coder:7b"]
        self.assertEqual(sorted(tool_models), sorted(expected_models),
                         "Should only include models that actually support tools")
    
    def test_supports_tools_verification(self):
        """Test that supports_tools method verifies tool support properly."""
        # Create a test configuration
        test_config = {
            "llama3:8b": {"supports_tools": True, "capabilities": ["coding"], "size_b": 8.0},
            "phi3:mini": {"supports_tools": True, "capabilities": ["coding"], "size_b": 2.0},
            "no-tools-model:small": {"supports_tools": False, "capabilities": ["coding"], "size_b": 7.0}
        }
        
        # Create the capability checker with our test config
        checker = ModelCapabilityChecker(auto_update=False, max_model_size_b=14.0)
        checker.models_config = test_config
        
        # Test model that passes verification
        self.assertTrue(checker.supports_tools("llama3:8b"), 
                       "Should return True for model that passes verification")
        
        # Test model that claims support but fails verification
        self.assertFalse(checker.supports_tools("phi3:mini"),
                        "Should return False for model that claims support but fails verification")
        
        # Test model that doesn't claim support
        self.assertFalse(checker.supports_tools("no-tools-model:small"),
                        "Should return False for model that doesn't claim support")
    
    def test_get_default_model_tool_verification(self):
        """Test that get_default_model correctly filters by tool support."""
        # Create a test configuration
        test_config = {
            "llama3:8b": {"supports_tools": True, "capabilities": ["coding"], "size_b": 8.0},
            "phi3:small": {"supports_tools": True, "capabilities": ["coding"], "size_b": 3.8},
            "phi3:mini": {"supports_tools": True, "capabilities": ["coding"], "size_b": 2.0},
            "mistral:7b": {"supports_tools": True, "capabilities": ["coding"], "size_b": 7.0}
        }
        
        # Create the capability checker with our test config
        checker = ModelCapabilityChecker(auto_update=False, max_model_size_b=14.0)
        checker.models_config = test_config
        
        # Test default model selection
        model = checker.get_default_model()
        
        # Should select a model that actually supports tools
        self.assertIn(model, ["llama3:8b", "phi3:small", "qwen2.5-coder:7b"], 
                     "Should select a model that actually supports tools")


if __name__ == "__main__":
    unittest.main()
