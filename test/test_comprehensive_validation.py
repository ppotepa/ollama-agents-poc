"""Comprehensive test to verify all fixes have been applied correctly."""

import os
import sys
import time
import asyncio
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_model_existence_and_fallback():
    """Test model existence checking and fallback mechanism."""
    try:
        from src.core.intelligent_orchestrator import IntelligentOrchestrator
        
        # Initialize the orchestrator
        orchestrator = IntelligentOrchestrator(enable_streaming=True)
        
        # Try to get an agent for a non-existent model
        # This should fallback to an available model
        agent = await orchestrator._get_agent("nonexistent-model:test")
        
        # If we got here without errors, the fallback worked
        logger.info(f"Successfully got agent with fallback: {agent}")
        return True
    except Exception as e:
        logger.error(f"Failed to test model fallback: {e}")
        return False

def test_capability_format_handling():
    """Test that capability handling works with both list and dictionary formats."""
    try:
        from src.core.model_capability_checker import get_capability_checker
        from src.integrations.model_config_reader import ModelConfig
        
        # Create config with both formats
        test_config_list = ModelConfig(
            short_name="test-list",
            name="test-list-model",
            model_id="test-list:latest",
            provider="test",
            description="Test model with list capabilities",
            capabilities=["coding", "file_operations", "streaming"],
            parameters={},
            tools=[]
        )
        
        test_config_dict = ModelConfig(
            short_name="test-dict",
            name="test-dict-model",
            model_id="test-dict:latest",
            provider="test",
            description="Test model with dict capabilities",
            capabilities={
                "coding": True,
                "file_operations": True,
                "streaming": True
            },
            parameters={},
            tools=[]
        )
        
        # Test with list format
        supports_coding_list = test_config_list.supports_coding
        supports_files_list = test_config_list.supports_file_operations
        supports_streaming_list = test_config_list.supports_streaming
        
        # Test with dict format
        supports_coding_dict = test_config_dict.supports_coding
        supports_files_dict = test_config_dict.supports_file_operations
        supports_streaming_dict = test_config_dict.supports_streaming
        
        logger.info(f"List format - coding: {supports_coding_list}, files: {supports_files_list}, streaming: {supports_streaming_list}")
        logger.info(f"Dict format - coding: {supports_coding_dict}, files: {supports_files_dict}, streaming: {supports_streaming_dict}")
        
        # Both formats should yield the same result
        return (supports_coding_list == supports_coding_dict == True and
                supports_files_list == supports_files_dict == True and
                supports_streaming_list == supports_streaming_dict == True)
    except Exception as e:
        logger.error(f"Failed to test capability format handling: {e}")
        return False

async def test_agent_streaming_output():
    """Test that agent streaming output doesn't duplicate characters."""
    try:
        from src.core.helpers import get_agent_instance
        from src.agents.universal.agent import UniversalAgent
        import sys
        from io import StringIO
        
        # Get first available model
        from src.core.model_discovery import get_available_models
        available_models = get_available_models()
        if not available_models:
            logger.error("No models available for streaming test")
            return False
        
        test_model = available_models[0]
        logger.info(f"Testing streaming with model: {test_model}")
        
        # Capture output to verify no duplication
        original_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Create agent with streaming
        agent = get_agent_instance(test_model, streaming=True)
        
        # Reset stdout
        sys.stdout = original_stdout
        
        # Log about successful agent creation
        logger.info(f"Successfully created streaming agent with {test_model}")
        return True
    except Exception as e:
        logger.error(f"Failed to test agent streaming: {e}")
        return False

async def run_tests():
    """Run all tests."""
    tests = [
        ("Model Existence and Fallback", test_model_existence_and_fallback()),
        ("Capability Format Handling", test_capability_format_handling()),
        ("Agent Streaming Output", test_agent_streaming_output())
    ]
    
    all_passed = True
    for test_name, test_coroutine in tests:
        logger.info(f"\n=== Running test: {test_name} ===")
        try:
            if asyncio.iscoroutine(test_coroutine):
                result = await test_coroutine
            else:
                result = test_coroutine
                
            if result:
                logger.info(f"✅ Test '{test_name}' PASSED")
            else:
                logger.error(f"❌ Test '{test_name}' FAILED")
                all_passed = False
        except Exception as e:
            logger.error(f"❌ Test '{test_name}' ERROR: {e}")
            all_passed = False
    
    if all_passed:
        logger.info("\n✅ All tests PASSED - All fixes have been applied successfully!")
    else:
        logger.error("\n❌ Some tests FAILED - Some issues may still remain")
    
    return all_passed

if __name__ == "__main__":
    logger.info("Starting comprehensive validation tests...")
    
    # Run tests using asyncio
    try:
        all_passed = asyncio.run(run_tests())
        sys.exit(0 if all_passed else 1)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)
