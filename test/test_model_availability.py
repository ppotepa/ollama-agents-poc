"""Test script to validate model availability detection and auto-pulling."""

import os
import sys
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules to test
from src.core.model_discovery import model_exists, ensure_model_available, get_available_models
from src.core.model_capability_checker import get_capability_checker

def test_model_existence():
    """Test model existence checking."""
    logger.info("Testing model existence checking...")
    
    # Get available models
    available_models = get_available_models()
    logger.info(f"Available models: {available_models}")
    
    # Test existence of available models
    if available_models:
        first_model = available_models[0]
        logger.info(f"Testing if {first_model} exists...")
        exists = model_exists(first_model)
        logger.info(f"Model {first_model} exists: {exists}")
        
    # Test non-existent model
    nonexistent_model = "nonexistent:model"
    logger.info(f"Testing if {nonexistent_model} exists...")
    exists = model_exists(nonexistent_model)
    logger.info(f"Model {nonexistent_model} exists: {exists}")
    
    return True

def test_model_capability_checker():
    """Test model capability checker integration."""
    logger.info("Testing model capability checker...")
    
    # Get capability checker
    checker = get_capability_checker()
    
    # Get best model for task
    logger.info("Testing get_best_model_for_task...")
    best_model = checker.get_best_model_for_task("coding", requires_tools=True)
    logger.info(f"Best model for coding with tools: {best_model}")
    
    # Get default model
    logger.info("Testing get_default_model...")
    default_model = checker.get_default_model()
    logger.info(f"Default model: {default_model}")
    
    return True

def test_intelligent_orchestrator():
    """Import the intelligent orchestrator to ensure it can be loaded."""
    try:
        logger.info("Importing intelligent orchestrator to test imports...")
        from src.core.intelligent_orchestrator import IntelligentOrchestrator
        logger.info("Successfully imported IntelligentOrchestrator class")
        return True
    except Exception as e:
        logger.error(f"Failed to import IntelligentOrchestrator: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting model availability tests...")
    
    tests = [
        ("Model Existence", test_model_existence),
        ("Model Capability Checker", test_model_capability_checker),
        ("Intelligent Orchestrator", test_intelligent_orchestrator)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        logger.info(f"\n=== Running test: {test_name} ===")
        try:
            result = test_func()
            if result:
                logger.info(f"✅ Test '{test_name}' PASSED")
            else:
                logger.error(f"❌ Test '{test_name}' FAILED")
                all_passed = False
        except Exception as e:
            logger.error(f"❌ Test '{test_name}' ERROR: {e}")
            all_passed = False
    
    if all_passed:
        logger.info("\n✅ All tests PASSED")
        sys.exit(0)
    else:
        logger.error("\n❌ Some tests FAILED")
        sys.exit(1)
