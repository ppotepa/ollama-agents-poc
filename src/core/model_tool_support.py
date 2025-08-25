"""
Tool Support Testing Module for Ollama Models
============================================

This module provides functions to test if a model supports tools.
"""

import os
import json
import tempfile
import subprocess
from typing import Dict, Any, Optional
from pathlib import Path
from src.utils.enhanced_logging import get_logger

# Cache file for tool support test results
CACHE_FILE = Path(os.path.join(os.path.dirname(__file__), "..", "config", "tool_support_cache.json"))

def load_tool_support_cache() -> Dict[str, bool]:
    """Load the tool support cache from disk."""
    if not CACHE_FILE.exists():
        return {}
        
    try:
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger = get_logger()
        logger.warning(f"Failed to load tool support cache: {e}")
        return {}
        
def save_tool_support_cache(cache: Dict[str, bool]) -> None:
    """Save the tool support cache to disk."""
    try:
        # Ensure directory exists
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger = get_logger()
        logger.warning(f"Failed to save tool support cache: {e}")

def test_model_tool_support(model_name: str, force_recheck: bool = False) -> bool:
    """Test if a model supports tools.
    
    Args:
        model_name: The name of the model to test
        force_recheck: Whether to force recheck even if cached
        
    Returns:
        bool: Whether the model supports tools
    """
    logger = get_logger()
    
    # Check cache first unless forced to recheck
    if not force_recheck:
        cache = load_tool_support_cache()
        if model_name in cache:
            logger.debug(f"Using cached tool support result for {model_name}: {cache[model_name]}")
            return cache[model_name]
    
    # Hard-coded known tool support status for common models
    known_tool_support = {
        # Known to support tools
        "llama3.1": True,
        "llama3:8b": True,
        "qwen2.5": True,
        "qwen2.5-coder": True,
        "qwen2.5:7b": True,
        "mistral": True,
        "mistral:7b": True,
        "phi3:mini": True,
        "phi3:small": True,
        "granite3.3": True,
        "hermes3:8b": True,
        
        # Known NOT to support tools
        "phi3:medium": False,  # Specific versions may vary
        "llava": False,        # Vision models often lack tool support
    }
    
    # Check against known list
    for prefix, supports in known_tool_support.items():
        if model_name.startswith(prefix):
            logger.debug(f"Using known tool support status for {model_name}: {supports}")
            
            # Cache the result
            cache = load_tool_support_cache()
            cache[model_name] = supports
            save_tool_support_cache(cache)
            
            return supports
    
    # Need to actually test the model
    logger.info(f"Testing tool support for model {model_name}...")
    
    # Use Ollama API to test if model supports tools
    supports_tools = _test_model_with_ollama_api(model_name)
    
    # Cache the result
    cache = load_tool_support_cache()
    cache[model_name] = supports_tools
    save_tool_support_cache(cache)
    
    logger.info(f"Model {model_name} tool support test result: {supports_tools}")
    return supports_tools
    
def _test_model_with_ollama_api(model_name: str) -> bool:
    """Test tool support using Ollama API.
    
    This creates a minimal test tool and tries to use it with the model.
    """
    logger = get_logger()
    
    try:
        # Create a minimal JSON test prompt with a tool
        test_prompt = {
            "model": model_name,
            "prompt": "What's the current date? Use the tool.",
            "format": "json",
            "options": {"temperature": 0.0},
            "tools": [{
                "name": "get_current_date",
                "description": "Get the current date",
                "input_schema": {}
            }]
        }
        
        # Use a temporary file for the request
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp:
            json.dump(test_prompt, tmp)
            tmp_path = tmp.name
        
        # Run the test with curl to avoid dependencies
        cmd = [
            "curl", "-s", "http://localhost:11434/api/chat",
            "-d", f"@{tmp_path}"
        ]
        
        try:
            # Run with a short timeout
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=10  # 10 second timeout
            )
            
            # Check if there was an error message about tool support
            error_keywords = [
                "does not support tools",
                "tools not supported",
                "tool support is not available"
            ]
            
            for keyword in error_keywords:
                if keyword in result.stdout.lower() or keyword in result.stderr.lower():
                    logger.debug(f"Model {model_name} does not support tools (detected from error)")
                    return False
                    
            # If we get a valid JSON response with tool_calls, it supports tools
            if '"tool_calls":' in result.stdout:
                logger.debug(f"Model {model_name} supports tools (detected tool_calls)")
                return True
                
            # Default to False if we can't confirm tool support
            logger.debug(f"Model {model_name} likely doesn't support tools (no tool_calls detected)")
            return False
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Tool support test for {model_name} timed out")
            return False
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Error testing tool support for {model_name}: {e}")
        return False

def get_verified_tool_supporting_models() -> list[str]:
    """
    Get a list of models that have been verified to support tools.
    
    Returns:
        list: A list of model names that have been verified to support tools.
    """
    cache = load_tool_support_cache()
    return [model for model, supports in cache.items() if supports]
    