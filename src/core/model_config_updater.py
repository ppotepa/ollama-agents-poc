"""Dynamic model configuration updater."""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, List
from src.core.model_discovery import get_available_models
from src.utils.enhanced_logging import get_logger


def update_models_config(force_refresh: bool = False) -> bool:
    """Update models.yaml with dynamically discovered models.
    
    Args:
        force_refresh: Force refresh discovery cache
        
    Returns:
        True if update was successful
    """
    logger = get_logger()
    
    try:
        # Discover available models
        logger.info("Discovering available models...")
        discovered_models = get_available_models(force_refresh=force_refresh)
        
        if not discovered_models:
            logger.error("No models discovered, keeping existing configuration")
            return False
        
        # Convert to models.yaml format
        models_config = {
            "models": {}
        }
        
        for model in discovered_models:
            model_id = model["model_id"]
            
            # Create configuration entry
            config_entry = {
                "size_b": model["size_b"],
                "type": model["type"],
                "capabilities": model["capabilities"],
                "supports_tools": model["supports_tools"],
                "description": model["description"],
                "strengths": model["strengths"],
                "use_cases": model["use_cases"],
                "auto_discovered": True  # Mark as auto-discovered
            }
            
            models_config["models"][model_id] = config_entry
        
        # Backup existing configuration
        config_path = Path("src/config/models.yaml")
        backup_path = Path("src/config/models.yaml.backup")
        
        if config_path.exists():
            import shutil
            shutil.copy2(config_path, backup_path)
            logger.info(f"Backed up existing configuration to {backup_path}")
        
        # Write new configuration
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(models_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Updated models.yaml with {len(discovered_models)} models")
        
        # Log summary
        tool_models = [m for m in discovered_models if m["supports_tools"]]
        non_tool_models = [m for m in discovered_models if not m["supports_tools"]]
        
        logger.info(f"Models with tool support: {len(tool_models)}")
        logger.info(f"Models without tool support: {len(non_tool_models)}")
        
        if non_tool_models:
            logger.info(f"Non-tool models: {[m['model_id'] for m in non_tool_models]}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to update models configuration: {e}")
        return False


def validate_models_config() -> bool:
    """Validate that the current models configuration matches available models.
    
    Returns:
        True if configuration is valid and up to date
    """
    logger = get_logger()
    
    try:
        # Load current configuration
        config_path = Path("src/config/models.yaml")
        if not config_path.exists():
            logger.warning("models.yaml not found, needs update")
            return False
        
        with open(config_path, 'r') as f:
            current_config = yaml.safe_load(f)
        
        configured_models = set(current_config.get("models", {}).keys())
        
        # Get available models
        available_models = get_available_models()
        available_model_ids = set(model["model_id"] for model in available_models)
        
        # Check for differences
        missing_in_config = available_model_ids - configured_models
        extra_in_config = configured_models - available_model_ids
        
        if missing_in_config:
            logger.warning(f"Models available but not in config: {missing_in_config}")
        
        if extra_in_config:
            logger.warning(f"Models in config but not available: {extra_in_config}")
        
        is_valid = not missing_in_config and not extra_in_config
        
        if is_valid:
            logger.info("Model configuration is up to date")
        else:
            logger.info("Model configuration needs update")
        
        return is_valid
        
    except Exception as e:
        logger.error(f"Failed to validate models configuration: {e}")
        return False


def get_model_status_report() -> Dict[str, Any]:
    """Get a comprehensive status report of model configuration.
    
    Returns:
        Dictionary with model status information
    """
    try:
        # Get available models
        available_models = get_available_models()
        
        # Categorize models
        tool_models = [m for m in available_models if m["supports_tools"]]
        non_tool_models = [m for m in available_models if not m["supports_tools"]]
        
        # Model type distribution
        type_counts = {}
        for model in available_models:
            model_type = model["type"]
            type_counts[model_type] = type_counts.get(model_type, 0) + 1
        
        # Size distribution
        size_ranges = {"small": 0, "medium": 0, "large": 0}
        for model in available_models:
            size_b = model["size_b"]
            if size_b <= 3:
                size_ranges["small"] += 1
            elif size_b <= 10:
                size_ranges["medium"] += 1
            else:
                size_ranges["large"] += 1
        
        return {
            "total_models": len(available_models),
            "tool_supporting": len(tool_models),
            "non_tool_supporting": len(non_tool_models),
            "tool_models": [m["model_id"] for m in tool_models],
            "non_tool_models": [m["model_id"] for m in non_tool_models],
            "type_distribution": type_counts,
            "size_distribution": size_ranges,
            "available_models": [m["model_id"] for m in available_models]
        }
        
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # CLI interface for model configuration management
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "update":
            force = "--force" in sys.argv
            success = update_models_config(force_refresh=force)
            print("Model configuration updated successfully" if success else "Failed to update configuration")
        
        elif command == "validate":
            is_valid = validate_models_config()
            print("Model configuration is valid" if is_valid else "Model configuration needs update")
        
        elif command == "status":
            report = get_model_status_report()
            print(json.dumps(report, indent=2))
        
        else:
            print("Usage: python model_config_updater.py [update|validate|status] [--force]")
    
    else:
        # Default: show status
        report = get_model_status_report()
        print(json.dumps(report, indent=2))
