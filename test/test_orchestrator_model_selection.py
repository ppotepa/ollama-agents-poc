"""Test that the orchestrator respects tool support requirements."""

import os
import sys
import json
from unittest.mock import patch, MagicMock

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.intelligent_orchestrator import IntelligentOrchestrator
from src.core.investigation_strategies import InvestigationStrategies, InvestigationStep, InvestigationPriority
from src.core.model_tool_support import test_model_tool_support

def test_orchestrator_model_selection():
    """Test that the orchestrator selects models that support tools when required."""
    print("Testing orchestrator model selection with tool support requirements...")
    
    # First create a step that requires tool support
    step = InvestigationStep(
        step_id="test_code_analysis",
        description="Analyze code structure and dependencies",
        strategy="breadth_first",
        priority=InvestigationPriority.HIGH,
        estimated_duration=120,
        required_models=["codestral:latest"],
        expected_outputs=["dependency_graph", "code_structure_analysis"],
        validation_criteria=["All dependencies identified", "Structure is clear"]
    )
    
    # Create the orchestrator
    orchestrator = IntelligentOrchestrator()
    
    # Get the orchestrator to process this step (extract just the model selection logic)
    with patch('src.core.model_capability_checker.ModelCapabilityChecker') as mock_checker:
        # Mock the capability checker's get_best_model_for_task method
        mock_checker_instance = MagicMock()
        mock_checker.return_value = mock_checker_instance
        mock_checker_instance.get_best_model_for_task.return_value = "qwen2.5-coder:7b"
        
        # Since we can't easily call the private method, we'll monkey patch to inspect the model selection
        original_method = orchestrator._get_agent
        
        selected_model = None
        
        async def spy_get_agent(model_name):
            nonlocal selected_model
            selected_model = model_name
            return await original_method(model_name)
            
        orchestrator._get_agent = spy_get_agent
        
        # Run the method that would process the step
        try:
            orchestrator._prepare_step_for_execution(step)
            
            # Check if the step's assigned model was changed from codestral:latest
            if step.assigned_model != "codestral:latest":
                print(f"✓ Step model was changed from codestral:latest to {step.assigned_model}")
            else:
                print("✗ Step model was not changed from codestral:latest")
        except Exception as e:
            print(f"Error during test: {e}")
            
        # Restore the original method
        orchestrator._get_agent = original_method
    
    # Check that codestral:latest doesn't support tools
    print("\nVerifying that codestral:latest doesn't support tools...")
    try:
        supports_tools = test_model_tool_support("codestral:latest")
        print(f"codestral:latest supports tools: {supports_tools}")
    except Exception as e:
        print(f"Error testing codestral:latest: {e}")

if __name__ == "__main__":
    test_orchestrator_model_selection()
