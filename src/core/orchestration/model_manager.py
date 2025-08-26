"""Model and agent management for orchestration."""

from typing import Dict, Optional

from src.agents.universal.agent import UniversalAgent
from src.core.execution_planner import ExecutionPlan, ExecutionStep
from src.core.helpers import get_agent_instance
from src.core.model_capability_checker import get_capability_checker
from src.core.model_discovery import ensure_model_available, model_exists
from src.utils.enhanced_logging import get_logger

from .session_manager import OrchestrationSession


class ModelManager:
    """Manages models and agents for orchestration."""

    def __init__(self, enable_streaming: bool = True, max_model_switches: int = 10):
        """Initialize model manager.
        
        Args:
            enable_streaming: Whether to enable streaming mode
            max_model_switches: Maximum model switches per session
        """
        self.enable_streaming = enable_streaming
        self.max_model_switches = max_model_switches
        self.agent_cache: Dict[str, UniversalAgent] = {}
        self.capability_checker = get_capability_checker()
        self.logger = get_logger()

    async def get_agent(self, model_name: str) -> UniversalAgent:
        """Get or create an agent for the specified model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Agent instance
            
        Raises:
            ValueError: If model is not available
        """
        # Check cache first
        if model_name in self.agent_cache:
            return self.agent_cache[model_name]

        # Ensure model is available
        if not model_exists(model_name):
            if not await ensure_model_available(model_name):
                raise ValueError(f"Model '{model_name}' is not available and could not be downloaded")

        # Create agent
        try:
            agent = get_agent_instance(model_name, streaming=self.enable_streaming)
            if not agent:
                raise ValueError(f"Failed to create agent for model '{model_name}'")
            
            # Cache the agent
            self.agent_cache[model_name] = agent
            self.logger.info(f"Created and cached agent for model: {model_name}")
            
            return agent
            
        except Exception as e:
            self.logger.error(f"Failed to create agent for model '{model_name}': {e}")
            raise

    async def handle_model_swap(self, 
                               session: OrchestrationSession,
                               current_step: ExecutionStep,
                               reason: str = "capability_mismatch") -> Optional[str]:
        """Handle switching to a different model for better capability match.
        
        Args:
            session: Orchestration session
            current_step: Current execution step
            reason: Reason for model swap
            
        Returns:
            New model name if swap occurred, None otherwise
        """
        if session.model_switches >= self.max_model_switches:
            self.logger.warning(f"Maximum model switches ({self.max_model_switches}) reached for session {session.session_id}")
            return None

        # Get current model from the step
        current_model = current_step.preferred_model

        # Find a better model based on step requirements
        better_model = await self._find_better_model(current_step, current_model)
        
        if better_model and better_model != current_model:
            self.logger.info(f"Swapping model from '{current_model}' to '{better_model}' for step '{current_step.step_id}' (reason: {reason})")
            
            # Update the step's preferred model
            current_step.preferred_model = better_model
            
            # Update session tracking
            session.model_switches += 1
            
            return better_model
        
        return None

    async def _find_better_model(self, step: ExecutionStep, current_model: str) -> Optional[str]:
        """Find a better model for the given step.
        
        Args:
            step: Execution step
            current_model: Current model name
            
        Returns:
            Better model name if found, None otherwise
        """
        # Get step requirements
        step_type = getattr(step, 'step_type', 'general')
        requires_tools = self._step_requires_tools(step)
        
        # Get available models with capabilities
        available_models = self.capability_checker.get_available_models()
        
        # Score models based on step requirements
        best_model = None
        best_score = 0
        
        for model_info in available_models:
            model_name = model_info.get('name', '')
            if model_name == current_model:
                continue
                
            score = self._score_model_for_step(model_info, step_type, requires_tools)
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model

    def _score_model_for_step(self, model_info: Dict, step_type: str, requires_tools: bool) -> int:
        """Score a model for a specific step.
        
        Args:
            model_info: Model information
            step_type: Type of step
            requires_tools: Whether step requires tools
            
        Returns:
            Score (higher is better)
        """
        score = 0
        
        # Base score from model size/capability
        if 'parameters' in model_info:
            params = model_info['parameters']
            if params >= 13:  # 13B+ models
                score += 30
            elif params >= 7:  # 7B+ models
                score += 20
            else:  # Smaller models
                score += 10
        
        # Bonus for coding capability if step is code-related
        if step_type in ['code', 'analysis', 'debugging']:
            if model_info.get('supports_coding', False):
                score += 25
        
        # Bonus for tool support if needed
        if requires_tools:
            if model_info.get('supports_tools', False):
                score += 20
        
        # Bonus for faster models for simple tasks
        if step_type in ['simple', 'format', 'summary']:
            if model_info.get('size', 'large') == 'small':
                score += 15
        
        return score

    def _step_requires_tools(self, step: ExecutionStep) -> bool:
        """Check if a step requires tool usage.
        
        Args:
            step: Execution step
            
        Returns:
            True if step requires tools
        """
        # Keywords that indicate tool usage
        tool_keywords = [
            'file', 'repository', 'github', 'clone', 'download',
            'search', 'analyze', 'execute', 'run', 'create',
            'modify', 'update', 'delete', 'generate'
        ]
        
        text_to_check = f"{step.description} {step.instructions or ''}".lower()
        
        return any(keyword in text_to_check for keyword in tool_keywords)

    def update_execution_plan_model(self, 
                                   execution_plan: ExecutionPlan,
                                   old_model: str,
                                   new_model: str) -> None:
        """Update all steps in execution plan to use new model.
        
        Args:
            execution_plan: Execution plan to update
            old_model: Old model name
            new_model: New model name
        """
        for step in execution_plan.steps:
            if step.preferred_model == old_model:
                step.preferred_model = new_model
        
        self.logger.info(f"Updated execution plan: {old_model} -> {new_model}")

    def get_cached_agents(self) -> Dict[str, str]:
        """Get information about cached agents.
        
        Returns:
            Dictionary mapping model names to agent info
        """
        return {
            model_name: f"Agent for {model_name} (streaming: {self.enable_streaming})"
            for model_name in self.agent_cache.keys()
        }

    def clear_agent_cache(self) -> None:
        """Clear the agent cache."""
        self.agent_cache.clear()
        self.logger.info("Agent cache cleared")
