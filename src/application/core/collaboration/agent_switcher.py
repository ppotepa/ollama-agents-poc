"""Agent switching logic for collaborative execution."""

import logging
from typing import Any, Optional, Dict

from ..agent_factory import get_agent_factory


class AgentSwitcher:
    """Handles switching between different agents during collaborative execution."""

    def __init__(self, main_agent, original_agent_id: Optional[str] = None):
        """Initialize the agent switcher."""
        self.main_agent = main_agent
        self.original_agent_id = original_agent_id or self._extract_agent_id(main_agent)
        self._agent_factory = None
        self.logger = logging.getLogger(__name__)
        
        # Track switching history
        self.switch_history = []

    def _extract_agent_id(self, agent) -> Optional[str]:
        """Extract agent ID from an agent instance."""
        if hasattr(agent, '_model_id'):
            return agent._model_id
        elif hasattr(agent, 'agent_id'):
            return agent.agent_id
        elif hasattr(agent, 'model_id'):
            return agent.model_id
        return None

    def _get_agent_factory(self):
        """Get or create the agent factory."""
        if self._agent_factory is None:
            # Default to streaming=True for collaborative system
            self._agent_factory = get_agent_factory(streaming=True)
        return self._agent_factory

    def should_switch_agent(self, query: str, current_iteration: int) -> Optional[str]:
        """Determine if we should switch to a different agent based on query analysis.
        
        Args:
            query: The current query being processed
            current_iteration: Current iteration number
            
        Returns:
            Agent ID to switch to, or None if no switch is recommended
        """
        # Don't switch on first iteration
        if current_iteration <= 1:
            return None

        current_agent_id = self._extract_agent_id(self.main_agent)
        
        # Analyze query characteristics
        query_lower = query.lower()
        
        # Code analysis patterns
        code_patterns = [
            'analyze', 'review', 'debug', 'fix', 'optimize', 'refactor',
            'function', 'class', 'method', 'variable', 'import', 'module'
        ]
        
        # Data/research patterns  
        data_patterns = [
            'research', 'find', 'search', 'investigate', 'explore',
            'data', 'dataset', 'statistics', 'analysis', 'report'
        ]
        
        # Creative/generation patterns
        creative_patterns = [
            'create', 'generate', 'write', 'design', 'build', 'make',
            'story', 'content', 'document', 'template', 'example'
        ]

        # Math/calculation patterns
        math_patterns = [
            'calculate', 'compute', 'solve', 'math', 'formula',
            'equation', 'statistics', 'probability', 'algorithm'
        ]

        # Count pattern matches
        code_score = sum(1 for pattern in code_patterns if pattern in query_lower)
        data_score = sum(1 for pattern in data_patterns if pattern in query_lower)
        creative_score = sum(1 for pattern in creative_patterns if pattern in query_lower)
        math_score = sum(1 for pattern in math_patterns if pattern in query_lower)

        # Get available agents
        agent_factory = self._get_agent_factory()
        available_agents = agent_factory.get_available_agents()

        # Determine best agent based on patterns
        recommendations = []

        if code_score >= 2:
            # Look for code-focused models
            code_agents = [
                'deepseek-coder-v2', 'codellama', 'codegemma', 'starcoder2',
                'qwen2.5-coder', 'granite-code'
            ]
            for agent_id in code_agents:
                if agent_id in available_agents and agent_id != current_agent_id:
                    recommendations.append((agent_id, code_score, "code analysis"))
                    break

        if data_score >= 2:
            # Look for research/analysis models
            research_agents = [
                'llama3.1', 'qwen2.5', 'gemma2', 'mistral', 'phi3'
            ]
            for agent_id in research_agents:
                if agent_id in available_agents and agent_id != current_agent_id:
                    recommendations.append((agent_id, data_score, "research/analysis"))
                    break

        if creative_score >= 2:
            # Look for creative models
            creative_agents = [
                'llama3.1', 'mistral', 'gemma2', 'qwen2.5'
            ]
            for agent_id in creative_agents:
                if agent_id in available_agents and agent_id != current_agent_id:
                    recommendations.append((agent_id, creative_score, "creative tasks"))
                    break

        if math_score >= 2:
            # Look for math-focused models
            math_agents = [
                'qwen2.5', 'deepseek-math', 'llama3.1', 'gemma2'
            ]
            for agent_id in math_agents:
                if agent_id in available_agents and agent_id != current_agent_id:
                    recommendations.append((agent_id, math_score, "mathematical computation"))
                    break

        # Return the best recommendation
        if recommendations:
            # Sort by score and return the highest
            best_recommendation = max(recommendations, key=lambda x: x[1])
            return best_recommendation[0]

        return None

    def switch_main_agent(self, new_agent_id: str) -> bool:
        """Switch the main agent to a new model.
        
        Args:
            new_agent_id: ID of the new agent to switch to
            
        Returns:
            True if switch was successful, False otherwise
        """
        try:
            current_agent_id = self._extract_agent_id(self.main_agent)
            
            if current_agent_id == new_agent_id:
                self.logger.info(f"Already using agent {new_agent_id}, no switch needed")
                return True

            # Get agent factory and create new agent
            agent_factory = self._get_agent_factory()
            new_agent = agent_factory.create_agent(new_agent_id)
            
            if new_agent is None:
                self.logger.error(f"Failed to create agent: {new_agent_id}")
                return False

            # Store the switch in history
            self.switch_history.append({
                "from_agent": current_agent_id,
                "to_agent": new_agent_id,
                "timestamp": __import__('time').time(),
                "reason": "Performance optimization"
            })

            # Perform the switch
            old_agent = self.main_agent
            self.main_agent = new_agent
            
            self.logger.info(f"Successfully switched from {current_agent_id} to {new_agent_id}")
            
            # Clean up old agent if needed
            if hasattr(old_agent, 'cleanup'):
                try:
                    old_agent.cleanup()
                except Exception as e:
                    self.logger.warning(f"Error cleaning up old agent: {e}")

            return True

        except Exception as e:
            self.logger.error(f"Error switching to agent {new_agent_id}: {str(e)}")
            return False

    def get_current_agent_info(self) -> Dict[str, Any]:
        """Get information about the current agent."""
        agent_id = self._extract_agent_id(self.main_agent)
        
        return {
            "agent_id": agent_id,
            "original_agent_id": self.original_agent_id,
            "is_original": agent_id == self.original_agent_id,
            "switch_count": len(self.switch_history),
            "last_switch": self.switch_history[-1] if self.switch_history else None
        }

    def get_switch_history(self) -> list[Dict[str, Any]]:
        """Get the complete switch history."""
        return self.switch_history.copy()

    def revert_to_original_agent(self) -> bool:
        """Revert back to the original agent."""
        if self.original_agent_id is None:
            self.logger.error("No original agent ID stored")
            return False

        return self.switch_main_agent(self.original_agent_id)

    def get_switching_statistics(self) -> Dict[str, Any]:
        """Get statistics about agent switching."""
        if not self.switch_history:
            return {
                "total_switches": 0,
                "unique_agents_used": 1,
                "most_used_agent": self.original_agent_id,
                "switch_frequency": 0.0
            }

        # Analyze switch history
        agents_used = set([self.original_agent_id])
        for switch in self.switch_history:
            agents_used.add(switch["from_agent"])
            agents_used.add(switch["to_agent"])

        # Count usage per agent
        agent_usage = {self.original_agent_id: 1}
        for switch in self.switch_history:
            agent_usage[switch["to_agent"]] = agent_usage.get(switch["to_agent"], 0) + 1

        most_used_agent = max(agent_usage.items(), key=lambda x: x[1])[0]

        # Calculate switch frequency (switches per hour)
        if self.switch_history:
            time_span = self.switch_history[-1]["timestamp"] - self.switch_history[0]["timestamp"]
            switch_frequency = len(self.switch_history) / max(time_span / 3600, 0.01)  # Per hour
        else:
            switch_frequency = 0.0

        return {
            "total_switches": len(self.switch_history),
            "unique_agents_used": len(agents_used),
            "most_used_agent": most_used_agent,
            "switch_frequency": switch_frequency,
            "agent_usage": agent_usage
        }
