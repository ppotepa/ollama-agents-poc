#!/usr/bin/env python3
"""
Agent Registry Module

Handles discovery and comparison of available agents against integration models.
Provides functionality to match models with their agent implementations.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class AgentInfo:
    """Information about an available agent implementation."""
    name: str
    module_path: str
    description: Optional[str] = None
    model_patterns: list[str] = None  # Patterns this agent can handle
    family: Optional[str] = None
    capabilities: list[str] = None

    def __post_init__(self):
        if self.model_patterns is None:
            self.model_patterns = []
        if self.capabilities is None:
            self.capabilities = []


@dataclass
class ModelAgentMatch:
    """Result of matching a model with available agents."""
    model_id: str
    model_info: dict[str, Any]
    has_agent: bool
    agent_info: Optional[AgentInfo] = None
    match_confidence: float = 0.0  # 0.0 to 1.0
    match_reason: str = ""


class AgentRegistry:
    """
    Registry for discovering and managing agent implementations.

    Responsibilities:
    - Discover available agent implementations
    - Match models with compatible agents
    - Provide agent metadata and capabilities
    """

    def __init__(self, agents_dir: str = None):
        """
        Initialize the agent registry.

        Args:
            agents_dir: Path to the agents directory
        """
        self.agents_dir = agents_dir or self._get_default_agents_dir()
        self._agents: dict[str, AgentInfo] = {}
        self._model_patterns: dict[str, list[str]] = {}
        self._discovered = False

    def _get_default_agents_dir(self) -> str:
        """Get the default agents directory path."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(os.path.dirname(current_dir), "agents")

    def discover_agents(self) -> list[AgentInfo]:
        """
        Discover all available agent implementations.

        Returns:
            List of discovered agent information
        """
        if self._discovered:
            return list(self._agents.values())

        self._agents.clear()
        self._model_patterns.clear()

        # Scan for agent modules
        agents_path = Path(self.agents_dir)
        if not agents_path.exists():
            print(f"Warning: Agents directory not found: {self.agents_dir}")
            return []

        # Look for agent directories
        for item in agents_path.iterdir():
            if item.is_dir() and not item.name.startswith('__') and item.name != 'base':
                try:
                    agent_info = self._discover_agent_in_directory(item)
                    if agent_info:
                        self._agents[agent_info.name] = agent_info
                except Exception as e:
                    print(f"Warning: Failed to discover agent in {item}: {e}")

        self._discovered = True
        return list(self._agents.values())

    def _discover_agent_in_directory(self, agent_dir: Path) -> Optional[AgentInfo]:
        """
        Discover agent information from a directory.

        Args:
            agent_dir: Path to the agent directory

        Returns:
            AgentInfo if found, None otherwise
        """
        # Look for main agent file or __init__.py
        possible_files = [
            agent_dir / "__init__.py",
            agent_dir / f"{agent_dir.name}_agent.py",
            agent_dir / "agent.py",
            agent_dir / "main.py"
        ]

        for agent_file in possible_files:
            if agent_file.exists():
                return self._extract_agent_info(agent_dir.name, agent_file)

        return None

    def _extract_agent_info(self, agent_name: str, agent_file: Path) -> Optional[AgentInfo]:
        """
        Extract agent information from a Python file.

        Args:
            agent_name: Name of the agent
            agent_file: Path to the agent file

        Returns:
            AgentInfo if valid agent found
        """
        try:
            # Read the file to extract metadata
            with open(agent_file, encoding='utf-8') as f:
                content = f.read()

            # Extract basic information
            description = self._extract_description(content)
            model_patterns = self._extract_model_patterns(agent_name, content)
            family = self._extract_family(agent_name, content)
            capabilities = self._extract_capabilities(content)

            return AgentInfo(
                name=agent_name,
                module_path=str(agent_file),
                description=description,
                model_patterns=model_patterns,
                family=family,
                capabilities=capabilities
            )

        except Exception as e:
            print(f"Error extracting agent info from {agent_file}: {e}")
            return None

    def _extract_description(self, content: str) -> Optional[str]:
        """Extract description from module docstring or comments."""
        lines = content.split('\n')

        # Look for module docstring
        in_docstring = False
        docstring_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('"""', "'''")):
                if in_docstring:
                    # End of docstring
                    break
                else:
                    # Start of docstring
                    in_docstring = True
                    # Check if it's a single-line docstring
                    if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                        return stripped.strip('"""').strip("'''").strip()
                    continue
            elif in_docstring:
                docstring_lines.append(stripped)

        if docstring_lines:
            return ' '.join(docstring_lines).strip()

        return None

    def _extract_model_patterns(self, agent_name: str, content: str) -> list[str]:
        """Extract model patterns this agent can handle."""
        patterns = []

        # Based on agent name, infer compatible models
        name_lower = agent_name.lower()

        if "deepcoder" in name_lower:
            patterns.extend(["deepcoder:*", "deepseek-coder:*", "*coder*"])
        elif "deepseek" in name_lower:
            patterns.extend(["deepseek:*", "deepseek-*"])
        elif "llama" in name_lower:
            patterns.extend(["llama*", "codellama:*"])
        elif "qwen" in name_lower:
            patterns.extend(["qwen*", "qwen2*"])
        elif "mistral" in name_lower:
            patterns.extend(["mistral:*"])
        elif "gemma" in name_lower:
            patterns.extend(["gemma:*"])
        elif "phi" in name_lower:
            patterns.extend(["phi*"])
        else:
            # Generic agent - can handle any model
            patterns.append("*")

        return patterns

    def _extract_family(self, agent_name: str, content: str) -> Optional[str]:
        """Extract model family from agent name or content."""
        name_lower = agent_name.lower()

        family_mapping = {
            "deepcoder": "deepseek-coder",
            "deepseek": "deepseek",
            "llama": "llama",
            "codellama": "codellama",
            "qwen": "qwen",
            "mistral": "mistral",
            "gemma": "gemma",
            "phi": "phi"
        }

        for family_key, family_value in family_mapping.items():
            if family_key in name_lower:
                return family_value

        return None

    def _extract_capabilities(self, content: str) -> list[str]:
        """Extract capabilities from agent content."""
        capabilities = []
        content_lower = content.lower()

        # Common capability patterns
        capability_patterns = {
            "code": ["code", "coding", "programming", "development"],
            "chat": ["chat", "conversation", "dialogue"],
            "analysis": ["analysis", "analyze", "analytical"],
            "tools": ["tools", "tool_use", "function_calling"],
            "files": ["file", "files", "filesystem"],
            "terminal": ["terminal", "command", "shell"],
            "git": ["git", "repository", "version_control"],
            "debugging": ["debug", "debugging", "troubleshoot"]
        }

        for capability, patterns in capability_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                capabilities.append(capability)

        return capabilities

    def match_models_with_agents(self, models: list[dict[str, Any]]) -> list[ModelAgentMatch]:
        """
        Match models with available agent implementations.

        Args:
            models: List of model dictionaries from integrations

        Returns:
            List of model-agent matches
        """
        if not self._discovered:
            self.discover_agents()

        matches = []

        for model in models:
            model_id = model.get("id", "")
            match = self._find_best_agent_match(model_id, model)
            matches.append(match)

        return matches

    def _find_best_agent_match(self, model_id: str, model_info: dict[str, Any]) -> ModelAgentMatch:
        """
        Find the best agent match for a specific model.

        Args:
            model_id: Model identifier
            model_info: Model information dictionary

        Returns:
            ModelAgentMatch result
        """
        best_match = None
        best_confidence = 0.0
        best_reason = ""

        for _agent_name, agent_info in self._agents.items():
            confidence, reason = self._calculate_match_confidence(model_id, model_info, agent_info)

            if confidence > best_confidence:
                best_match = agent_info
                best_confidence = confidence
                best_reason = reason

        return ModelAgentMatch(
            model_id=model_id,
            model_info=model_info,
            has_agent=best_match is not None and best_confidence > 0.3,
            agent_info=best_match,
            match_confidence=best_confidence,
            match_reason=best_reason
        )

    def _calculate_match_confidence(self, model_id: str, model_info: dict[str, Any], agent_info: AgentInfo) -> tuple[float, str]:
        """
        Calculate confidence score for a model-agent match.

        Returns:
            Tuple of (confidence_score, reason)
        """
        model_id_lower = model_id.lower()
        agent_name_lower = agent_info.name.lower()

        # Exact name match
        if agent_name_lower == model_id_lower.split(':')[0]:
            return 1.0, f"Exact name match: {agent_info.name}"

        # Family match
        model_family = model_info.get("details", {}).get("family", "").lower()
        if model_family and agent_info.family and model_family == agent_info.family.lower():
            return 0.9, f"Family match: {model_family}"

        # Pattern matching
        for pattern in agent_info.model_patterns:
            if self._pattern_matches(model_id, pattern):
                confidence = 0.8 if pattern != "*" else 0.4
                return confidence, f"Pattern match: {pattern}"

        # Partial name match
        for part in agent_name_lower.split('_'):
            if part in model_id_lower:
                return 0.6, f"Partial name match: {part}"

        return 0.0, "No match found"

    def _pattern_matches(self, model_id: str, pattern: str) -> bool:
        """Check if model ID matches a pattern."""
        if pattern == "*":
            return True

        pattern = pattern.lower()
        model_id = model_id.lower()

        if pattern.endswith("*"):
            return model_id.startswith(pattern[:-1])
        elif pattern.startswith("*"):
            return model_id.endswith(pattern[1:])
        elif "*" in pattern:
            parts = pattern.split("*")
            return all(part in model_id for part in parts if part)
        else:
            return pattern == model_id

    def get_agent_info(self, agent_name: str) -> Optional[AgentInfo]:
        """Get information about a specific agent."""
        if not self._discovered:
            self.discover_agents()
        return self._agents.get(agent_name)

    def list_agents(self) -> list[AgentInfo]:
        """List all discovered agents."""
        if not self._discovered:
            self.discover_agents()
        return list(self._agents.values())

    def get_agents_for_model(self, model_id: str) -> list[AgentInfo]:
        """Get all agents that can handle a specific model."""
        if not self._discovered:
            self.discover_agents()

        compatible_agents = []
        for agent_info in self._agents.values():
            confidence, _ = self._calculate_match_confidence(model_id, {}, agent_info)
            if confidence > 0.3:  # Threshold for compatibility
                compatible_agents.append(agent_info)

        return compatible_agents
