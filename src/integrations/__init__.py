#!/usr/bin/env python3
"""
Integrations module for external service connections.

This module provides integrations with external services following the
single responsibility principle. Each integration handles communication
with a specific external service.
"""

from .agent_registry import AgentInfo, AgentRegistry, ModelAgentMatch
from .base_integration import BaseIntegration
from .integration_manager import IntegrationManager
from .model_config_reader import ModelConfig, ModelConfigReader
from .ollama_integration import OllamaIntegration

__all__ = ["BaseIntegration", "OllamaIntegration", "IntegrationManager", "AgentRegistry", "AgentInfo", "ModelAgentMatch", "ModelConfigReader", "ModelConfig"]
