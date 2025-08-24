#!/usr/bin/env python3
"""
Integrations module for external service connections.

This module provides integrations with external services following the
single responsibility principle. Each integration handles communication
with a specific external service.
"""

from .base_integration import BaseIntegration
from .ollama_integration import OllamaIntegration
from .integration_manager import IntegrationManager
from .agent_registry import AgentRegistry, AgentInfo, ModelAgentMatch
from .model_config_reader import ModelConfigReader, ModelConfig

__all__ = ["BaseIntegration", "OllamaIntegration", "IntegrationManager", "AgentRegistry", "AgentInfo", "ModelAgentMatch", "ModelConfigReader", "ModelConfig"]
