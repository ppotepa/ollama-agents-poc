#!/usr/bin/env python3
"""
Integration Manager

Coordinates multiple external service integrations and provides
a unified interface for the application to interact with all services.
"""

from typing import Any, Optional

from .base_integration import BaseIntegration
from .ollama_integration import OllamaIntegration


class IntegrationManager:
    """
    Manages all external service integrations.

    Responsibilities:
    - Initialize and maintain integration instances
    - Provide unified access to all integrations
    - Handle integration discovery and health monitoring
    - Aggregate results from multiple sources
    """

    def __init__(self):
        """Initialize the integration manager with default integrations."""
        self._integrations: dict[str, BaseIntegration] = {}
        self._initialize_default_integrations()

    def _initialize_default_integrations(self):
        """Initialize default integrations."""
        # Try localhost first for local development, then container address
        hosts_to_try = [
            "http://localhost:11434",  # Local development
            "http://ollama:11434"      # Container environment
        ]

        ollama_integration = None
        for host in hosts_to_try:
            try:
                test_integration = OllamaIntegration(host=host)
                if test_integration.is_available():
                    ollama_integration = test_integration
                    break
            except Exception:
                continue

        if ollama_integration is None:
            # Use localhost as fallback even if not available
            ollama_integration = OllamaIntegration(host="http://localhost:11434")

        self.add_integration("ollama", ollama_integration)

    def add_integration(self, name: str, integration: BaseIntegration):
        """
        Add a new integration to the manager.

        Args:
            name: Unique name for the integration
            integration: Integration instance
        """
        self._integrations[name] = integration

    def remove_integration(self, name: str):
        """
        Remove an integration from the manager.

        Args:
            name: Name of the integration to remove
        """
        if name in self._integrations:
            del self._integrations[name]

    def get_integration(self, name: str) -> Optional[BaseIntegration]:
        """
        Get a specific integration by name.

        Args:
            name: Name of the integration

        Returns:
            Integration instance or None if not found
        """
        return self._integrations.get(name)

    def list_integrations(self) -> list[str]:
        """
        Get list of all registered integration names.

        Returns:
            List of integration names
        """
        return list(self._integrations.keys())

    def get_all_models(self) -> list[dict[str, Any]]:
        """
        Get models from all available integrations.

        Returns:
            Aggregated list of models from all integrations
        """
        all_models = []
        for name, integration in self._integrations.items():
            try:
                if integration.is_available():
                    models = integration.get_models()
                    # Add source information to each model
                    for model in models:
                        model["source"] = name
                    all_models.extend(models)
            except Exception as e:
                print(f"Failed to get models from {name}: {e}")

        return all_models

    def get_models_from(self, integration_name: str) -> list[dict[str, Any]]:
        """
        Get models from a specific integration.

        Args:
            integration_name: Name of the integration

        Returns:
            List of models from the specified integration
        """
        integration = self.get_integration(integration_name)
        if integration:
            try:
                return integration.get_models()
            except Exception as e:
                print(f"Failed to get models from {integration_name}: {e}")
        return []

    def health_check(self) -> dict[str, dict[str, Any]]:
        """
        Perform health check on all integrations.

        Returns:
            Dictionary with health status for each integration
        """
        health_status = {}
        for name, integration in self._integrations.items():
            try:
                health_status[name] = integration.health_check()
            except Exception as e:
                health_status[name] = {
                    "available": False,
                    "version": None,
                    "status": "error",
                    "error": str(e)
                }

        return health_status

    def get_available_integrations(self) -> list[str]:
        """
        Get list of currently available integrations.

        Returns:
            List of integration names that are currently available
        """
        available = []
        for name, integration in self._integrations.items():
            try:
                if integration.is_available():
                    available.append(name)
            except Exception:
                pass
        return available
