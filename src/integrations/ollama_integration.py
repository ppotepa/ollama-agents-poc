#!/usr/bin/env python3
"""
Ollama Integration Module

Handles all communication with Ollama service for model discovery and management.
Follows single responsibility principle by focusing only on Ollama-specific operations.
"""

import os
import subprocess
import time
from typing import Any, Optional

import requests

from .base_integration import BaseIntegration


class OllamaIntegration(BaseIntegration):
    """
    Integration class for Ollama service communication.

    Responsibilities:
    - Discover available models from Ollama
    - Format model information for OpenAI compatibility
    - Handle connection errors gracefully
    - Provide fallback mechanisms
    """

    def __init__(self, host: Optional[str] = None, timeout: int = 5):
        """
        Initialize Ollama integration.

        Args:
            host: Ollama server host URL (defaults to OLLAMA_HOST env var or container default)
            timeout: Request timeout in seconds
        """
        self.host = host or os.getenv("OLLAMA_HOST", "http://ollama:11434")
        self.timeout = timeout

    def get_models(self) -> list[dict[str, Any]]:
        """
        Fetch available models from Ollama server.

        Returns:
            List of model dictionaries in OpenAI-compatible format
        """
        try:
            # Try HTTP API first
            models = self._fetch_models_via_api()
            if models:
                return models
        except Exception as e:
            print(f"Failed to fetch models via HTTP API: {e}")

        try:
            # Fallback to CLI
            models = self._fetch_models_via_cli()
            if models:
                return models
        except Exception as e:
            print(f"Failed to fetch models via CLI: {e}")

        # Return default model if all methods fail
        return [self._get_default_model()]

    def _fetch_models_via_api(self) -> list[dict[str, Any]]:
        """Fetch models using Ollama HTTP API."""
        response = requests.get(f"{self.host}/api/tags", timeout=self.timeout)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            return [self._format_model_from_api(model) for model in models]
        return []

    def _fetch_models_via_cli(self) -> list[dict[str, Any]]:
        """Fetch models using Ollama CLI command."""
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
            encoding='utf-8',
            errors='replace'
        )
        if result.returncode == 0:
            return self._parse_cli_output(result.stdout)
        return []

    def _format_model_from_api(self, model_data: dict[str, Any]) -> dict[str, Any]:
        """Format model data from API response into OpenAI-compatible format."""
        model_name = model_data.get("name", "unknown")
        model_size = model_data.get("size", 0)
        modified_at = model_data.get("modified_at", "")

        # Convert timestamp
        created_timestamp = self._parse_timestamp(modified_at) if modified_at else self._now()

        return {
            "id": model_name,
            "object": "model",
            "created": created_timestamp,
            "owned_by": "ollama",
            "permission": [],
            "supports_agent": True,
            "supports_tools": True,
            "supports_tool_calls": True,
            "supports_function_calling": True,
            "capabilities": {
                "agent": True,
                "tools": True,
                "tool_calls": True,
                "functions": True,
                "function_calling": True,
            },
            "tool_resources": ["code", "files", "terminal", "docs", "diff", "problems", "folder", "codebase"],
            "type": "chat.completions",
            "details": {
                "size": self._format_size(model_size) if model_size else "Unknown",
                "family": self._guess_model_family(model_name),
                "parameter_size": model_data.get("details", {}).get("parameter_size", "Unknown")
            }
        }

    def _parse_cli_output(self, output: str) -> list[dict[str, Any]]:
        """Parse 'ollama list' command output into model list."""
        models = []
        lines = output.strip().split('\n')

        # Skip header line if present
        for line in lines[1:] if len(lines) > 1 else lines:
            if line.strip():
                parts = line.split()
                if parts:
                    model_name = parts[0]
                    size = parts[2] if len(parts) > 2 else "Unknown"

                    models.append({
                        "id": model_name,
                        "object": "model",
                        "created": self._now(),
                        "owned_by": "ollama",
                        "permission": [],
                        "supports_agent": True,
                        "supports_tools": True,
                        "supports_tool_calls": True,
                        "supports_function_calling": True,
                        "capabilities": {
                            "agent": True,
                            "tools": True,
                            "tool_calls": True,
                            "functions": True,
                            "function_calling": True,
                        },
                        "tool_resources": ["code", "files", "terminal", "docs", "diff", "problems", "folder", "codebase"],
                        "type": "chat.completions",
                        "details": {
                            "size": size,
                            "family": self._guess_model_family(model_name),
                        }
                    })

        return models if models else [self._get_default_model()]

    def _format_size(self, size_bytes: int) -> str:
        """Format model size in bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}PB"

    def _guess_model_family(self, model_name: str) -> str:
        """Guess model family from model name."""
        name_lower = model_name.lower()

        if "deepseek" in name_lower:
            return "deepseek-coder" if "coder" in name_lower else "deepseek"
        elif "llama" in name_lower:
            return "codellama" if "code" in name_lower else "llama"
        elif "qwen" in name_lower:
            return "qwen-coder" if "coder" in name_lower else "qwen"
        elif "mistral" in name_lower:
            return "mistral"
        elif "gemma" in name_lower:
            return "gemma"
        elif "phi" in name_lower:
            return "phi"
        elif "tinyllama" in name_lower:
            return "tinyllama"
        else:
            return "unknown"

    def _parse_timestamp(self, timestamp_str: str) -> int:
        """Parse ISO timestamp string to Unix timestamp."""
        try:
            return int(time.mktime(time.strptime(timestamp_str[:19], "%Y-%m-%dT%H:%M:%S")))
        except (ValueError, TypeError):
            return self._now()

    def _now(self) -> int:
        """Get current Unix timestamp."""
        return int(time.time())

    def _get_default_model(self) -> dict[str, Any]:
        """Return a default model when Ollama is not available."""
        return {
            "id": "deepseek-agent",
            "object": "model",
            "created": self._now(),
            "owned_by": "local",
            "permission": [],
            "supports_agent": True,
            "supports_tools": True,
            "supports_tool_calls": True,
            "supports_function_calling": True,
            "capabilities": {
                "agent": True,
                "tools": True,
                "tool_calls": True,
                "functions": True,
                "function_calling": True,
            },
            "tool_resources": ["code", "files", "terminal", "docs", "diff", "problems", "folder", "codebase"],
            "type": "chat.completions",
            "details": {
                "size": "Unknown",
                "family": "deepseek",
                "status": "fallback - ollama not available"
            }
        }

    def is_available(self) -> bool:
        """
        Check if Ollama service is available.

        Returns:
            True if Ollama is reachable, False otherwise
        """
        try:
            response = requests.get(f"{self.host}/api/version", timeout=self.timeout)
            return response.status_code == 200
        except Exception:
            return False

    def get_version(self) -> Optional[str]:
        """
        Get Ollama version information.

        Returns:
            Version string if available, None otherwise
        """
        try:
            response = requests.get(f"{self.host}/api/version", timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                return data.get("version", "unknown")
        except Exception:
            pass
        return None
