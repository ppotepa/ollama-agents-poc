"""API-based model discovery for Ollama instances."""

import json
from typing import Any, Dict, List

from src.utils.enhanced_logging import get_logger

try:
    import requests
except ImportError:
    import urllib.error
    import urllib.request
    requests = None


class APIModelDiscovery:
    """Discovers models via Ollama API."""

    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        """Initialize API discovery.
        
        Args:
            ollama_base_url: Base URL for Ollama API
        """
        self.ollama_base_url = ollama_base_url
        self.logger = get_logger()

    def discover_models(self) -> List[Dict[str, Any]]:
        """Discover models via API.
        
        Returns:
            List of model configurations
        """
        try:
            models = []
            
            if requests:
                models = self._discover_with_requests()
            else:
                models = self._discover_with_urllib()
                
            self.logger.info(f"Discovered {len(models)} models via API")
            return models
            
        except Exception as e:
            self.logger.error(f"API discovery failed: {e}")
            return []

    def _discover_with_requests(self) -> List[Dict[str, Any]]:
        """Discover models using requests library.
        
        Returns:
            List of model configurations
        """
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            data = response.json()
            models = data.get("models", [])
            
            return [self._process_api_model(model) for model in models]
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Requests API call failed: {e}")
            return []

    def _discover_with_urllib(self) -> List[Dict[str, Any]]:
        """Discover models using urllib (fallback).
        
        Returns:
            List of model configurations
        """
        try:
            import urllib.request
            import urllib.error
            
            req = urllib.request.Request(f"{self.ollama_base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                models = data.get("models", [])
                
                return [self._process_api_model(model) for model in models]
                
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            self.logger.error(f"Urllib API call failed: {e}")
            return []

    def _process_api_model(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw API model data into standardized format.
        
        Args:
            model_data: Raw model data from API
            
        Returns:
            Processed model configuration
        """
        name = model_data.get("name", "unknown")
        size = model_data.get("size", 0)
        
        return {
            "name": name,
            "size": size,
            "size_b": self._convert_to_billions(size),
            "modified": model_data.get("modified"),
            "digest": model_data.get("digest"),
            "details": model_data.get("details", {}),
            "source": "api"
        }

    def _convert_to_billions(self, size_bytes: int) -> float:
        """Convert bytes to billions of parameters (rough estimate).
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Estimated size in billions of parameters
        """
        if size_bytes == 0:
            return 0.0
        
        # Rough estimate: 1 billion parameters ≈ 2GB for FP16 models
        return size_bytes / (2 * 1024 * 1024 * 1024)

    def pull_model(self, model_name: str) -> bool:
        """Pull a model via API.
        
        Args:
            model_name: Name of model to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if requests:
                return self._pull_with_requests(model_name)
            else:
                return self._pull_with_urllib(model_name)
                
        except Exception as e:
            self.logger.error(f"Failed to pull model {model_name}: {e}")
            return False

    def _pull_with_requests(self, model_name: str) -> bool:
        """Pull model using requests library.
        
        Args:
            model_name: Name of model to pull
            
        Returns:
            True if successful
        """
        try:
            payload = {"name": model_name}
            response = requests.post(
                f"{self.ollama_base_url}/api/pull",
                json=payload,
                timeout=300  # 5 minutes for pulling
            )
            response.raise_for_status()
            
            self.logger.info(f"Successfully pulled model: {model_name}")
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to pull {model_name} with requests: {e}")
            return False

    def _pull_with_urllib(self, model_name: str) -> bool:
        """Pull model using urllib (fallback).
        
        Args:
            model_name: Name of model to pull
            
        Returns:
            True if successful
        """
        try:
            import urllib.request
            import urllib.error
            
            payload = json.dumps({"name": model_name}).encode()
            req = urllib.request.Request(
                f"{self.ollama_base_url}/api/pull",
                data=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=300) as response:
                if response.status == 200:
                    self.logger.info(f"Successfully pulled model: {model_name}")
                    return True
                    
            return False
            
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to pull {model_name} with urllib: {e}")
            return False

    def check_model_exists(self, model_name: str) -> bool:
        """Check if a model exists via API.
        
        Args:
            model_name: Name of model to check
            
        Returns:
            True if model exists
        """
        models = self.discover_models()
        return any(model["name"] == model_name for model in models)

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model.
        
        Args:
            model_name: Name of model
            
        Returns:
            Model information dictionary
        """
        models = self.discover_models()
        for model in models:
            if model["name"] == model_name:
                return model
        
        return {}
