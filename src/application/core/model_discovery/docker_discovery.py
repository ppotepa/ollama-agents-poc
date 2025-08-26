"""Docker-based model discovery for containerized Ollama."""

import json
import subprocess
from typing import Any, Dict, List

from src.utils.enhanced_logging import get_logger


class DockerModelDiscovery:
    """Discovers models via Docker container commands."""

    def __init__(self, docker_container: str = "ollama"):
        """Initialize Docker discovery.
        
        Args:
            docker_container: Name of Docker container
        """
        self.docker_container = docker_container
        self.logger = get_logger()

    def discover_models(self) -> List[Dict[str, Any]]:
        """Discover models via Docker commands.
        
        Returns:
            List of model configurations
        """
        try:
            # Try ollama list command first
            models = self._discover_via_ollama_list()
            
            if not models:
                # Fallback to ls command
                models = self._discover_via_ls_command()
            
            self.logger.info(f"Discovered {len(models)} models via Docker")
            return models
            
        except Exception as e:
            self.logger.error(f"Docker discovery failed: {e}")
            return []

    def _discover_via_ollama_list(self) -> List[Dict[str, Any]]:
        """Use 'ollama list' command to discover models.
        
        Returns:
            List of model configurations
        """
        try:
            cmd = ["docker", "exec", self.docker_container, "ollama", "list", "--json"]
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30,
                check=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                models = data.get("models", [])
                return [self._process_docker_model(model) for model in models]
            
            return []
            
        except (subprocess.SubprocessError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
            self.logger.debug(f"ollama list command failed: {e}")
            return []

    def _discover_via_ls_command(self) -> List[Dict[str, Any]]:
        """Use ls command to discover model files.
        
        Returns:
            List of model configurations
        """
        try:
            # List model directories
            cmd = ["docker", "exec", self.docker_container, "ls", "-la", "/root/.ollama/models/manifests/registry.ollama.ai/"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return []
            
            models = []
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 9 and not parts[-1].startswith('.'):
                    model_name = parts[-1]
                    
                    # Get more detailed info for each model
                    model_info = self._get_model_details(model_name)
                    if model_info:
                        models.append(model_info)
            
            return models
            
        except (subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
            self.logger.debug(f"ls command failed: {e}")
            return []

    def _get_model_details(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model detail dictionary
        """
        try:
            # Try to get model info
            cmd = ["docker", "exec", self.docker_container, "ollama", "show", model_name, "--json"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                return self._process_docker_model({
                    "name": model_name,
                    "details": data
                })
            
            # Fallback to basic info
            return {
                "name": model_name,
                "size": 0,
                "size_b": 0.0,
                "source": "docker_ls"
            }
            
        except (subprocess.SubprocessError, json.JSONDecodeError, subprocess.TimeoutExpired):
            return {
                "name": model_name,
                "size": 0,
                "size_b": 0.0,
                "source": "docker_basic"
            }

    def _process_docker_model(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw Docker model data into standardized format.
        
        Args:
            model_data: Raw model data from Docker
            
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
            "source": "docker"
        }

    def _convert_to_billions(self, size_bytes: int) -> float:
        """Convert bytes to billions of parameters.
        
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
        """Pull a model via Docker.
        
        Args:
            model_name: Name of model to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = ["docker", "exec", self.docker_container, "ollama", "pull", model_name]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes for pulling
            )
            
            if result.returncode == 0:
                self.logger.info(f"Successfully pulled model via Docker: {model_name}")
                return True
            else:
                self.logger.error(f"Docker pull failed for {model_name}: {result.stderr}")
                return False
                
        except (subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
            self.logger.error(f"Docker pull failed for {model_name}: {e}")
            return False

    def check_model_exists(self, model_name: str) -> bool:
        """Check if a model exists via Docker.
        
        Args:
            model_name: Name of model to check
            
        Returns:
            True if model exists
        """
        models = self.discover_models()
        return any(model["name"] == model_name for model in models)

    def is_docker_available(self) -> bool:
        """Check if Docker container is available.
        
        Returns:
            True if Docker container is accessible
        """
        try:
            cmd = ["docker", "exec", self.docker_container, "echo", "test"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
            
        except (subprocess.SubprocessError, subprocess.TimeoutExpired):
            return False
