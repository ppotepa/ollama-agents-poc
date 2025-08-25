"""Dynamic model discovery for Ollama instances."""

import json
import subprocess
import os
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from src.utils.enhanced_logging import get_logger

try:
    import requests
except ImportError:
    # Fallback to urllib if requests is not available
    import urllib.request
    import urllib.error
    requests = None


class OllamaModelDiscovery:
    """Discovers available models from Ollama instance using multiple fallback methods."""
    
    # Class-level cache to avoid repeated API calls
    _cached_models = None
    _cache_initialized = False
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434", docker_container: str = "ollama"):
        self.ollama_base_url = ollama_base_url
        self.docker_container = docker_container
        self.logger = get_logger()
        
    def discover_models(self) -> List[Dict[str, Any]]:
        """Discover available models using multiple fallback methods.
        
        Returns:
            List of model configurations
        """
        # Return cached models if available
        if OllamaModelDiscovery._cache_initialized and OllamaModelDiscovery._cached_models:
            return OllamaModelDiscovery._cached_models
        
        models = []
        
        # Method 1: API call
        try:
            models = self._discover_via_api()
            if models:
                if not OllamaModelDiscovery._cache_initialized:
                    self.logger.info(f"Found {len(models)} models via API")
                    OllamaModelDiscovery._cache_initialized = True
                OllamaModelDiscovery._cached_models = models
                return models
        except Exception as e:
            self.logger.warning(f"API discovery failed: {e}")
        
        # Method 2: Docker exec
        try:
            models = self._discover_via_docker()
            if models:
                if not OllamaModelDiscovery._cache_initialized:
                    self.logger.info(f"Found {len(models)} models via Docker")
                    OllamaModelDiscovery._cache_initialized = True
                OllamaModelDiscovery._cached_models = models
                return models
        except Exception as e:
            self.logger.warning(f"Docker discovery failed: {e}")
    
    @classmethod
    def refresh_cache(cls):
        """Refresh the cached model list."""
        cls._cache_initialized = False
        cls._cached_models = None
    
    def pull_model(self, model_name: str, progress_callback: Optional[Callable[[str], None]] = None) -> bool:
        """Pull a model using docker exec ollama ollama pull.
        
        Args:
            model_name: Name of the model to pull
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Pulling model: {model_name}")
        
        # Refresh cache as we're going to add a new model
        OllamaModelDiscovery.refresh_cache()
        
        cmd = ["docker", "exec", self.docker_container, "ollama", "pull", model_name]
        
        try:
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                bufsize=1,
                encoding='utf-8',
                errors='replace'  # Handle encoding errors gracefully
            )
            
            if progress_callback:
                for line in process.stdout:
                    progress_callback(line.rstrip())
            else:
                # Default progress display
                for line in process.stdout:
                    print(line.rstrip())
            
            return_code = process.wait()
            
            if return_code == 0:
                self.logger.info(f"Successfully pulled model: {model_name}")
                return True
            else:
                self.logger.error(f"Failed to pull model {model_name}, exit code: {return_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    def ensure_model_available(self, model_name: str, auto_pull: bool = False) -> bool:
        """Check if a model is available, optionally pull it if not found.
        
        Args:
            model_name: Name of the model to check
            auto_pull: Whether to automatically pull the model if not found
            
        Returns:
            True if model is available (or successfully pulled), False otherwise
        """
        # First check if model is already available
        if self.model_exists(model_name):
            return True
        
        if auto_pull:
            self.logger.info(f"Model {model_name} not found, attempting to pull...")
            return self.pull_model(model_name)
        
        return False
        
    def model_exists(self, model_name: str) -> bool:
        """Check if a model exists in the Ollama instance.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if the model exists, False otherwise
        """
        available_models = self.discover_models()
        model_names = [model.get('name', '') for model in available_models]
        
        # Also check by model ID or partial matches
        exists = any(
            model_name == name or 
            model_name in name or 
            name.startswith(model_name)
            for name in model_names
        )
        
        return exists
        
        # Method 3: Models folder scan
        try:
            models = self._discover_via_models_folder()
            if models:
                self.logger.info(f"Found {len(models)} models via folder scan")
                return models
        except Exception as e:
            self.logger.warning(f"Folder scan discovery failed: {e}")
        
        # Method 4: Fallback to default configuration
        self.logger.warning("All discovery methods failed, using fallback configuration")
        return self._get_fallback_models()
    
    def _discover_via_api(self) -> List[Dict[str, Any]]:
        """Discover models via Ollama API."""
        url = f"{self.ollama_base_url}/api/tags"
        
        try:
            if requests:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                data = response.json()
            else:
                # Fallback to urllib
                request = urllib.request.Request(url)
                with urllib.request.urlopen(request, timeout=5) as response:
                    data = json.loads(response.read().decode())
            
            models = []
            
            for model_info in data.get("models", []):
                name = model_info.get("name", "")
                size = model_info.get("size", 0)
                
                if name:
                    model_config = self._create_model_config(name, size)
                    models.append(model_config)
            
            return models
            
        except Exception as e:
            self.logger.warning(f"API call failed: {e}")
            raise
    
    def _discover_via_docker(self) -> List[Dict[str, Any]]:
        """Discover models via Docker exec command."""
        try:
            result = subprocess.run(
                ["docker", "exec", "ollama", "ollama", "list"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
                encoding='utf-8',
                errors='replace'
            )
            
            models = []
            lines = result.stdout.strip().split('\n')
            
            # Skip header line
            for line in lines[1:]:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        name = parts[0]
                        size_str = parts[1] if len(parts) > 1 else "0B"
                        size = self._parse_size(size_str)
                        
                        model_config = self._create_model_config(name, size)
                        models.append(model_config)
            
            return models
            
        except subprocess.TimeoutExpired:
            self.logger.warning("Docker command timed out")
            return []
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Docker command failed: {e}")
            return []
    
    def _discover_via_models_folder(self) -> List[Dict[str, Any]]:
        """Discover models by scanning the models folder."""
        models_dir = Path("models/models")
        if not models_dir.exists():
            return []
        
        models = []
        
        for model_path in models_dir.iterdir():
            if model_path.is_dir():
                # Extract model name and size info
                name = model_path.name
                size = self._estimate_model_size(model_path)
                
                model_config = self._create_model_config(name, size)
                models.append(model_config)
        
        return models
    
    def _create_model_config(self, name: str, size: int) -> Dict[str, Any]:
        """Create a model configuration from name and size."""
        # Parse size in billions of parameters
        size_b = self._convert_to_billions(size)
        
        # Determine model type and capabilities
        model_type, capabilities = self._analyze_model_type(name)
        
        # Determine tool support based on model analysis
        supports_tools = self._determine_tool_support(name, model_type)
        
        return {
            "model_id": name,
            "size_b": size_b,
            "type": model_type,
            "capabilities": capabilities,
            "supports_tools": supports_tools,
            "description": f"Auto-discovered {model_type} model",
            "strengths": self._get_model_strengths(name, model_type),
            "use_cases": self._get_model_use_cases(model_type)
        }
    
    def _analyze_model_type(self, name: str) -> tuple[str, List[str]]:
        """Analyze model type based on name."""
        name_lower = name.lower()
        
        if any(keyword in name_lower for keyword in ['coder', 'code', 'deepcoder']):
            return "coding", ["coding", "analysis", "file_operations"]
        elif 'instruct' in name_lower:
            return "instruct", ["general_qa", "analysis", "reasoning"]
        elif 'chat' in name_lower:
            return "chat", ["conversation", "general_qa"]
        elif any(keyword in name_lower for keyword in ['llama', 'qwen', 'gemma']):
            return "general", ["general_qa", "analysis", "reasoning"]
        else:
            return "general", ["general_qa"]
    
    def _determine_tool_support(self, name: str, model_type: str) -> bool:
        """Determine if model supports tools based on analysis."""
        name_lower = name.lower()
        
        # Known models with tool support
        tool_supporting_patterns = [
            'qwen2.5', 'qwen2', 'llama3', 'gemma', 'mistral', 'codellama'
        ]
        
        # Known models without tool support
        non_tool_patterns = [
            'deepcoder', 'granite'
        ]
        
        # Check for non-tool patterns first
        for pattern in non_tool_patterns:
            if pattern in name_lower:
                return False
        
        # Check for tool-supporting patterns
        for pattern in tool_supporting_patterns:
            if pattern in name_lower:
                return True
        
        # Default based on model type
        return model_type in ["instruct", "general", "chat"]
    
    def _get_model_strengths(self, name: str, model_type: str) -> List[str]:
        """Get model strengths based on name and type."""
        if model_type == "coding":
            return ["Code generation", "Code analysis", "Technical documentation"]
        elif model_type == "instruct":
            return ["Instruction following", "Analysis", "Problem solving"]
        elif model_type == "chat":
            return ["Conversation", "General Q&A", "Creative tasks"]
        else:
            return ["General purpose", "Analysis", "Q&A"]
    
    def _get_model_use_cases(self, model_type: str) -> List[str]:
        """Get model use cases based on type."""
        if model_type == "coding":
            return ["code_analysis", "development", "debugging"]
        elif model_type == "instruct":
            return ["analysis", "qa", "reasoning"]
        elif model_type == "chat":
            return ["conversation", "creative", "qa"]
        else:
            return ["qa", "analysis", "general"]
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '4.1GB' to bytes."""
        if not size_str or size_str == "0B":
            return 0
        
        try:
            # Remove any non-numeric suffix and convert
            size_str = size_str.upper().replace('B', '')
            multipliers = {'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4}
            
            for suffix, multiplier in multipliers.items():
                if size_str.endswith(suffix):
                    number = float(size_str[:-1])
                    return int(number * multiplier)
            
            return int(float(size_str))
            
        except (ValueError, IndexError):
            return 0
    
    def _estimate_model_size(self, model_path: Path) -> int:
        """Estimate model size from folder contents."""
        total_size = 0
        try:
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except (OSError, PermissionError):
            pass
        return total_size
    
    def _convert_to_billions(self, size_bytes: int) -> float:
        """Convert size in bytes to billions of parameters (rough estimate)."""
        if size_bytes == 0:
            return 1.0  # Default size for unknown models
        
        # Rough conversion: 1B parameters ≈ 2GB for 16-bit models
        gb_size = size_bytes / (1024**3)
        return round(gb_size / 2.0, 1)
    
    def _get_fallback_models(self) -> List[Dict[str, Any]]:
        """Get fallback model configuration when discovery fails."""
        return [
            {
                "model_id": "llama3.2:latest",
                "size_b": 3.0,
                "type": "general",
                "capabilities": ["general_qa", "analysis"],
                "supports_tools": True,
                "description": "Fallback general model",
                "strengths": ["General purpose", "Analysis"],
                "use_cases": ["qa", "analysis"]
            },
            {
                "model_id": "qwen2.5:latest",
                "size_b": 7.0,
                "type": "instruct",
                "capabilities": ["general_qa", "analysis", "reasoning"],
                "supports_tools": True,
                "description": "Fallback instruction model",
                "strengths": ["Instruction following", "Analysis"],
                "use_cases": ["analysis", "qa", "reasoning"]
            }
        ]


def get_available_models(force_refresh: bool = False) -> List[Dict[str, Any]]:
    """Get available models with caching."""
    cache_file = Path("models_cache.json")
    
    # Use cache if available and not forcing refresh
    if not force_refresh and cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            # Check if cache is recent (within 1 hour)
            import time
            if time.time() - cached_data.get('timestamp', 0) < 3600:
                return cached_data.get('models', [])
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Discover models
    discovery = OllamaModelDiscovery()
    models = discovery.discover_models()
    
    # Cache the results
    try:
        import time
        cache_data = {
            'timestamp': time.time(),
            'models': models
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        get_logger().warning(f"Failed to cache models: {e}")
    
    return models


# Global helper functions for easier access in other modules
_discovery_instance = None

def get_discovery_instance() -> OllamaModelDiscovery:
    """Get a global instance of OllamaModelDiscovery."""
    global _discovery_instance
    if _discovery_instance is None:
        _discovery_instance = OllamaModelDiscovery()
    return _discovery_instance

def model_exists(model_name: str) -> bool:
    """Check if a model exists in the Ollama instance.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        True if model exists, False otherwise
    """
    discovery = get_discovery_instance()
    return discovery.model_exists(model_name)

def ensure_model_available(model_name: str, auto_pull: bool = True) -> bool:
    """Ensure a model is available, pulling it if necessary.
    
    Args:
        model_name: Name of the model to ensure
        auto_pull: Whether to attempt pulling the model if not found
        
    Returns:
        True if model is available after operation, False otherwise
    """
    discovery = get_discovery_instance()
    return discovery.ensure_model_available(model_name, auto_pull=auto_pull)

def get_available_models() -> List[str]:
    """Get list of all available model names.
    
    Returns:
        List of available model names
    """
    discovery = get_discovery_instance()
    return [model["model_id"] for model in discovery.discover_models()]


if __name__ == "__main__":
    # Test the discovery system
    discovery = OllamaModelDiscovery()
    models = discovery.discover_models()
    
    print(f"Discovered {len(models)} models:")
    for model in models:
        print(f"- {model['model_id']}: {model['size_b']}B, tools={model['supports_tools']}")
