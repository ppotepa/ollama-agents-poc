"""File system-based model discovery for local Ollama installations."""

from pathlib import Path
from typing import Any, Dict, List

from src.utils.enhanced_logging import get_logger


class FileSystemModelDiscovery:
    """Discovers models by scanning the local file system."""

    def __init__(self):
        """Initialize file system discovery."""
        self.logger = get_logger()
        
        # Common Ollama model storage locations
        self.model_paths = [
            Path.home() / ".ollama" / "models",
            Path("/usr/share/ollama/models"),
            Path("/opt/ollama/models"),
            Path("./models"),  # Local models directory
        ]

    def discover_models(self) -> List[Dict[str, Any]]:
        """Discover models by scanning file system.
        
        Returns:
            List of model configurations
        """
        try:
            models = []
            
            for path in self.model_paths:
                if path.exists() and path.is_dir():
                    models.extend(self._scan_models_directory(path))
            
            # Remove duplicates based on name
            unique_models = {}
            for model in models:
                name = model["name"]
                if name not in unique_models:
                    unique_models[name] = model
            
            result = list(unique_models.values())
            self.logger.info(f"Discovered {len(result)} models via file system")
            return result
            
        except Exception as e:
            self.logger.error(f"File system discovery failed: {e}")
            return []

    def _scan_models_directory(self, models_dir: Path) -> List[Dict[str, Any]]:
        """Scan a specific models directory.
        
        Args:
            models_dir: Path to models directory
            
        Returns:
            List of models found in directory
        """
        models = []
        
        try:
            # Look for manifests directory
            manifests_dir = models_dir / "manifests"
            if manifests_dir.exists():
                models.extend(self._scan_manifests_directory(manifests_dir))
            
            # Look for blobs directory (model files)
            blobs_dir = models_dir / "blobs"
            if blobs_dir.exists():
                models.extend(self._scan_blobs_directory(blobs_dir))
            
        except Exception as e:
            self.logger.debug(f"Error scanning {models_dir}: {e}")
        
        return models

    def _scan_manifests_directory(self, manifests_dir: Path) -> List[Dict[str, Any]]:
        """Scan manifests directory for model information.
        
        Args:
            manifests_dir: Path to manifests directory
            
        Returns:
            List of models from manifests
        """
        models = []
        
        try:
            # Look for registry.ollama.ai subdirectory
            registry_dir = manifests_dir / "registry.ollama.ai"
            if registry_dir.exists():
                for library_dir in registry_dir.iterdir():
                    if library_dir.is_dir():
                        models.extend(self._scan_library_directory(library_dir))
            
        except Exception as e:
            self.logger.debug(f"Error scanning manifests {manifests_dir}: {e}")
        
        return models

    def _scan_library_directory(self, library_dir: Path) -> List[Dict[str, Any]]:
        """Scan a library directory for models.
        
        Args:
            library_dir: Path to library directory
            
        Returns:
            List of models in library
        """
        models = []
        
        try:
            for model_dir in library_dir.iterdir():
                if model_dir.is_dir():
                    model_name = f"{library_dir.name}/{model_dir.name}"
                    
                    # Look for tag files
                    for tag_file in model_dir.iterdir():
                        if tag_file.is_file():
                            tag_name = tag_file.name
                            full_name = f"{model_name}:{tag_name}"
                            
                            model_info = self._create_model_info(full_name, tag_file)
                            if model_info:
                                models.append(model_info)
            
        except Exception as e:
            self.logger.debug(f"Error scanning library {library_dir}: {e}")
        
        return models

    def _scan_blobs_directory(self, blobs_dir: Path) -> List[Dict[str, Any]]:
        """Scan blobs directory for model files.
        
        Args:
            blobs_dir: Path to blobs directory
            
        Returns:
            List of models from blobs
        """
        models = []
        
        try:
            # Count blob files to estimate models
            blob_files = list(blobs_dir.glob("*"))
            
            # Group blobs by prefix (rough model estimation)
            blob_groups = {}
            for blob_file in blob_files:
                if blob_file.is_file():
                    prefix = blob_file.name[:8]  # First 8 chars
                    if prefix not in blob_groups:
                        blob_groups[prefix] = []
                    blob_groups[prefix].append(blob_file)
            
            # Create model entries for larger blob groups
            for prefix, files in blob_groups.items():
                if len(files) > 1:  # Likely a complete model
                    total_size = sum(f.stat().st_size for f in files if f.exists())
                    
                    model_info = {
                        "name": f"blob_model_{prefix}",
                        "size": total_size,
                        "size_b": self._convert_to_billions(total_size),
                        "source": "filesystem_blobs",
                        "files": len(files)
                    }
                    models.append(model_info)
            
        except Exception as e:
            self.logger.debug(f"Error scanning blobs {blobs_dir}: {e}")
        
        return models

    def _create_model_info(self, model_name: str, tag_file: Path) -> Dict[str, Any]:
        """Create model info from tag file.
        
        Args:
            model_name: Name of the model
            tag_file: Path to tag file
            
        Returns:
            Model information dictionary
        """
        try:
            file_size = tag_file.stat().st_size if tag_file.exists() else 0
            
            return {
                "name": model_name,
                "size": file_size,
                "size_b": self._estimate_model_size_from_name(model_name),
                "source": "filesystem",
                "tag_file": str(tag_file),
                "modified": tag_file.stat().st_mtime if tag_file.exists() else 0
            }
            
        except Exception as e:
            self.logger.debug(f"Error creating model info for {model_name}: {e}")
            return None

    def _estimate_model_size_from_name(self, model_name: str) -> float:
        """Estimate model size from name patterns.
        
        Args:
            model_name: Model name
            
        Returns:
            Estimated size in billions of parameters
        """
        # Extract size indicators from name
        size_patterns = {
            "1b": 1.0, "2b": 2.0, "3b": 3.0, "6.7b": 6.7, "7b": 7.0,
            "8b": 8.0, "13b": 13.0, "14b": 14.0, "30b": 30.0, "70b": 70.0
        }
        
        model_lower = model_name.lower()
        for pattern, size in size_patterns.items():
            if pattern in model_lower:
                return size
        
        return 7.0  # Default size

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

    def get_models_directory_info(self) -> Dict[str, Any]:
        """Get information about models directories.
        
        Returns:
            Dictionary with directory information
        """
        info = {
            "searched_paths": [str(p) for p in self.model_paths],
            "existing_paths": [],
            "total_models": 0
        }
        
        for path in self.model_paths:
            if path.exists():
                info["existing_paths"].append({
                    "path": str(path),
                    "size": self._get_directory_size(path),
                    "is_dir": path.is_dir()
                })
        
        return info

    def _get_directory_size(self, directory: Path) -> int:
        """Get total size of a directory.
        
        Args:
            directory: Directory path
            
        Returns:
            Total size in bytes
        """
        try:
            return sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
        except Exception:
            return 0
