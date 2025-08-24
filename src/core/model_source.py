"""Model source management for dynamic model discovery."""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about a discovered model."""
    name: str
    path: Optional[str] = None
    size: Optional[str] = None
    family: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class ModelSource:
    """Manages model discovery from various sources."""
    
    def __init__(self, base_path: str = "."):
        """Initialize model source manager.
        
        Args:
            base_path: Base directory to look for model sources
        """
        self.base_path = Path(base_path)
        self.models_txt_path = self.base_path / "models.txt"
        self.models_dir_path = self.base_path / "models"
    
    def discover_models(self) -> List[ModelInfo]:
        """Discover models from available sources.
        
        Returns:
            List of discovered models
        """
        if self.models_txt_path.exists():
            print(f"📄 Reading models from {self.models_txt_path}")
            return self._read_from_models_txt()
        elif self.models_dir_path.exists():
            print(f"📁 Scanning models directory: {self.models_dir_path}")
            return self._scan_models_directory()
        else:
            print("⚠️ No model source found, using default model list")
            return self._get_default_models()
    
    def _read_from_models_txt(self) -> List[ModelInfo]:
        """Read models from models.txt file.
        
        Expected format:
        - Simple format: one model name per line
        - JSON format: JSON array of model objects
        - Extended format: name|family|description
        
        Returns:
            List of model information
        """
        models = []
        
        try:
            content = self.models_txt_path.read_text(encoding='utf-8').strip()
            
            # Try to parse as JSON first
            if content.startswith('[') or content.startswith('{'):
                try:
                    json_data = json.loads(content)
                    if isinstance(json_data, list):
                        for item in json_data:
                            if isinstance(item, str):
                                models.append(ModelInfo(name=item))
                            elif isinstance(item, dict):
                                models.append(ModelInfo(
                                    name=item.get('name', ''),
                                    family=item.get('family'),
                                    description=item.get('description'),
                                    size=item.get('size'),
                                    tags=item.get('tags', [])
                                ))
                    return models
                except json.JSONDecodeError:
                    pass  # Fall back to line-based parsing
            
            # Parse line by line
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Check for extended format: name|family|description
                if '|' in line:
                    parts = [p.strip() for p in line.split('|')]
                    name = parts[0] if len(parts) > 0 else ''
                    family = parts[1] if len(parts) > 1 else None
                    description = parts[2] if len(parts) > 2 else None
                    if name:
                        models.append(ModelInfo(
                            name=name,
                            family=family,
                            description=description
                        ))
                else:
                    # Simple format: just model name
                    models.append(ModelInfo(name=line))
                    
        except Exception as e:
            print(f"⚠️ Error reading models.txt: {e}")
            return self._get_default_models()
        
        return models
    
    def _scan_models_directory(self) -> List[ModelInfo]:
        """Scan the models directory for available models.
        
        Returns:
            List of discovered models from directory
        """
        models = []
        
        try:
            # Look for Ollama-style structure (models/manifests/)
            manifests_path = self.models_dir_path / "models" / "manifests"
            if manifests_path.exists():
                models.extend(self._scan_ollama_manifests(manifests_path))
            
            # Look for direct model files
            for item in self.models_dir_path.iterdir():
                if item.is_file() and not item.name.startswith('.'):
                    # Skip common non-model files
                    if item.name in ['history', 'id_ed25519', 'id_ed25519.pub']:
                        continue
                    
                    size = self._format_file_size(item.stat().st_size)
                    models.append(ModelInfo(
                        name=item.stem,
                        path=str(item),
                        size=size
                    ))
            
            # If no models found, scan subdirectories
            if not models:
                for subdir in self.models_dir_path.iterdir():
                    if subdir.is_dir() and not subdir.name.startswith('.'):
                        models.append(ModelInfo(
                            name=subdir.name,
                            path=str(subdir)
                        ))
                        
        except Exception as e:
            print(f"⚠️ Error scanning models directory: {e}")
            return self._get_default_models()
        
        return models if models else self._get_default_models()
    
    def _scan_ollama_manifests(self, manifests_path: Path) -> List[ModelInfo]:
        """Scan Ollama manifest files for model information.
        
        Args:
            manifests_path: Path to Ollama manifests directory
            
        Returns:
            List of models found in manifests
        """
        models = []
        
        try:
            # Ollama stores manifests in subdirectories by registry/namespace/model
            for registry_dir in manifests_path.iterdir():
                if not registry_dir.is_dir():
                    continue
                    
                for namespace_dir in registry_dir.iterdir():
                    if not namespace_dir.is_dir():
                        continue
                        
                    for model_dir in namespace_dir.iterdir():
                        if not model_dir.is_dir():
                            continue
                            
                        # Look for manifest files (usually named with sha256)
                        for manifest_file in model_dir.iterdir():
                            if manifest_file.is_file():
                                model_name = f"{namespace_dir.name}/{model_dir.name}"
                                if registry_dir.name != "registry.ollama.ai":
                                    model_name = f"{registry_dir.name}/{model_name}"
                                
                                models.append(ModelInfo(
                                    name=model_name,
                                    path=str(manifest_file),
                                    family=self._guess_model_family(model_name)
                                ))
                                break  # One manifest per model
                                
        except Exception as e:
            print(f"⚠️ Error scanning Ollama manifests: {e}")
        
        return models
    
    def _guess_model_family(self, model_name: str) -> str:
        """Guess model family from model name.
        
        Args:
            model_name: Model name to analyze
            
        Returns:
            Guessed model family
        """
        name_lower = model_name.lower()
        
        if "llama" in name_lower:
            return "codellama" if "code" in name_lower else "llama"
        elif "qwen" in name_lower:
            return "qwen-coder" if "coder" in name_lower else "qwen"
        elif "mistral" in name_lower:
            return "mistral"
        elif "gemma" in name_lower:
            return "gemma"
        elif "phi" in name_lower:
            return "phi"
        elif "deepseek" in name_lower:
            return "deepseek-coder" if "coder" in name_lower else "deepseek"
        elif "tiny" in name_lower:
            return "tinyllama"
        
        return "unknown"
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Formatted size string
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}PB"
    
    def _get_default_models(self) -> List[ModelInfo]:
        """Get default model list when no other source is available.
        
        Returns:
            List of default models
        """
        return [
            ModelInfo(name="deepcoder:14b", family="deepseek", description="Main coding assistant"),
            ModelInfo(name="llama3.3:70b-q2_k", family="llama", description="High capability, memory efficient"),
            ModelInfo(name="llama3.3:70b-q3_k_m", family="llama", description="Maximum quality"),
            ModelInfo(name="codellama:13b-instruct", family="codellama", description="Meta's coding specialist"),
            ModelInfo(name="deepseek-coder:6.7b", family="deepseek", description="Compact coding model"),
            ModelInfo(name="qwen2.5-coder:7b", family="qwen", description="Multilingual coding"),
            ModelInfo(name="qwen2.5:3b-instruct", family="qwen", description="Compact multilingual"),
            ModelInfo(name="gemma:7b-instruct", family="gemma", description="Google's efficient model"),
            ModelInfo(name="mistral:7b-instruct", family="mistral", description="Efficient conversational"),
            ModelInfo(name="phi3:mini", family="phi", description="Microsoft's compact model"),
            ModelInfo(name="tinyllama", family="tinyllama", description="Minimal testing model"),
        ]
    
    def create_sample_models_txt(self) -> None:
        """Create a sample models.txt file with documentation."""
        sample_content = """# Models Configuration File
# 
# This file defines the available models for the Ollama Agent system.
# You can use different formats:
#
# 1. Simple format (one model per line):
# deepcoder:14b
# llama3.3:70b
#
# 2. Extended format (name|family|description):
# deepcoder:14b|deepseek|Main coding assistant
# llama3.3:70b|llama|High capability model
#
# 3. JSON format (array of objects):
# [
#   {
#     "name": "deepcoder:14b",
#     "family": "deepseek", 
#     "description": "Main coding assistant",
#     "size": "14B",
#     "tags": ["coding", "primary"]
#   }
# ]

# Primary coding model
deepcoder:14b|deepseek|Main coding assistant

# Large language models  
llama3.3:70b-q2_k|llama|High capability, memory efficient
llama3.3:70b-q3_k_m|llama|Maximum quality

# Specialized coding models
codellama:13b-instruct|codellama|Meta's coding specialist
deepseek-coder:6.7b|deepseek|Compact coding model
qwen2.5-coder:7b|qwen|Multilingual coding

# Efficient general models
qwen2.5:3b-instruct|qwen|Compact multilingual
gemma:7b-instruct|gemma|Google's efficient model
mistral:7b-instruct|mistral|Efficient conversational
phi3:mini|phi|Microsoft's compact model

# Testing model
tinyllama|tinyllama|Minimal testing model
"""
        
        try:
            self.models_txt_path.write_text(sample_content, encoding='utf-8')
            print(f"✓ Created sample models.txt at {self.models_txt_path}")
        except Exception as e:
            print(f"✗ Failed to create models.txt: {e}")


def get_available_models(base_path: str = ".") -> List[ModelInfo]:
    """Convenience function to get available models.
    
    Args:
        base_path: Base directory to search for models
        
    Returns:
        List of available models
    """
    source = ModelSource(base_path)
    return source.discover_models()


if __name__ == "__main__":
    # Test the model source discovery
    models = get_available_models()
    print(f"\nFound {len(models)} models:")
    for model in models:
        print(f"  - {model.name}")
        if model.family:
            print(f"    Family: {model.family}")
        if model.description:
            print(f"    Description: {model.description}")
        if model.size:
            print(f"    Size: {model.size}")
        print()
