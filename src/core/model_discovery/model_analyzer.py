"""Model analysis utilities for model discovery system."""

import re
from typing import Any, Dict, List, Tuple

from src.utils.enhanced_logging import get_logger


class ModelAnalyzer:
    """Analyzes model information and provides insights."""

    def __init__(self):
        """Initialize model analyzer."""
        self.logger = get_logger()
        
        # Model category mappings
        self.model_categories = {
            "code": ["codellama", "codegemma", "starcoder", "codestral", "deepseek-coder"],
            "chat": ["llama", "mistral", "gemma", "phi", "qwen", "yi"],
            "embedding": ["nomic-embed", "mxbai-embed", "all-minilm"],
            "vision": ["llava", "minicpm-v", "moondream"],
            "function": ["llama3-groq-tool-use", "firefunction"],
            "reasoning": ["deepseek-r1", "qwen2.5-coder"],
            "creative": ["storyteller", "creative-writer"]
        }
        
        # Performance tier mappings based on size
        self.performance_tiers = {
            "light": (0, 3),      # 0-3B parameters
            "medium": (3, 8),     # 3-8B parameters  
            "heavy": (8, 15),     # 8-15B parameters
            "enterprise": (15, 999)  # 15B+ parameters
        }

    def analyze_models(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze a list of models and provide insights.
        
        Args:
            models: List of model configurations
            
        Returns:
            Analysis results
        """
        if not models:
            return {"total": 0, "categories": {}, "tiers": {}, "recommendations": []}
        
        analysis = {
            "total": len(models),
            "categories": self._categorize_models(models),
            "tiers": self._tier_models(models),
            "size_stats": self._calculate_size_stats(models),
            "recommendations": self._generate_recommendations(models),
            "compatibility": self._assess_compatibility(models)
        }
        
        return analysis

    def _categorize_models(self, models: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Categorize models by type.
        
        Args:
            models: List of models
            
        Returns:
            Dictionary of categories with model lists
        """
        categories = {cat: [] for cat in self.model_categories.keys()}
        categories["other"] = []
        
        for model in models:
            name = model.get("name", "").lower()
            categorized = False
            
            for category, patterns in self.model_categories.items():
                if any(pattern in name for pattern in patterns):
                    categories[category].append(model["name"])
                    categorized = True
                    break
            
            if not categorized:
                categories["other"].append(model["name"])
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}

    def _tier_models(self, models: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Tier models by performance/size.
        
        Args:
            models: List of models
            
        Returns:
            Dictionary of tiers with model lists
        """
        tiers = {tier: [] for tier in self.performance_tiers.keys()}
        
        for model in models:
            size_b = model.get("size_b", 7.0)  # Default to 7B
            
            for tier, (min_size, max_size) in self.performance_tiers.items():
                if min_size <= size_b < max_size:
                    tiers[tier].append(model["name"])
                    break
        
        return {k: v for k, v in tiers.items() if v}

    def _calculate_size_stats(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate size statistics for models.
        
        Args:
            models: List of models
            
        Returns:
            Size statistics
        """
        sizes = [model.get("size_b", 7.0) for model in models]
        
        if not sizes:
            return {}
        
        return {
            "min_size": min(sizes),
            "max_size": max(sizes),
            "avg_size": sum(sizes) / len(sizes),
            "total_params": sum(sizes),
            "distribution": self._size_distribution(sizes)
        }

    def _size_distribution(self, sizes: List[float]) -> Dict[str, int]:
        """Calculate size distribution.
        
        Args:
            sizes: List of model sizes
            
        Returns:
            Distribution by size ranges
        """
        distribution = {
            "tiny (<1B)": 0,
            "small (1-3B)": 0,
            "medium (3-8B)": 0,
            "large (8-15B)": 0,
            "huge (15B+)": 0
        }
        
        for size in sizes:
            if size < 1:
                distribution["tiny (<1B)"] += 1
            elif size < 3:
                distribution["small (1-3B)"] += 1
            elif size < 8:
                distribution["medium (3-8B)"] += 1
            elif size < 15:
                distribution["large (8-15B)"] += 1
            else:
                distribution["huge (15B+)"] += 1
        
        return distribution

    def _generate_recommendations(self, models: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on model analysis.
        
        Args:
            models: List of models
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check for essential categories
        categories = self._categorize_models(models)
        
        if not categories.get("chat"):
            recommendations.append("Consider adding a general chat model like llama3.2 or mistral")
        
        if not categories.get("code"):
            recommendations.append("Consider adding a code model like codellama or codegemma")
        
        # Check size distribution
        tiers = self._tier_models(models)
        
        if not tiers.get("light"):
            recommendations.append("Consider adding a lightweight model (<3B) for faster responses")
        
        if not tiers.get("heavy") and not tiers.get("enterprise"):
            recommendations.append("Consider adding a larger model (8B+) for complex tasks")
        
        # Check for duplicates
        names = [model.get("name", "") for model in models]
        base_names = [self._extract_base_name(name) for name in names]
        duplicates = set([name for name in base_names if base_names.count(name) > 1])
        
        if duplicates:
            recommendations.append(f"Multiple versions detected for: {', '.join(duplicates)}")
        
        return recommendations

    def _assess_compatibility(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess model compatibility and requirements.
        
        Args:
            models: List of models
            
        Returns:
            Compatibility assessment
        """
        compatibility = {
            "memory_requirements": {},
            "gpu_recommendations": {},
            "cpu_only_suitable": []
        }
        
        for model in models:
            name = model.get("name", "")
            size_b = model.get("size_b", 7.0)
            
            # Estimate memory requirements (rough)
            memory_gb = size_b * 1.5  # Approximate memory usage
            
            compatibility["memory_requirements"][name] = {
                "estimated_ram": f"{memory_gb:.1f}GB",
                "recommended_ram": f"{memory_gb * 1.5:.1f}GB"
            }
            
            # GPU recommendations
            if size_b <= 3:
                compatibility["gpu_recommendations"][name] = "Any modern GPU with 4GB+ VRAM"
                compatibility["cpu_only_suitable"].append(name)
            elif size_b <= 7:
                compatibility["gpu_recommendations"][name] = "GPU with 8GB+ VRAM recommended"
            elif size_b <= 13:
                compatibility["gpu_recommendations"][name] = "High-end GPU with 16GB+ VRAM"
            else:
                compatibility["gpu_recommendations"][name] = "Multiple GPUs or 24GB+ VRAM required"
        
        return compatibility

    def _extract_base_name(self, model_name: str) -> str:
        """Extract base model name without version/tag.
        
        Args:
            model_name: Full model name
            
        Returns:
            Base model name
        """
        # Remove common version patterns
        name = re.sub(r':\w+', '', model_name)  # Remove :tag
        name = re.sub(r'-\d+\.?\d*[bB]', '', name)  # Remove size indicators
        name = re.sub(r'-v\d+', '', name)  # Remove version numbers
        name = re.sub(r'-\d+\.\d+', '', name)  # Remove decimal versions
        
        return name.lower()

    def find_model_by_capability(self, models: List[Dict[str, Any]], capability: str) -> List[str]:
        """Find models that match a specific capability.
        
        Args:
            models: List of models
            capability: Desired capability (code, chat, vision, etc.)
            
        Returns:
            List of matching model names
        """
        if capability not in self.model_categories:
            return []
        
        patterns = self.model_categories[capability]
        matching_models = []
        
        for model in models:
            name = model.get("name", "").lower()
            if any(pattern in name for pattern in patterns):
                matching_models.append(model["name"])
        
        return matching_models

    def recommend_best_model(self, models: List[Dict[str, Any]], 
                           task_type: str = "general", 
                           resource_constraint: str = "medium") -> str:
        """Recommend the best model for a specific task and resource constraint.
        
        Args:
            models: List of available models
            task_type: Type of task (general, code, vision, etc.)
            resource_constraint: Resource constraint (light, medium, heavy)
            
        Returns:
            Recommended model name or empty string if none found
        """
        if not models:
            return ""
        
        # Filter by task type
        if task_type in self.model_categories:
            task_models = self.find_model_by_capability(models, task_type)
            candidate_models = [m for m in models if m.get("name") in task_models]
        else:
            candidate_models = models
        
        if not candidate_models:
            candidate_models = models  # Fallback to all models
        
        # Filter by resource constraint
        if resource_constraint in self.performance_tiers:
            min_size, max_size = self.performance_tiers[resource_constraint]
            candidate_models = [m for m in candidate_models 
                              if min_size <= m.get("size_b", 7.0) < max_size]
        
        if not candidate_models:
            return ""
        
        # Sort by size (prefer larger within constraint for better performance)
        candidate_models.sort(key=lambda x: x.get("size_b", 7.0), reverse=True)
        
        return candidate_models[0].get("name", "")

    def get_model_insights(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed insights for a specific model.
        
        Args:
            model: Model configuration
            
        Returns:
            Model insights
        """
        name = model.get("name", "")
        size_b = model.get("size_b", 7.0)
        
        insights = {
            "name": name,
            "category": self._get_model_category(name),
            "tier": self._get_model_tier(size_b),
            "capabilities": self._infer_capabilities(name),
            "use_cases": self._suggest_use_cases(name),
            "performance_notes": self._get_performance_notes(size_b),
            "alternatives": self._suggest_alternatives(name, size_b)
        }
        
        return insights

    def _get_model_category(self, name: str) -> str:
        """Get the category of a model.
        
        Args:
            name: Model name
            
        Returns:
            Model category
        """
        name_lower = name.lower()
        for category, patterns in self.model_categories.items():
            if any(pattern in name_lower for pattern in patterns):
                return category
        return "general"

    def _get_model_tier(self, size_b: float) -> str:
        """Get the performance tier of a model.
        
        Args:
            size_b: Model size in billions of parameters
            
        Returns:
            Performance tier
        """
        for tier, (min_size, max_size) in self.performance_tiers.items():
            if min_size <= size_b < max_size:
                return tier
        return "medium"

    def _infer_capabilities(self, name: str) -> List[str]:
        """Infer capabilities from model name.
        
        Args:
            name: Model name
            
        Returns:
            List of capabilities
        """
        capabilities = []
        name_lower = name.lower()
        
        if any(pattern in name_lower for pattern in self.model_categories["code"]):
            capabilities.extend(["code_generation", "code_explanation", "debugging"])
        
        if any(pattern in name_lower for pattern in self.model_categories["vision"]):
            capabilities.extend(["image_understanding", "visual_reasoning"])
        
        if any(pattern in name_lower for pattern in self.model_categories["chat"]):
            capabilities.extend(["conversation", "question_answering", "reasoning"])
        
        if "instruct" in name_lower or "chat" in name_lower:
            capabilities.append("instruction_following")
        
        return capabilities or ["general_language_modeling"]

    def _suggest_use_cases(self, name: str) -> List[str]:
        """Suggest use cases for a model.
        
        Args:
            name: Model name
            
        Returns:
            List of suggested use cases
        """
        category = self._get_model_category(name)
        
        use_case_map = {
            "code": ["Software development", "Code review", "API documentation", "Bug fixing"],
            "chat": ["Customer support", "General Q&A", "Content creation", "Research assistance"],
            "vision": ["Image analysis", "Document OCR", "Visual content moderation"],
            "embedding": ["Semantic search", "Document similarity", "Clustering"],
            "function": ["Tool usage", "API integration", "Structured output"],
            "reasoning": ["Complex problem solving", "Mathematical reasoning", "Logic puzzles"],
            "creative": ["Story writing", "Content generation", "Creative brainstorming"]
        }
        
        return use_case_map.get(category, ["General language tasks", "Text processing"])

    def _get_performance_notes(self, size_b: float) -> List[str]:
        """Get performance notes based on model size.
        
        Args:
            size_b: Model size in billions of parameters
            
        Returns:
            List of performance notes
        """
        notes = []
        
        if size_b < 1:
            notes.extend(["Very fast inference", "Low memory usage", "May have limited capabilities"])
        elif size_b < 3:
            notes.extend(["Fast inference", "Moderate memory usage", "Good for simple tasks"])
        elif size_b < 8:
            notes.extend(["Balanced speed/quality", "Standard memory usage", "Versatile performance"])
        elif size_b < 15:
            notes.extend(["High quality output", "Higher memory usage", "Slower inference"])
        else:
            notes.extend(["Excellent quality", "High memory requirements", "Requires powerful hardware"])
        
        return notes

    def _suggest_alternatives(self, name: str, size_b: float) -> List[str]:
        """Suggest alternative models.
        
        Args:
            name: Current model name
            size_b: Current model size
            
        Returns:
            List of alternative model suggestions
        """
        category = self._get_model_category(name)
        tier = self._get_model_tier(size_b)
        
        # Common alternatives by category
        alternatives_map = {
            "code": ["codellama:7b", "codegemma:7b", "deepseek-coder:6.7b"],
            "chat": ["llama3.2:3b", "mistral:7b", "gemma2:9b"],
            "vision": ["llava:7b", "minicpm-v:8b"],
            "embedding": ["nomic-embed-text", "all-minilm:l6-v2"]
        }
        
        base_alternatives = alternatives_map.get(category, ["llama3.2:3b", "mistral:7b"])
        
        # Filter out the current model
        current_base = self._extract_base_name(name)
        alternatives = [alt for alt in base_alternatives 
                       if self._extract_base_name(alt) != current_base]
        
        return alternatives[:3]  # Return top 3 alternatives
