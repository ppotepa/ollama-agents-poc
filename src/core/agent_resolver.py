"""AgentResolver - Automatically selects the best agent/model for a given prompt."""
from __future__ import annotations

import re
import yaml
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from src.utils.enhanced_logging import get_logger


@dataclass
class ModelCandidate:
    """Represents a model candidate with scoring information."""
    model_id: str
    name: str
    config: Dict[str, Any]
    score: float
    reasoning: List[str]
    size_param: Optional[str] = None
    
    @property
    def size_in_billions(self) -> float:
        """Extract model size in billions of parameters."""
        if not self.size_param:
            # Try to extract from model_id or name
            text = f"{self.model_id} {self.name}".lower()
            # Look for patterns like "7b", "13b", "70b", etc.
            size_match = re.search(r'(\d+(?:\.\d+)?)b(?:\D|$)', text)
            if size_match:
                return float(size_match.group(1))
            return 0.0
        
        # Parse explicit size parameter
        if isinstance(self.size_param, str):
            size_match = re.search(r'(\d+(?:\.\d+)?)', self.size_param.lower())
            if size_match:
                return float(size_match.group(1))
        return 0.0


class AgentResolver:
    """Automatically resolves the best agent/model for a given prompt."""
    
    def __init__(self, models_config_path: Optional[str] = None, max_size_b: float = 14.0):
        """Initialize the agent resolver.
        
        Args:
            models_config_path: Path to models.yaml configuration file
            max_size_b: Maximum model size in billions of parameters
        """
        self.logger = get_logger()
        self.max_size_b = max_size_b
        self.models_config = self._load_models_config(models_config_path)
        
        # Define intent patterns and their preferred model types
        self.intent_patterns = {
            # Coding-related patterns
            'coding': {
                'patterns': [
                    r'\b(?:code|coding|program|script|function|class|method|algorithm|debug|refactor|implement|api)\b',
                    r'\b(?:python|javascript|java|c\+\+|typescript|rust|go|php|ruby|swift)\b',
                    r'\b(?:fix|error|bug|exception|traceback|syntax|compile)\b',
                    r'\b(?:write|create|build|develop|generate).*(?:code|script|program|function)\b',
                    r'\b(?:optimize|performance|efficiency|speed up)\b.*\b(?:code|algorithm)\b'
                ],
                'preferred_models': ['deepcoder', 'qwen2.5-coder', 'codellama'],
                'weight': 3.0
            },
            
            # File operations
            'file_ops': {
                'patterns': [
                    r'\b(?:file|folder|directory|path|read|write|list|search|find)\b',
                    r'\b(?:analyze|examine|explore|scan|index).*(?:project|repository|codebase)\b',
                    r'\b(?:structure|organization|hierarchy|tree)\b'
                ],
                'preferred_models': ['deepcoder', 'qwen2.5-coder'],
                'weight': 2.5
            },
            
            # General question answering
            'qa': {
                'patterns': [
                    r'\b(?:what|how|why|when|where|who|explain|describe|tell me)\b',
                    r'\b(?:question|answer|help|assist|guide|tutorial)\b',
                    r'\b(?:understand|clarify|meaning|definition)\b'
                ],
                'preferred_models': ['qwen2.5', 'gemma', 'mistral'],
                'weight': 1.5
            },
            
            # Documentation and analysis
            'analysis': {
                'patterns': [
                    r'\b(?:analyze|analysis|examine|review|assess|evaluate)\b',
                    r'\b(?:document|documentation|readme|guide|manual)\b',
                    r'\b(?:summary|summarize|overview|report)\b'
                ],
                'preferred_models': ['qwen2.5', 'deepcoder', 'gemma'],
                'weight': 2.0
            },
            
            # Creative/text generation
            'creative': {
                'patterns': [
                    r'\b(?:write|create|generate|compose|draft)\b.*(?:text|content|article|blog|story)\b',
                    r'\b(?:creative|imagination|brainstorm|idea)\b',
                    r'\b(?:poem|poetry|narrative|fiction)\b'
                ],
                'preferred_models': ['gemma', 'qwen2.5', 'mistral'],
                'weight': 1.8
            }
        }
    
    def _load_models_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load models configuration from YAML file."""
        if not config_path:
            # Default path relative to this file
            config_path = Path(__file__).parent.parent / "config" / "models.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self.logger.info(f"Loaded models configuration from {config_path}")
                return config.get('models', {})
        except Exception as e:
            self.logger.error(f"Failed to load models config from {config_path}: {e}")
            return {}
    
    def _analyze_intent(self, prompt: str) -> Dict[str, float]:
        """Analyze the intent of the prompt and return intent scores."""
        prompt_lower = prompt.lower()
        intent_scores = {}
        
        for intent_name, intent_config in self.intent_patterns.items():
            score = 0.0
            patterns = intent_config['patterns']
            weight = intent_config['weight']
            
            for pattern in patterns:
                matches = len(re.findall(pattern, prompt_lower, re.IGNORECASE))
                if matches > 0:
                    # Score based on number of matches and pattern weight
                    score += matches * weight * 0.1
            
            intent_scores[intent_name] = score
        
        return intent_scores
    
    def _score_model_for_intent(self, model_id: str, model_config: Dict[str, Any], 
                               intent_scores: Dict[str, float]) -> Tuple[float, List[str]]:
        """Score a model based on how well it matches the detected intents."""
        total_score = 0.0
        reasoning = []
        
        # Get model capabilities
        capabilities = model_config.get('capabilities', {})
        model_name_lower = model_id.lower()
        
        # Base scoring from intent patterns
        for intent_name, intent_score in intent_scores.items():
            if intent_score > 0:
                intent_config = self.intent_patterns[intent_name]
                preferred_models = intent_config['preferred_models']
                
                # Check if this model is preferred for this intent
                model_preference_score = 0.0
                for preferred in preferred_models:
                    if preferred.lower() in model_name_lower:
                        model_preference_score = 1.0
                        reasoning.append(f"Preferred for {intent_name} tasks")
                        break
                    elif any(preferred_part in model_name_lower for preferred_part in preferred.split('-')):
                        model_preference_score = 0.7
                        reasoning.append(f"Good match for {intent_name} tasks")
                        break
                
                total_score += intent_score * model_preference_score
        
        # Capability-based scoring
        if capabilities.get('coding', False):
            coding_score = intent_scores.get('coding', 0) + intent_scores.get('file_ops', 0)
            if coding_score > 0:
                total_score += coding_score * 0.5
                reasoning.append("Has coding capabilities")
        
        if capabilities.get('file_operations', False):
            file_ops_score = intent_scores.get('file_ops', 0)
            if file_ops_score > 0:
                total_score += file_ops_score * 0.3
                reasoning.append("Has file operations capabilities")
        
        if capabilities.get('general_qa', False):
            qa_score = intent_scores.get('qa', 0) + intent_scores.get('analysis', 0)
            if qa_score > 0:
                total_score += qa_score * 0.2
                reasoning.append("Good for general Q&A")
        
        # Model-specific bonuses
        if 'coder' in model_name_lower:
            coding_total = intent_scores.get('coding', 0) + intent_scores.get('file_ops', 0)
            if coding_total > 0:
                total_score += coding_total * 0.3
                reasoning.append("Specialized coding model")
        
        if 'deepcoder' in model_name_lower:
            total_score += 0.2  # Small bonus for primary model
            reasoning.append("Primary development model")
        
        return total_score, reasoning
    
    def _filter_by_size(self, candidates: List[ModelCandidate]) -> List[ModelCandidate]:
        """Filter candidates by maximum size constraint."""
        filtered = []
        for candidate in candidates:
            size = candidate.size_in_billions
            if size <= self.max_size_b or size == 0.0:  # Include unknown sizes
                filtered.append(candidate)
            else:
                self.logger.debug(f"Filtered out {candidate.model_id} ({size}B > {self.max_size_b}B)")
        
        return filtered
    
    def resolve_best_agent(self, prompt: str) -> Optional[str]:
        """Resolve the best agent/model for the given prompt.
        
        Args:
            prompt: The user's prompt/query
            
        Returns:
            The model_id of the best suited agent, or None if no suitable agent found
        """
        if not self.models_config:
            self.logger.warning("No models configuration available")
            return None
        
        # Analyze the prompt intent
        intent_scores = self._analyze_intent(prompt)
        self.logger.debug(f"Intent analysis: {intent_scores}")
        
        # Score all models
        candidates = []
        for model_key, model_config in self.models_config.items():
            model_id = model_config.get('model_id', model_key)
            model_name = model_config.get('name', model_key)
            
            score, reasoning = self._score_model_for_intent(model_id, model_config, intent_scores)
            
            if score > 0:  # Only include models with some relevance
                candidate = ModelCandidate(
                    model_id=model_id,
                    name=model_name,
                    config=model_config,
                    score=score,
                    reasoning=reasoning,
                    size_param=model_config.get('size')
                )
                candidates.append(candidate)
        
        if not candidates:
            self.logger.warning("No suitable candidates found for prompt")
            return None
        
        # Filter by size constraint
        candidates = self._filter_by_size(candidates)
        
        if not candidates:
            self.logger.warning(f"No candidates under {self.max_size_b}B size limit")
            return None
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        # Log the selection process
        best_candidate = candidates[0]
        self.logger.info(f"Selected model: {best_candidate.model_id} (score: {best_candidate.score:.2f})")
        self.logger.info(f"Reasoning: {', '.join(best_candidate.reasoning)}")
        
        if len(candidates) > 1:
            self.logger.debug("Other candidates:")
            for i, candidate in enumerate(candidates[1:4], 1):  # Show top 3 alternatives
                self.logger.debug(f"  {i+1}. {candidate.model_id} (score: {candidate.score:.2f})")
        
        return best_candidate.model_id
    
    def get_model_recommendations(self, prompt: str, top_n: int = 3) -> List[ModelCandidate]:
        """Get top N model recommendations for a prompt.
        
        Args:
            prompt: The user's prompt/query
            top_n: Number of top recommendations to return
            
        Returns:
            List of top ModelCandidate objects
        """
        if not self.models_config:
            return []
        
        intent_scores = self._analyze_intent(prompt)
        candidates = []
        
        for model_key, model_config in self.models_config.items():
            model_id = model_config.get('model_id', model_key)
            model_name = model_config.get('name', model_key)
            
            score, reasoning = self._score_model_for_intent(model_id, model_config, intent_scores)
            
            candidate = ModelCandidate(
                model_id=model_id,
                name=model_name,
                config=model_config,
                score=score,
                reasoning=reasoning,
                size_param=model_config.get('size')
            )
            candidates.append(candidate)
        
        # Filter by size and sort
        candidates = self._filter_by_size(candidates)
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        return candidates[:top_n]


def create_agent_resolver(max_size_b: float = 14.0) -> AgentResolver:
    """Create an AgentResolver instance with the specified constraints."""
    return AgentResolver(max_size_b=max_size_b)


__all__ = ["AgentResolver", "ModelCandidate", "create_agent_resolver"]
