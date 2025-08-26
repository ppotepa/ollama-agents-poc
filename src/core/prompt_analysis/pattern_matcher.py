"""Pattern recognition for prompt analysis and classification."""

import re
from enum import Enum
from typing import Dict, List, Any, Tuple


class PromptType(Enum):
    """Types of prompts that can be intercepted and processed."""
    REPOSITORY_ANALYSIS = "repository_analysis"
    CODE_ANALYSIS = "code_analysis"
    FILE_OPERATIONS = "file_operations"
    GENERAL_CODING = "general_coding"
    QUESTION_ANSWERING = "question_answering"
    PROJECT_NAVIGATION = "project_navigation"
    TECHNOLOGY_INQUIRY = "technology_inquiry"
    ARCHITECTURE_REVIEW = "architecture_review"
    DEBUGGING = "debugging"
    GENERAL = "general"


class PromptPatternMatcher:
    """Handles pattern matching and prompt classification."""

    def __init__(self):
        """Initialize the pattern matcher."""
        self.prompt_patterns = self._build_prompt_patterns()

    def _build_prompt_patterns(self) -> Dict[PromptType, Dict[str, Any]]:
        """Build patterns for recognizing different types of prompts."""
        return {
            PromptType.REPOSITORY_ANALYSIS: {
                "patterns": [
                    r"analyz[e|ing]*\s+repositor[y|ies]*",
                    r"repositor[y|ies]*\s+analys[is|e]*",
                    r"analyz[e|ing]*\s+project\s+structure",
                    r"project\s+structure",
                    r"code\s+structure",
                    r"codebase\s+analys[is|e]*",
                    r"what.*language.*project",
                    r"what.*technologies.*used",
                    r"project\s+overview",
                    r"repository\s+structure",
                    r"file\s+structure",
                    r"directory\s+structure",
                    r"how\s+is.*organized",
                    r"show.*project.*layout",
                    r"project.*composition"
                ],
                "required_context": ["repository_structure", "languages", "technologies"],
                "confidence_threshold": 0.7
            },

            PromptType.CODE_ANALYSIS: {
                "patterns": [
                    r"analyz[e|ing]*\s+code",
                    r"code\s+analys[is|e]*",
                    r"review\s+code",
                    r"code\s+review",
                    r"explain.*code",
                    r"understand.*code",
                    r"what.*code.*do",
                    r"how.*code.*work",
                    r"code\s+quality",
                    r"refactor",
                    r"optimize.*code",
                    r"improve.*code",
                    r"code\s+smell",
                    r"technical\s+debt"
                ],
                "required_context": ["file_content", "code_structure", "dependencies"],
                "confidence_threshold": 0.6
            },

            PromptType.FILE_OPERATIONS: {
                "patterns": [
                    r"create.*file",
                    r"make.*file",
                    r"new.*file",
                    r"add.*file",
                    r"modify.*file",
                    r"edit.*file",
                    r"update.*file",
                    r"change.*file",
                    r"delete.*file",
                    r"remove.*file",
                    r"rename.*file",
                    r"move.*file",
                    r"copy.*file",
                    r"find.*file",
                    r"search.*file",
                    r"list.*file"
                ],
                "required_context": ["file_structure", "current_files"],
                "confidence_threshold": 0.5
            },

            PromptType.GENERAL_CODING: {
                "patterns": [
                    r"write.*function",
                    r"create.*function",
                    r"implement",
                    r"build.*class",
                    r"create.*class",
                    r"add.*method",
                    r"write.*method",
                    r"fix.*bug",
                    r"debug",
                    r"solve.*problem",
                    r"write.*script",
                    r"create.*script",
                    r"automate",
                    r"generate.*code"
                ],
                "required_context": ["code_context", "project_structure"],
                "confidence_threshold": 0.6
            },

            PromptType.TECHNOLOGY_INQUIRY: {
                "patterns": [
                    r"what.*framework",
                    r"what.*library",
                    r"what.*technology",
                    r"what.*language",
                    r"which.*framework",
                    r"which.*library",
                    r"which.*technology",
                    r"how.*configured",
                    r"what.*version",
                    r"dependencies",
                    r"requirements",
                    r"package.*json",
                    r"requirements.*txt",
                    r"pom.*xml",
                    r"cargo.*toml"
                ],
                "required_context": ["technologies", "dependencies", "config_files"],
                "confidence_threshold": 0.6
            },

            PromptType.DEBUGGING: {
                "patterns": [
                    r"debug",
                    r"error",
                    r"bug",
                    r"fix",
                    r"broken",
                    r"not\s+working",
                    r"doesn't\s+work",
                    r"issue",
                    r"problem",
                    r"exception",
                    r"crash",
                    r"fail",
                    r"troubleshoot"
                ],
                "required_context": ["error_logs", "recent_changes", "code_context"],
                "confidence_threshold": 0.5
            },

            PromptType.GENERAL: {
                "patterns": [
                    r".*"  # Catch-all pattern
                ],
                "required_context": ["basic_context"],
                "confidence_threshold": 0.1
            }
        }

    def analyze_prompt_type(self, prompt: str) -> Tuple[PromptType, float, str]:
        """Analyze prompt and determine its type, confidence, and intent.
        
        Args:
            prompt: The user prompt to analyze
            
        Returns:
            Tuple of (prompt_type, confidence, detected_intent)
        """
        prompt_lower = prompt.lower()
        best_match = (PromptType.GENERAL, 0.1, "general inquiry")
        
        for prompt_type, config in self.prompt_patterns.items():
            confidence = self._calculate_pattern_confidence(prompt_lower, config["patterns"])
            
            if confidence >= config["confidence_threshold"] and confidence > best_match[1]:
                intent = self._extract_intent(prompt, prompt_type)
                best_match = (prompt_type, confidence, intent)
        
        return best_match

    def _calculate_pattern_confidence(self, prompt: str, patterns: List[str]) -> float:
        """Calculate confidence score for a prompt against patterns.
        
        Args:
            prompt: The prompt text (lowercased)
            patterns: List of regex patterns to match
            
        Returns:
            Confidence score between 0 and 1
        """
        if not patterns:
            return 0.0
        
        matches = 0
        total_weight = 0
        
        for pattern in patterns:
            try:
                if re.search(pattern, prompt, re.IGNORECASE):
                    matches += 1
                    # Weight patterns by specificity (longer patterns get higher weight)
                    weight = len(pattern.replace(r'\s+', ' ').replace('[', '').replace(']', ''))
                    total_weight += weight
            except re.error:
                continue  # Skip invalid patterns
        
        if matches == 0:
            return 0.0
        
        # Base confidence from match ratio
        base_confidence = matches / len(patterns)
        
        # Boost confidence for more specific/longer patterns
        specificity_bonus = min(0.3, total_weight / (len(patterns) * 20))
        
        return min(1.0, base_confidence + specificity_bonus)

    def _extract_intent(self, prompt: str, prompt_type: PromptType) -> str:
        """Extract the specific intent from a prompt based on its type.
        
        Args:
            prompt: The original prompt
            prompt_type: The classified prompt type
            
        Returns:
            A description of the detected intent
        """
        intent_extractors = {
            PromptType.REPOSITORY_ANALYSIS: self._extract_repository_intent,
            PromptType.CODE_ANALYSIS: self._extract_code_intent,
            PromptType.FILE_OPERATIONS: self._extract_file_intent,
            PromptType.GENERAL_CODING: self._extract_coding_intent,
            PromptType.TECHNOLOGY_INQUIRY: self._extract_technology_intent,
            PromptType.DEBUGGING: self._extract_debugging_intent,
        }
        
        extractor = intent_extractors.get(prompt_type, self._extract_general_intent)
        return extractor(prompt)

    def _extract_repository_intent(self, prompt: str) -> str:
        """Extract intent for repository analysis prompts."""
        if re.search(r"structure|organization|layout", prompt, re.IGNORECASE):
            return "analyze repository structure and organization"
        elif re.search(r"language|technology|framework", prompt, re.IGNORECASE):
            return "identify technologies and languages used"
        elif re.search(r"overview|summary", prompt, re.IGNORECASE):
            return "provide repository overview and summary"
        else:
            return "general repository analysis"

    def _extract_code_intent(self, prompt: str) -> str:
        """Extract intent for code analysis prompts."""
        if re.search(r"explain|understand", prompt, re.IGNORECASE):
            return "explain and understand code functionality"
        elif re.search(r"review|quality", prompt, re.IGNORECASE):
            return "review code quality and best practices"
        elif re.search(r"refactor|improve|optimize", prompt, re.IGNORECASE):
            return "improve and optimize code"
        else:
            return "general code analysis"

    def _extract_file_intent(self, prompt: str) -> str:
        """Extract intent for file operation prompts."""
        if re.search(r"create|make|new|add", prompt, re.IGNORECASE):
            return "create new files"
        elif re.search(r"modify|edit|update|change", prompt, re.IGNORECASE):
            return "modify existing files"
        elif re.search(r"find|search|list", prompt, re.IGNORECASE):
            return "find and locate files"
        else:
            return "general file operations"

    def _extract_coding_intent(self, prompt: str) -> str:
        """Extract intent for general coding prompts."""
        if re.search(r"function|method", prompt, re.IGNORECASE):
            return "implement functions or methods"
        elif re.search(r"class|object", prompt, re.IGNORECASE):
            return "create classes and objects"
        elif re.search(r"script|automate", prompt, re.IGNORECASE):
            return "create automation scripts"
        else:
            return "general code implementation"

    def _extract_technology_intent(self, prompt: str) -> str:
        """Extract intent for technology inquiry prompts."""
        if re.search(r"what.*framework|which.*framework", prompt, re.IGNORECASE):
            return "identify frameworks used"
        elif re.search(r"dependencies|requirements", prompt, re.IGNORECASE):
            return "analyze project dependencies"
        elif re.search(r"version|configuration", prompt, re.IGNORECASE):
            return "check versions and configuration"
        else:
            return "general technology inquiry"

    def _extract_debugging_intent(self, prompt: str) -> str:
        """Extract intent for debugging prompts."""
        if re.search(r"error|exception", prompt, re.IGNORECASE):
            return "debug errors and exceptions"
        elif re.search(r"not working|doesn't work|broken", prompt, re.IGNORECASE):
            return "fix non-working functionality"
        elif re.search(r"performance|slow", prompt, re.IGNORECASE):
            return "improve performance issues"
        else:
            return "general debugging and troubleshooting"

    def _extract_general_intent(self, prompt: str) -> str:
        """Extract intent for general prompts."""
        return "general inquiry or assistance"

    def get_required_context(self, prompt_type: PromptType, prompt: str) -> List[str]:
        """Get the required context types for a given prompt type.
        
        Args:
            prompt_type: The classified prompt type
            prompt: The original prompt for additional context clues
            
        Returns:
            List of required context types
        """
        base_context = self.prompt_patterns[prompt_type]["required_context"].copy()
        
        # Add additional context based on specific prompt content
        prompt_lower = prompt.lower()
        
        if "file" in prompt_lower or "directory" in prompt_lower:
            if "file_structure" not in base_context:
                base_context.append("file_structure")
        
        if "error" in prompt_lower or "debug" in prompt_lower:
            if "error_logs" not in base_context:
                base_context.append("error_logs")
        
        if "config" in prompt_lower or "setting" in prompt_lower:
            if "config_files" not in base_context:
                base_context.append("config_files")
        
        return base_context
