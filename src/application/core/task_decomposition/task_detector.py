"""Task type detection and classification for query analysis."""

import re
from typing import Dict

from src.core.task_decomposition.types import TaskType
from src.utils.enhanced_logging import get_logger


class TaskTypeDetector:
    """Detects task types from user queries using pattern matching."""

    def __init__(self):
        """Initialize the task type detector."""
        self.logger = get_logger()
        self._init_patterns()

    def _init_patterns(self):
        """Initialize regex patterns for task identification."""
        self.coding_patterns = [
            r'\b(?:implement|code|write|create|build|develop)\b.*(?:function|class|script|program|application|api|library)',
            r'\b(?:fix|debug|refactor|optimize)\b.*(?:code|bug|error|performance)',
            r'\b(?:python|javascript|java|c\+\+|rust|go|typescript)\b',
            r'\b(?:algorithm|data structure|design pattern)\b',
            r'\b(?:async|await|threading|concurrent|parallel)\b',
            r'\b(?:docker|kubernetes|microservice|api|rest|graphql)\b'
        ]

        self.research_patterns = [
            r'\b(?:research|investigate|analyze|study|explore|examine)\b',
            r'\b(?:what is|how does|why|when|where)\b',
            r'\b(?:compare|contrast|difference|similar|relationship)\b',
            r'\b(?:best practice|recommendation|approach|strategy)\b',
            r'\b(?:documentation|reference|guide|tutorial)\b'
        ]

        self.file_analysis_patterns = [
            r'\b(?:read|parse|analyze|scan|examine)\b.*(?:file|directory|folder|code|document)',
            r'\b(?:find|search|locate|discover)\b.*(?:in|within|across)\b.*(?:files|codebase|project)',
            r'\b(?:structure|organization|architecture)\b.*(?:project|codebase|repository)',
            r'\b(?:list|show|display)\b.*(?:files|directories|contents)'
        ]

        self.system_patterns = [
            r'\b(?:run|execute|launch|start|stop|restart)\b.*(?:command|process|service|application)',
            r'\b(?:install|configure|setup|deploy)\b',
            r'\b(?:monitor|check|status|health)\b.*(?:system|service|process)',
            r'\b(?:permission|access|security|authentication)\b',
            r'\b(?:environment|variable|config|setting)\b'
        ]

        self.data_processing_patterns = [
            r'\b(?:process|transform|convert|parse|extract)\b.*(?:data|file|json|csv|xml)',
            r'\b(?:filter|sort|group|aggregate|summarize)\b',
            r'\b(?:database|sql|query|table|record)\b',
            r'\b(?:import|export|migrate|backup)\b.*(?:data)',
            r'\b(?:statistical|analysis|metrics|report)\b'
        ]

    def detect_task_types(self, query: str) -> Dict[str, float]:
        """Detect task types from a query.

        Args:
            query: The user query to analyze

        Returns:
            Dictionary mapping task types to confidence scores (0.0-1.0)
        """
        query_lower = query.lower()
        indicators = {}

        # Check coding patterns
        coding_score = 0.0
        for pattern in self.coding_patterns:
            if re.search(pattern, query_lower):
                coding_score += 0.2
        indicators[TaskType.CODING.value] = min(coding_score, 1.0)

        # Check research patterns
        research_score = 0.0
        for pattern in self.research_patterns:
            if re.search(pattern, query_lower):
                research_score += 0.3
        indicators[TaskType.RESEARCH.value] = min(research_score, 1.0)

        # Check file analysis patterns
        file_score = 0.0
        for pattern in self.file_analysis_patterns:
            if re.search(pattern, query_lower):
                file_score += 0.4
        indicators[TaskType.FILE_ANALYSIS.value] = min(file_score, 1.0)

        # Check system operation patterns
        system_score = 0.0
        for pattern in self.system_patterns:
            if re.search(pattern, query_lower):
                system_score += 0.3
        indicators[TaskType.SYSTEM_OPERATION.value] = min(system_score, 1.0)

        # Check data processing patterns
        data_score = 0.0
        for pattern in self.data_processing_patterns:
            if re.search(pattern, query_lower):
                data_score += 0.3
        indicators[TaskType.DATA_PROCESSING.value] = min(data_score, 1.0)

        # Determine if it's creative or general QA
        creative_keywords = ['creative', 'story', 'poem', 'art', 'design', 'brainstorm']
        if any(word in query_lower for word in creative_keywords):
            indicators[TaskType.CREATIVE.value] = 0.8

        # If no strong indicators, it's likely general QA
        max_score = max(indicators.values()) if indicators else 0.0
        if max_score < 0.3:
            indicators[TaskType.GENERAL_QA.value] = 0.7

        return indicators

    def get_primary_task_type(self, query: str) -> TaskType:
        """Get the primary task type for a query.

        Args:
            query: The user query to analyze

        Returns:
            The most likely task type
        """
        indicators = self.detect_task_types(query)
        
        if not indicators:
            return TaskType.GENERAL_QA
        
        # Find the task type with the highest confidence
        primary_type = max(indicators, key=indicators.get)
        
        # Convert string back to enum
        for task_type in TaskType:
            if task_type.value == primary_type:
                return task_type
        
        return TaskType.GENERAL_QA

    def is_complex_query(self, query: str) -> bool:
        """Determine if a query is complex and needs decomposition.

        Args:
            query: The user query to analyze

        Returns:
            True if query is complex, False otherwise
        """
        complexity_indicators = [
            r'\b(?:and|also|then|after|before|while|during)\b',  # Sequential operations
            r'\b(?:multiple|several|various|different)\b',  # Multiple items
            r'\b(?:step|phase|stage|part)\b',  # Multi-step process
            len(query.split()) > 20,  # Long queries
            query.count('?') > 1,  # Multiple questions
            query.count(',') > 2,  # Multiple clauses
        ]

        complexity_score = 0
        query_lower = query.lower()

        for indicator in complexity_indicators[:-3]:  # Skip the boolean indicators
            if isinstance(indicator, str) and re.search(indicator, query_lower):
                complexity_score += 1

        # Add boolean indicators
        if len(query.split()) > 20:
            complexity_score += 1
        if query.count('?') > 1:
            complexity_score += 1
        if query.count(',') > 2:
            complexity_score += 1

        return complexity_score >= 2

    def extract_entities(self, query: str) -> Dict[str, list[str]]:
        """Extract key entities from the query.

        Args:
            query: The user query to analyze

        Returns:
            Dictionary of entity types and their values
        """
        entities = {
            "files": [],
            "languages": [],
            "frameworks": [],
            "tools": [],
            "actions": []
        }

        # File patterns
        file_patterns = [
            r'\b[\w\-\.]+\.(?:py|js|ts|java|cpp|c|h|rs|go|php|rb|swift|kt|scala)\b',
            r'\b(?:README|Dockerfile|package\.json|requirements\.txt|Cargo\.toml)\b'
        ]

        # Language patterns
        language_patterns = [
            r'\b(?:python|javascript|typescript|java|cpp|c\+\+|rust|go|php|ruby|swift|kotlin|scala)\b'
        ]

        # Framework patterns
        framework_patterns = [
            r'\b(?:react|vue|angular|django|flask|fastapi|express|spring|rails)\b'
        ]

        # Tool patterns
        tool_patterns = [
            r'\b(?:git|docker|kubernetes|npm|pip|cargo|maven|gradle)\b'
        ]

        # Action patterns
        action_patterns = [
            r'\b(?:create|build|implement|fix|debug|analyze|test|deploy|install|configure)\b'
        ]

        query_lower = query.lower()

        # Extract files
        for pattern in file_patterns:
            entities["files"].extend(re.findall(pattern, query, re.IGNORECASE))

        # Extract languages
        for pattern in language_patterns:
            entities["languages"].extend(re.findall(pattern, query_lower))

        # Extract frameworks
        for pattern in framework_patterns:
            entities["frameworks"].extend(re.findall(pattern, query_lower))

        # Extract tools
        for pattern in tool_patterns:
            entities["tools"].extend(re.findall(pattern, query_lower))

        # Extract actions
        for pattern in action_patterns:
            entities["actions"].extend(re.findall(pattern, query_lower))

        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))

        return entities
