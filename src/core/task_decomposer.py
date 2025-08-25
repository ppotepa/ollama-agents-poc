"""Task Decomposer - Breaks complex queries into executable subtasks."""
from __future__ import annotations

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from src.utils.enhanced_logging import get_logger
from src.core.model_capability_checker import get_capability_checker


class TaskType(Enum):
    """Types of tasks that can be identified."""
    CODING = "coding"
    RESEARCH = "research" 
    FILE_ANALYSIS = "file_analysis"
    SYSTEM_OPERATION = "system_operation"
    DATA_PROCESSING = "data_processing"
    CREATIVE = "creative"
    GENERAL_QA = "general_qa"


class TaskPriority(Enum):
    """Priority levels for tasks."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Subtask:
    """Represents a decomposed subtask."""
    id: str
    description: str
    task_type: TaskType
    priority: TaskPriority
    estimated_complexity: float  # 0.0 to 1.0
    required_capabilities: List[str]
    dependencies: List[str] = field(default_factory=list)
    preferred_models: List[str] = field(default_factory=list)
    context_needed: List[str] = field(default_factory=list)
    expected_output: str = ""
    
    def __post_init__(self):
        """Set preferred models based on task type."""
        if not self.preferred_models:
            self.preferred_models = self._get_default_preferred_models()
    
    def _get_default_preferred_models(self) -> List[str]:
        """Get default preferred models for this task type using capability checker."""
        capability_checker = get_capability_checker()
        
        # Determine if this task requires tools based on task type and required capabilities
        requires_tools = self._task_requires_tools()
        
        # Map task type to capability name
        task_type_str = self.task_type.value if hasattr(self.task_type, 'value') else str(self.task_type)
        
        # Get the best model for this task
        best_model = capability_checker.get_best_model_for_task(task_type_str, requires_tools)
        
        if best_model:
            # Get additional models for fallback
            all_suitable = []
            
            if requires_tools:
                # Only models that support tools
                tool_models = capability_checker.get_tool_supporting_models()
                capability_models = capability_checker.get_models_with_capability(
                    self._get_required_capability()
                )
                all_suitable = [m for m in capability_models if m in tool_models]
            else:
                # Any model with the right capability
                all_suitable = capability_checker.get_models_with_capability(
                    self._get_required_capability()
                )
            
            # Ensure best model is first, add others as fallbacks
            preferred = [best_model]
            for model in all_suitable:
                if model != best_model and model not in preferred:
                    preferred.append(model)
            
            return preferred[:3]  # Return top 3 models
        
        # Fallback to original hardcoded preferences if capability checker fails
        model_preferences = {
            TaskType.CODING: ["qwen2.5-coder:7b", "deepseek-coder:6.7b", "qwen2.5:7b-instruct-q4_K_M"],
            TaskType.RESEARCH: ["qwen2.5:7b-instruct-q4_K_M", "mistral:7b-instruct"],
            TaskType.FILE_ANALYSIS: ["qwen2.5:7b-instruct-q4_K_M", "gemma:7b-instruct-q4_K_M"],
            TaskType.SYSTEM_OPERATION: ["qwen2.5:7b-instruct-q4_K_M", "gemma:7b-instruct-q4_K_M"],
            TaskType.DATA_PROCESSING: ["qwen2.5:7b-instruct-q4_K_M", "gemma:7b-instruct-q4_K_M"],
            TaskType.CREATIVE: ["gemma:7b-instruct-q4_K_M", "mistral:7b-instruct"],
            TaskType.GENERAL_QA: ["qwen2.5:7b-instruct-q4_K_M", "gemma:7b-instruct-q4_K_M"]
        }
        return model_preferences.get(self.task_type, ["qwen2.5:7b-instruct-q4_K_M"])
    
    def _task_requires_tools(self) -> bool:
        """Determine if this task requires tool support."""
        # Tasks that typically require tools
        tool_requiring_capabilities = [
            "file_operations", "system_operations", "code_execution", 
            "web_operations", "project_operations"
        ]
        
        # Check if any required capabilities need tools
        for capability in self.required_capabilities:
            if any(tool_cap in capability.lower() for tool_cap in tool_requiring_capabilities):
                return True
        
        # Check task type for tool requirements
        if self.task_type in [TaskType.CODING, TaskType.SYSTEM_OPERATION]:
            # These often need file operations or code execution
            return True
        
        return False
    
    def _get_required_capability(self) -> str:
        """Get the primary capability required for this task type."""
        capability_mapping = {
            TaskType.CODING: "coding",
            TaskType.RESEARCH: "general_qa", 
            TaskType.FILE_ANALYSIS: "file_operations",
            TaskType.SYSTEM_OPERATION: "general_qa",
            TaskType.DATA_PROCESSING: "general_qa",
            TaskType.CREATIVE: "general_qa",
            TaskType.GENERAL_QA: "general_qa"
        }
        return capability_mapping.get(self.task_type, "general_qa")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "task_type": self.task_type.value if hasattr(self.task_type, 'value') else str(self.task_type),
            "priority": self.priority.value if hasattr(self.priority, 'value') else str(self.priority),
            "estimated_complexity": self.estimated_complexity,
            "required_capabilities": self.required_capabilities,
            "dependencies": self.dependencies,
            "preferred_models": self.preferred_models,
            "context_needed": self.context_needed,
            "expected_output": self.expected_output
        }


@dataclass
class TaskDecomposition:
    """Result of task decomposition."""
    original_query: str
    subtasks: List[Subtask]
    execution_strategy: str
    estimated_total_complexity: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_query": self.original_query,
            "subtasks": [subtask.to_dict() for subtask in self.subtasks],
            "execution_strategy": self.execution_strategy,
            "estimated_total_complexity": self.estimated_total_complexity,
            "metadata": self.metadata
        }
class TaskDecomposer:
    """Decomposes complex queries into executable subtasks."""
    
    def __init__(self):
        """Initialize the task decomposer."""
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
            r'\b(?:run|execute|command|shell|terminal|bash|powershell)\b',
            r'\b(?:install|setup|configure|environment)\b',
            r'\b(?:system|os|operating system|machine|computer)\b',
            r'\b(?:process|service|daemon|background)\b'
        ]
    
    def decompose(self, query: str, context: Optional[Dict[str, Any]] = None) -> TaskDecomposition:
        """Decompose a complex query into subtasks.
        
        Args:
            query: The complex query to decompose
            context: Optional context information
            
        Returns:
            TaskDecomposition with subtasks and execution strategy
        """
        self.logger.info(f"Decomposing query: {query[:100]}...")
        
        # Analyze the query to identify different task types
        task_indicators = self._analyze_query_complexity(query)
        
        # Generate subtasks based on identified patterns
        subtasks = self._generate_subtasks(query, task_indicators, context)
        
        # Determine execution strategy
        strategy = self._determine_execution_strategy(subtasks)
        
        # Calculate total complexity
        total_complexity = sum(task.estimated_complexity for task in subtasks) / len(subtasks) if subtasks else 0.0
        
        decomposition = TaskDecomposition(
            original_query=query,
            subtasks=subtasks,
            execution_strategy=strategy,
            estimated_total_complexity=total_complexity,
            metadata={
                "task_indicators": task_indicators,
                "decomposition_timestamp": __import__("time").time(),
                "context_provided": context is not None
            }
        )
        
        self.logger.info(f"Decomposed into {len(subtasks)} subtasks with strategy: {strategy}")
        return decomposition
    
    def _analyze_query_complexity(self, query: str) -> Dict[str, float]:
        """Analyze query to identify different task type indicators.
        
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
        
        # Determine if it's creative or general QA
        creative_keywords = ['creative', 'story', 'poem', 'art', 'design', 'brainstorm']
        if any(word in query_lower for word in creative_keywords):
            indicators[TaskType.CREATIVE.value] = 0.8
        
        # If no strong indicators, it's likely general QA
        max_score = max(indicators.values()) if indicators else 0.0
        if max_score < 0.3:
            indicators[TaskType.GENERAL_QA.value] = 0.7
        
        return indicators
    
    def _generate_subtasks(self, query: str, task_indicators: Dict[str, float], context: Optional[Dict[str, Any]]) -> List[Subtask]:
        """Generate subtasks based on task indicators and query analysis."""
        subtasks = []
        
        # Get the dominant task type
        dominant_task = max(task_indicators.items(), key=lambda x: x[1]) if task_indicators else (TaskType.GENERAL_QA.value, 0.7)
        dominant_type = TaskType(dominant_task[0])
        
        # Generate subtasks based on dominant task type
        if dominant_type == TaskType.CODING:
            subtasks.extend(self._generate_coding_subtasks(query))
        elif dominant_type == TaskType.RESEARCH:
            subtasks.extend(self._generate_research_subtasks(query))
        elif dominant_type == TaskType.FILE_ANALYSIS:
            subtasks.extend(self._generate_file_analysis_subtasks(query))
        elif dominant_type == TaskType.SYSTEM_OPERATION:
            subtasks.extend(self._generate_system_subtasks(query))
        else:
            subtasks.extend(self._generate_general_subtasks(query, dominant_type))
        
        # Add cross-cutting concerns
        if task_indicators.get(TaskType.FILE_ANALYSIS.value, 0) > 0.2:
            subtasks.insert(0, self._create_context_gathering_subtask(query))
        
        return subtasks
    
    def _generate_coding_subtasks(self, query: str) -> List[Subtask]:
        """Generate subtasks for coding-related queries."""
        subtasks = []
        
        # Context gathering
        subtasks.append(Subtask(
            id="coding_context",
            description="Analyze project structure and existing code",
            task_type=TaskType.FILE_ANALYSIS,
            priority=TaskPriority.HIGH,
            estimated_complexity=0.3,
            required_capabilities=["file_operations", "code_analysis"],
            context_needed=["repository_structure", "existing_code"],
            expected_output="Project structure and relevant code analysis"
        ))
        
        # Planning
        subtasks.append(Subtask(
            id="coding_planning", 
            description="Plan implementation approach and architecture",
            task_type=TaskType.CODING,
            priority=TaskPriority.HIGH,
            estimated_complexity=0.5,
            required_capabilities=["coding", "architecture_design"],
            dependencies=["coding_context"],
            expected_output="Implementation plan and architecture design"
        ))
        
        # Implementation
        subtasks.append(Subtask(
            id="coding_implementation",
            description="Implement the requested functionality",
            task_type=TaskType.CODING,
            priority=TaskPriority.CRITICAL,
            estimated_complexity=0.8,
            required_capabilities=["coding", "file_operations"],
            dependencies=["coding_planning"],
            expected_output="Complete code implementation"
        ))
        
        # Testing/Validation
        subtasks.append(Subtask(
            id="coding_validation",
            description="Test and validate the implementation",
            task_type=TaskType.CODING,
            priority=TaskPriority.MEDIUM,
            estimated_complexity=0.4,
            required_capabilities=["coding", "testing"],
            dependencies=["coding_implementation"],
            expected_output="Test results and validation report"
        ))
        
        return subtasks
    
    def _generate_research_subtasks(self, query: str) -> List[Subtask]:
        """Generate subtasks for research-related queries."""
        subtasks = []
        
        # Information gathering
        subtasks.append(Subtask(
            id="research_gathering",
            description="Gather relevant information and resources",
            task_type=TaskType.RESEARCH,
            priority=TaskPriority.HIGH,
            estimated_complexity=0.6,
            required_capabilities=["research", "web_search"],
            expected_output="Collected relevant information and sources"
        ))
        
        # Analysis
        subtasks.append(Subtask(
            id="research_analysis", 
            description="Analyze and synthesize gathered information",
            task_type=TaskType.RESEARCH,
            priority=TaskPriority.CRITICAL,
            estimated_complexity=0.7,
            required_capabilities=["analysis", "reasoning"],
            dependencies=["research_gathering"],
            expected_output="Comprehensive analysis and insights"
        ))
        
        return subtasks
    
    def _generate_file_analysis_subtasks(self, query: str) -> List[Subtask]:
        """Generate subtasks for file analysis queries."""
        subtasks = []
        
        # Directory scanning
        subtasks.append(Subtask(
            id="file_scanning",
            description="Scan directory structure and identify relevant files",
            task_type=TaskType.FILE_ANALYSIS,
            priority=TaskPriority.HIGH,
            estimated_complexity=0.3,
            required_capabilities=["file_operations"],
            expected_output="Directory structure and file inventory"
        ))
        
        # File content analysis
        subtasks.append(Subtask(
            id="file_content_analysis",
            description="Analyze content of relevant files",
            task_type=TaskType.FILE_ANALYSIS,
            priority=TaskPriority.CRITICAL,
            estimated_complexity=0.6,
            required_capabilities=["file_operations", "content_analysis"],
            dependencies=["file_scanning"],
            expected_output="Detailed file content analysis"
        ))
        
        return subtasks
    
    def _generate_system_subtasks(self, query: str) -> List[Subtask]:
        """Generate subtasks for system operation queries."""
        return [
            Subtask(
                id="system_operation",
                description="Execute system operations and commands",
                task_type=TaskType.SYSTEM_OPERATION,
                priority=TaskPriority.CRITICAL,
                estimated_complexity=0.5,
                required_capabilities=["system_operations"],
                expected_output="System operation results"
            )
        ]
    
    def _generate_general_subtasks(self, query: str, task_type: TaskType) -> List[Subtask]:
        """Generate subtasks for general queries."""
        return [
            Subtask(
                id="general_response",
                description="Provide comprehensive response to query",
                task_type=task_type,
                priority=TaskPriority.CRITICAL,
                estimated_complexity=0.4,
                required_capabilities=["general_qa"],
                expected_output="Complete response to user query"
            )
        ]
    
    def _create_context_gathering_subtask(self, query: str) -> Subtask:
        """Create a context gathering subtask."""
        return Subtask(
            id="context_gathering",
            description="Gather relevant context and background information",
            task_type=TaskType.FILE_ANALYSIS,
            priority=TaskPriority.HIGH,
            estimated_complexity=0.2,
            required_capabilities=["file_operations", "context_analysis"],
            expected_output="Relevant context and background information"
        )
    
    def _determine_execution_strategy(self, subtasks: List[Subtask]) -> str:
        """Determine the best execution strategy for the subtasks."""
        if not subtasks:
            return "sequential"
        
        # Count dependencies
        total_dependencies = sum(len(task.dependencies) for task in subtasks)
        
        # If many dependencies, use sequential
        if total_dependencies > len(subtasks) * 0.5:
            return "sequential"
        
        # If high complexity tasks, use adaptive
        high_complexity_tasks = sum(1 for task in subtasks if task.estimated_complexity > 0.7)
        if high_complexity_tasks > 0:
            return "adaptive"
        
        # Default to parallel for independent tasks
        return "parallel"


def create_task_decomposer() -> TaskDecomposer:
    """Create a TaskDecomposer instance."""
    return TaskDecomposer()


__all__ = ["TaskDecomposer", "Subtask", "TaskDecomposition", "TaskType", "TaskPriority", "create_task_decomposer"]
