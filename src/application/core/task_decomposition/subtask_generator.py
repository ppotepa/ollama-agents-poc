"""Subtask generation and decomposition logic."""

import uuid
from typing import Any, Dict, List

from src.core.task_decomposition.types import TaskType, TaskPriority, Subtask, TaskDecomposition
from src.core.task_decomposition.task_detector import TaskTypeDetector
from src.utils.enhanced_logging import get_logger


class SubtaskGenerator:
    """Generates subtasks from complex queries."""

    def __init__(self):
        """Initialize the subtask generator."""
        self.logger = get_logger()
        self.task_detector = TaskTypeDetector()

    def generate_subtasks(self, query: str, task_indicators: Dict[str, float], 
                         context: Dict[str, Any] = None) -> List[Subtask]:
        """Generate subtasks based on task indicators and query analysis.

        Args:
            query: The original user query
            task_indicators: Task type confidence scores
            context: Optional context information

        Returns:
            List of generated subtasks
        """
        if context is None:
            context = {}

        subtasks = []
        entities = self.task_detector.extract_entities(query)

        # Get the primary task type
        primary_task_type = max(task_indicators, key=task_indicators.get)
        primary_type_enum = self._string_to_task_type(primary_task_type)

        # Generate subtasks based on primary task type
        if primary_type_enum == TaskType.CODING:
            subtasks.extend(self._generate_coding_subtasks(query, entities, context))
        elif primary_type_enum == TaskType.RESEARCH:
            subtasks.extend(self._generate_research_subtasks(query, entities, context))
        elif primary_type_enum == TaskType.FILE_ANALYSIS:
            subtasks.extend(self._generate_file_analysis_subtasks(query, entities, context))
        elif primary_type_enum == TaskType.SYSTEM_OPERATION:
            subtasks.extend(self._generate_system_operation_subtasks(query, entities, context))
        elif primary_type_enum == TaskType.DATA_PROCESSING:
            subtasks.extend(self._generate_data_processing_subtasks(query, entities, context))
        else:
            # For CREATIVE and GENERAL_QA, create a single subtask
            subtasks.append(self._create_single_subtask(query, primary_type_enum))

        # If no subtasks generated, create a fallback
        if not subtasks:
            subtasks.append(self._create_fallback_subtask(query))

        return subtasks

    def _generate_coding_subtasks(self, query: str, entities: Dict[str, List[str]], 
                                 context: Dict[str, Any]) -> List[Subtask]:
        """Generate subtasks for coding tasks."""
        subtasks = []

        # Analysis phase
        if entities.get("files") or "analyze" in query.lower():
            subtasks.append(Subtask(
                id=self._generate_task_id(),
                description=f"Analyze existing code structure and identify requirements",
                task_type=TaskType.FILE_ANALYSIS,
                priority=TaskPriority.HIGH,
                estimated_complexity=0.3,
                required_capabilities=["file_operations", "code_analysis"],
                context_needed=["project_structure", "existing_code"]
            ))

        # Implementation phase
        subtasks.append(Subtask(
            id=self._generate_task_id(),
            description=f"Implement the requested code solution",
            task_type=TaskType.CODING,
            priority=TaskPriority.CRITICAL,
            estimated_complexity=0.7,
            required_capabilities=["coding", "problem_solving"],
            dependencies=[subtasks[0].id] if subtasks else [],
            context_needed=["requirements", "code_standards"]
        ))

        # Testing phase
        if "test" in query.lower() or context.get("require_tests"):
            subtasks.append(Subtask(
                id=self._generate_task_id(),
                description=f"Create tests for the implemented solution",
                task_type=TaskType.CODING,
                priority=TaskPriority.MEDIUM,
                estimated_complexity=0.4,
                required_capabilities=["coding", "testing"],
                dependencies=[subtasks[-1].id],
                context_needed=["implementation_details"]
            ))

        return subtasks

    def _generate_research_subtasks(self, query: str, entities: Dict[str, List[str]], 
                                   context: Dict[str, Any]) -> List[Subtask]:
        """Generate subtasks for research tasks."""
        subtasks = []

        # Information gathering
        subtasks.append(Subtask(
            id=self._generate_task_id(),
            description=f"Gather relevant information and documentation",
            task_type=TaskType.RESEARCH,
            priority=TaskPriority.HIGH,
            estimated_complexity=0.5,
            required_capabilities=["research", "information_retrieval"],
            context_needed=["search_terms", "information_sources"]
        ))

        # Analysis and synthesis
        subtasks.append(Subtask(
            id=self._generate_task_id(),
            description=f"Analyze gathered information and synthesize findings",
            task_type=TaskType.RESEARCH,
            priority=TaskPriority.CRITICAL,
            estimated_complexity=0.6,
            required_capabilities=["analysis", "synthesis"],
            dependencies=[subtasks[0].id],
            context_needed=["gathered_information"]
        ))

        # Recommendation generation
        if "recommend" in query.lower() or "best" in query.lower():
            subtasks.append(Subtask(
                id=self._generate_task_id(),
                description=f"Generate recommendations based on research",
                task_type=TaskType.RESEARCH,
                priority=TaskPriority.MEDIUM,
                estimated_complexity=0.4,
                required_capabilities=["reasoning", "recommendation"],
                dependencies=[subtasks[-1].id],
                context_needed=["analysis_results", "criteria"]
            ))

        return subtasks

    def _generate_file_analysis_subtasks(self, query: str, entities: Dict[str, List[str]], 
                                        context: Dict[str, Any]) -> List[Subtask]:
        """Generate subtasks for file analysis tasks."""
        subtasks = []

        # File discovery
        subtasks.append(Subtask(
            id=self._generate_task_id(),
            description=f"Discover and catalog relevant files",
            task_type=TaskType.FILE_ANALYSIS,
            priority=TaskPriority.HIGH,
            estimated_complexity=0.3,
            required_capabilities=["file_operations", "search"],
            context_needed=["search_criteria", "file_patterns"]
        ))

        # Content analysis
        subtasks.append(Subtask(
            id=self._generate_task_id(),
            description=f"Analyze file contents and extract insights",
            task_type=TaskType.FILE_ANALYSIS,
            priority=TaskPriority.CRITICAL,
            estimated_complexity=0.7,
            required_capabilities=["file_operations", "content_analysis"],
            dependencies=[subtasks[0].id],
            context_needed=["file_list", "analysis_goals"]
        ))

        # Summary generation
        subtasks.append(Subtask(
            id=self._generate_task_id(),
            description=f"Generate summary and findings report",
            task_type=TaskType.FILE_ANALYSIS,
            priority=TaskPriority.MEDIUM,
            estimated_complexity=0.4,
            required_capabilities=["synthesis", "reporting"],
            dependencies=[subtasks[-1].id],
            context_needed=["analysis_results"]
        ))

        return subtasks

    def _generate_system_operation_subtasks(self, query: str, entities: Dict[str, List[str]], 
                                           context: Dict[str, Any]) -> List[Subtask]:
        """Generate subtasks for system operation tasks."""
        subtasks = []

        # Preparation phase
        subtasks.append(Subtask(
            id=self._generate_task_id(),
            description=f"Prepare system environment and check prerequisites",
            task_type=TaskType.SYSTEM_OPERATION,
            priority=TaskPriority.HIGH,
            estimated_complexity=0.3,
            required_capabilities=["system_operations", "environment_check"],
            context_needed=["system_state", "requirements"]
        ))

        # Execution phase
        subtasks.append(Subtask(
            id=self._generate_task_id(),
            description=f"Execute the requested system operation",
            task_type=TaskType.SYSTEM_OPERATION,
            priority=TaskPriority.CRITICAL,
            estimated_complexity=0.6,
            required_capabilities=["system_operations", "command_execution"],
            dependencies=[subtasks[0].id],
            context_needed=["operation_details", "parameters"]
        ))

        # Verification phase
        subtasks.append(Subtask(
            id=self._generate_task_id(),
            description=f"Verify operation success and check system state",
            task_type=TaskType.SYSTEM_OPERATION,
            priority=TaskPriority.MEDIUM,
            estimated_complexity=0.3,
            required_capabilities=["system_operations", "verification"],
            dependencies=[subtasks[-1].id],
            context_needed=["expected_results"]
        ))

        return subtasks

    def _generate_data_processing_subtasks(self, query: str, entities: Dict[str, List[str]], 
                                          context: Dict[str, Any]) -> List[Subtask]:
        """Generate subtasks for data processing tasks."""
        subtasks = []

        # Data validation
        subtasks.append(Subtask(
            id=self._generate_task_id(),
            description=f"Validate and prepare input data",
            task_type=TaskType.DATA_PROCESSING,
            priority=TaskPriority.HIGH,
            estimated_complexity=0.3,
            required_capabilities=["data_validation", "preprocessing"],
            context_needed=["data_sources", "validation_rules"]
        ))

        # Processing
        subtasks.append(Subtask(
            id=self._generate_task_id(),
            description=f"Process data according to requirements",
            task_type=TaskType.DATA_PROCESSING,
            priority=TaskPriority.CRITICAL,
            estimated_complexity=0.7,
            required_capabilities=["data_processing", "transformation"],
            dependencies=[subtasks[0].id],
            context_needed=["processing_rules", "output_format"]
        ))

        # Output generation
        subtasks.append(Subtask(
            id=self._generate_task_id(),
            description=f"Generate and format output results",
            task_type=TaskType.DATA_PROCESSING,
            priority=TaskPriority.MEDIUM,
            estimated_complexity=0.4,
            required_capabilities=["output_formatting", "result_generation"],
            dependencies=[subtasks[-1].id],
            context_needed=["output_requirements"]
        ))

        return subtasks

    def _create_single_subtask(self, query: str, task_type: TaskType) -> Subtask:
        """Create a single subtask for simple queries."""
        return Subtask(
            id=self._generate_task_id(),
            description=query,
            task_type=task_type,
            priority=TaskPriority.MEDIUM,
            estimated_complexity=0.5,
            required_capabilities=[task_type.value],
            context_needed=["user_query"]
        )

    def _create_fallback_subtask(self, query: str) -> Subtask:
        """Create a fallback subtask when decomposition fails."""
        return Subtask(
            id=self._generate_task_id(),
            description=f"Process query: {query}",
            task_type=TaskType.GENERAL_QA,
            priority=TaskPriority.MEDIUM,
            estimated_complexity=0.5,
            required_capabilities=["general_qa"],
            context_needed=["user_query"]
        )

    def _string_to_task_type(self, task_type_str: str) -> TaskType:
        """Convert string to TaskType enum."""
        for task_type in TaskType:
            if task_type.value == task_type_str:
                return task_type
        return TaskType.GENERAL_QA

    def _generate_task_id(self) -> str:
        """Generate a unique task ID."""
        return f"task_{uuid.uuid4().hex[:8]}"
