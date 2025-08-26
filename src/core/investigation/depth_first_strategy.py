"""Depth-first investigation strategy implementation."""

from typing import Any, Dict, List

from .base_strategy import BaseInvestigationStrategy
from .types import InvestigationPlan, InvestigationPriority, InvestigationStep, InvestigationStrategy


class DepthFirstStrategy(BaseInvestigationStrategy):
    """Depth-first investigation strategy that dives deep into specific areas."""

    def __init__(self):
        """Initialize the depth-first strategy."""
        super().__init__(InvestigationStrategy.DEPTH_FIRST)

    def create_investigation_plan(self, query: str, context: Dict[str, Any]) -> InvestigationPlan:
        """Create a depth-first investigation plan."""
        self.usage_stats["plans_created"] += 1
        
        # Analyze the query to determine focus area
        focus_area = self._identify_focus_area(query)
        complexity = self._analyze_query_complexity(query)
        intent = self._categorize_query_intent(query)
        
        # Generate steps based on focus area
        if focus_area == "code":
            steps = self._create_code_investigation_steps(query, complexity, intent)
        elif focus_area == "files":
            steps = self._create_file_investigation_steps(query, complexity, intent)
        elif focus_area == "implementation":
            steps = self._create_implementation_investigation_steps(query, complexity, intent)
        elif focus_area == "structure":
            steps = self._create_structure_investigation_steps(query, complexity, intent)
        else:
            steps = self._create_general_investigation_steps(query, complexity, intent)
        
        # Update stats
        self.usage_stats["total_steps_generated"] += len(steps)
        
        # Calculate total duration
        total_duration = sum(step.estimated_duration for step in steps)
        
        # Create success criteria
        success_criteria = self._generate_success_criteria(query, focus_area)
        
        # Generate investigation ID
        investigation_id = self._generate_step_id("depth_first_plan", 0)
        
        return InvestigationPlan(
            investigation_id=investigation_id,
            query=query,
            strategy=self.strategy_type,
            steps=steps,
            total_estimated_duration=total_duration,
            success_criteria=success_criteria,
            fallback_strategies=[InvestigationStrategy.BREADTH_FIRST, InvestigationStrategy.TARGETED]
        )

    def _identify_focus_area(self, query: str) -> str:
        """Identify the primary focus area for deep investigation."""
        query_lower = query.lower()
        
        # Code-related patterns
        if any(word in query_lower for word in ["function", "method", "class", "algorithm", "code"]):
            return "code"
        
        # File-related patterns
        elif any(word in query_lower for word in ["file", "directory", "folder", "path"]):
            return "files"
        
        # Implementation patterns
        elif any(word in query_lower for word in ["implement", "build", "create", "develop"]):
            return "implementation"
        
        # Structure patterns
        elif any(word in query_lower for word in ["structure", "architecture", "design", "organization"]):
            return "structure"
        
        else:
            return "general"

    def _create_code_investigation_steps(self, query: str, complexity: str, intent: str) -> List[InvestigationStep]:
        """Create investigation steps focused on code analysis."""
        steps = []
        models = self._prioritize_models_for_intent(intent)
        
        # Step 1: Initial code discovery
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("code_discovery", 1),
            description="Discover and catalog code files in the repository",
            strategy=self.strategy_type,
            priority=InvestigationPriority.HIGH,
            estimated_duration=self._estimate_duration("file_analysis", complexity),
            required_models=models[:2],
            dependencies=[],
            expected_outputs=["file_list", "code_structure_overview"],
            validation_criteria=["Found relevant code files", "Identified main programming languages"]
        ))
        
        # Step 2: Analyze main code structures
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("structure_analysis", 2),
            description="Analyze main code structures, classes, and functions",
            strategy=self.strategy_type,
            priority=InvestigationPriority.CRITICAL,
            estimated_duration=self._estimate_duration("code_review", complexity),
            required_models=[models[0]],  # Best model for code analysis
            dependencies=[steps[0].step_id],
            expected_outputs=["class_hierarchy", "function_signatures", "code_relationships"],
            validation_criteria=["Identified key classes/functions", "Mapped code relationships"]
        ))
        
        # Step 3: Deep dive into specific components
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("component_deep_dive", 3),
            description="Deep analysis of specific code components relevant to the query",
            strategy=self.strategy_type,
            priority=InvestigationPriority.HIGH,
            estimated_duration=self._estimate_duration("implementation_review", complexity),
            required_models=[models[0]],
            dependencies=[steps[1].step_id],
            expected_outputs=["detailed_component_analysis", "implementation_details"],
            validation_criteria=["Thoroughly analyzed relevant components", "Identified implementation patterns"]
        ))
        
        # Step 4: Analyze dependencies and imports
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("dependency_analysis", 4),
            description="Analyze code dependencies and import relationships",
            strategy=self.strategy_type,
            priority=InvestigationPriority.MEDIUM,
            estimated_duration=self._estimate_duration("dependency_check", complexity),
            required_models=models[:2],
            dependencies=[steps[1].step_id],
            expected_outputs=["dependency_map", "import_analysis"],
            validation_criteria=["Mapped external dependencies", "Identified internal dependencies"]
        ))
        
        return steps

    def _create_file_investigation_steps(self, query: str, complexity: str, intent: str) -> List[InvestigationStep]:
        """Create investigation steps focused on file exploration."""
        steps = []
        models = self._prioritize_models_for_intent(intent)
        
        # Step 1: File system overview
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("filesystem_overview", 1),
            description="Get comprehensive overview of file system structure",
            strategy=self.strategy_type,
            priority=InvestigationPriority.HIGH,
            estimated_duration=self._estimate_duration("structure_analysis", complexity),
            required_models=models[:1],
            dependencies=[],
            expected_outputs=["directory_structure", "file_counts", "file_types"],
            validation_criteria=["Mapped directory structure", "Identified file patterns"]
        ))
        
        # Step 2: Analyze file content patterns
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("content_pattern_analysis", 2),
            description="Analyze patterns in file content and organization",
            strategy=self.strategy_type,
            priority=InvestigationPriority.CRITICAL,
            estimated_duration=self._estimate_duration("file_analysis", complexity),
            required_models=models[:2],
            dependencies=[steps[0].step_id],
            expected_outputs=["content_patterns", "organization_insights"],
            validation_criteria=["Identified content patterns", "Found organizational logic"]
        ))
        
        # Step 3: Deep dive into specific files
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("specific_file_analysis", 3),
            description="Deep analysis of specific files relevant to the query",
            strategy=self.strategy_type,
            priority=InvestigationPriority.HIGH,
            estimated_duration=self._estimate_duration("documentation_review", complexity),
            required_models=[models[0]],
            dependencies=[steps[1].step_id],
            expected_outputs=["detailed_file_analysis", "relevant_content"],
            validation_criteria=["Thoroughly analyzed target files", "Extracted relevant information"]
        ))
        
        return steps

    def _create_implementation_investigation_steps(self, query: str, complexity: str, intent: str) -> List[InvestigationStep]:
        """Create investigation steps focused on implementation details."""
        steps = []
        models = self._prioritize_models_for_intent(intent)
        
        # Step 1: Implementation overview
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("implementation_overview", 1),
            description="Get overview of current implementation approach",
            strategy=self.strategy_type,
            priority=InvestigationPriority.HIGH,
            estimated_duration=self._estimate_duration("structure_analysis", complexity),
            required_models=models[:2],
            dependencies=[],
            expected_outputs=["implementation_summary", "architectural_overview"],
            validation_criteria=["Understood current approach", "Identified key components"]
        ))
        
        # Step 2: Design pattern analysis
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("design_pattern_analysis", 2),
            description="Analyze design patterns and architectural decisions",
            strategy=self.strategy_type,
            priority=InvestigationPriority.CRITICAL,
            estimated_duration=self._estimate_duration("implementation_review", complexity),
            required_models=[models[0]],
            dependencies=[steps[0].step_id],
            expected_outputs=["design_patterns", "architectural_decisions"],
            validation_criteria=["Identified design patterns", "Understood architectural choices"]
        ))
        
        # Step 3: Implementation quality assessment
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("quality_assessment", 3),
            description="Assess implementation quality and identify improvement areas",
            strategy=self.strategy_type,
            priority=InvestigationPriority.MEDIUM,
            estimated_duration=self._estimate_duration("code_review", complexity),
            required_models=models[:2],
            dependencies=[steps[1].step_id],
            expected_outputs=["quality_assessment", "improvement_suggestions"],
            validation_criteria=["Assessed code quality", "Identified improvement opportunities"]
        ))
        
        return steps

    def _create_structure_investigation_steps(self, query: str, complexity: str, intent: str) -> List[InvestigationStep]:
        """Create investigation steps focused on structural analysis."""
        steps = []
        models = self._prioritize_models_for_intent(intent)
        
        # Step 1: High-level structure mapping
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("structure_mapping", 1),
            description="Map high-level project structure and organization",
            strategy=self.strategy_type,
            priority=InvestigationPriority.CRITICAL,
            estimated_duration=self._estimate_duration("structure_analysis", complexity),
            required_models=models[:1],
            dependencies=[],
            expected_outputs=["structure_map", "component_relationships"],
            validation_criteria=["Mapped project structure", "Identified main components"]
        ))
        
        # Step 2: Component interaction analysis
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("interaction_analysis", 2),
            description="Analyze how components interact and dependencies flow",
            strategy=self.strategy_type,
            priority=InvestigationPriority.HIGH,
            estimated_duration=self._estimate_duration("dependency_check", complexity),
            required_models=models[:2],
            dependencies=[steps[0].step_id],
            expected_outputs=["interaction_diagram", "dependency_flow"],
            validation_criteria=["Mapped component interactions", "Understood data flow"]
        ))
        
        # Step 3: Architectural deep dive
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("architecture_deep_dive", 3),
            description="Deep dive into architectural decisions and their implications",
            strategy=self.strategy_type,
            priority=InvestigationPriority.HIGH,
            estimated_duration=self._estimate_duration("implementation_review", complexity),
            required_models=[models[0]],
            dependencies=[steps[1].step_id],
            expected_outputs=["architectural_analysis", "design_rationale"],
            validation_criteria=["Understood architectural decisions", "Identified design trade-offs"]
        ))
        
        return steps

    def _create_general_investigation_steps(self, query: str, complexity: str, intent: str) -> List[InvestigationStep]:
        """Create general investigation steps when focus area is unclear."""
        steps = []
        models = self._prioritize_models_for_intent(intent)
        
        # Step 1: Initial exploration
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("initial_exploration", 1),
            description="Initial exploration to understand the query context",
            strategy=self.strategy_type,
            priority=InvestigationPriority.HIGH,
            estimated_duration=self._estimate_duration("structure_analysis", complexity),
            required_models=models[:1],
            dependencies=[],
            expected_outputs=["initial_findings", "context_understanding"],
            validation_criteria=["Understood query context", "Identified investigation direction"]
        ))
        
        # Step 2: Focused investigation
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("focused_investigation", 2),
            description="Focused investigation based on initial findings",
            strategy=self.strategy_type,
            priority=InvestigationPriority.CRITICAL,
            estimated_duration=self._estimate_duration("implementation_review", complexity),
            required_models=models[:2],
            dependencies=[steps[0].step_id],
            expected_outputs=["detailed_analysis", "specific_findings"],
            validation_criteria=["Conducted focused analysis", "Found relevant information"]
        ))
        
        return steps

    def _generate_success_criteria(self, query: str, focus_area: str) -> List[str]:
        """Generate success criteria for the investigation."""
        base_criteria = [
            "Successfully addressed the main query",
            "Provided comprehensive and accurate information",
            "Covered all relevant aspects of the question"
        ]
        
        focus_specific_criteria = {
            "code": [
                "Analyzed relevant code components",
                "Identified implementation patterns",
                "Explained code functionality"
            ],
            "files": [
                "Explored relevant files and directories",
                "Understood file organization",
                "Extracted important file content"
            ],
            "implementation": [
                "Analyzed implementation approach",
                "Identified design decisions",
                "Assessed implementation quality"
            ],
            "structure": [
                "Mapped project structure",
                "Understood component relationships",
                "Analyzed architectural decisions"
            ]
        }
        
        specific_criteria = focus_specific_criteria.get(focus_area, [])
        return base_criteria + specific_criteria
