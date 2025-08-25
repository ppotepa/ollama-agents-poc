"""Investigation Strategies - Different exploration patterns for various query types."""
from __future__ import annotations

import re
import time
from typing import Dict, List, Any, Optional, Tuple, Protocol
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from src.utils.enhanced_logging import get_logger
from src.core.task_decomposer import TaskType, TaskDecomposer, Subtask
from src.core.execution_planner import ExecutionPlanner, ExecutionPlan
from src.core.context_manager import ContextManager, get_context_manager


class InvestigationStrategy(Enum):
    """Different investigation strategies."""
    DEPTH_FIRST = "depth_first"
    BREADTH_FIRST = "breadth_first"
    TARGETED = "targeted"
    EXPLORATORY = "exploratory"
    HYPOTHESIS_DRIVEN = "hypothesis_driven"
    INCREMENTAL = "incremental"


class InvestigationPriority(Enum):
    """Priority levels for investigation steps."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    OPTIONAL = "optional"


@dataclass
class InvestigationStep:
    """Individual step in an investigation."""
    step_id: str
    description: str
    strategy: InvestigationStrategy
    priority: InvestigationPriority
    estimated_duration: float
    required_models: List[str]
    dependencies: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    validation_criteria: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "description": self.description,
            "strategy": self.strategy.value,
            "priority": self.priority.value,
            "estimated_duration": self.estimated_duration,
            "required_models": self.required_models,
            "dependencies": self.dependencies,
            "expected_outputs": self.expected_outputs,
            "validation_criteria": self.validation_criteria,
            "metadata": self.metadata
        }


@dataclass
class InvestigationPlan:
    """Complete investigation plan with ordered steps."""
    investigation_id: str
    query: str
    strategy: InvestigationStrategy
    steps: List[InvestigationStep]
    total_estimated_duration: float
    success_criteria: List[str]
    fallback_strategies: List[InvestigationStrategy] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    
    def get_next_executable_steps(self, completed_steps: List[str]) -> List[InvestigationStep]:
        """Get steps that can be executed next based on completed dependencies."""
        executable = []
        
        for step in self.steps:
            if step.step_id in completed_steps:
                continue
            
            # Check if all dependencies are completed
            if all(dep in completed_steps for dep in step.dependencies):
                executable.append(step)
        
        # Sort by priority
        priority_order = {
            InvestigationPriority.CRITICAL: 0,
            InvestigationPriority.HIGH: 1,
            InvestigationPriority.MEDIUM: 2,
            InvestigationPriority.LOW: 3,
            InvestigationPriority.OPTIONAL: 4
        }
        
        executable.sort(key=lambda x: priority_order[x.priority])
        return executable
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "investigation_id": self.investigation_id,
            "query": self.query,
            "strategy": self.strategy.value,
            "steps": [step.to_dict() for step in self.steps],
            "total_estimated_duration": self.total_estimated_duration,
            "success_criteria": self.success_criteria,
            "fallback_strategies": [s.value for s in self.fallback_strategies],
            "created_at": self.created_at
        }


class BaseInvestigationStrategy(ABC):
    """Base class for investigation strategies."""
    
    def __init__(self, strategy_type: InvestigationStrategy):
        """Initialize base strategy.
        
        Args:
            strategy_type: Type of investigation strategy
        """
        self.strategy_type = strategy_type
        self.logger = get_logger()
    
    @abstractmethod
    def create_investigation_plan(self, query: str, context: Dict[str, Any]) -> InvestigationPlan:
        """Create an investigation plan for the given query.
        
        Args:
            query: User query to investigate
            context: Additional context information
            
        Returns:
            InvestigationPlan with ordered steps
        """
        pass
    
    @abstractmethod
    def adapt_plan(self, plan: InvestigationPlan, 
                   execution_results: List[Dict[str, Any]]) -> InvestigationPlan:
        """Adapt the investigation plan based on execution results.
        
        Args:
            plan: Original investigation plan
            execution_results: Results from executed steps
            
        Returns:
            Adapted investigation plan
        """
        pass
    
    def _generate_step_id(self, base_name: str, index: int) -> str:
        """Generate a unique step ID."""
        return f"{self.strategy_type.value}_{base_name}_{index}_{int(time.time())}"
    
    def _estimate_duration(self, step_type: str, complexity: str = "medium") -> float:
        """Estimate duration for a step type."""
        base_durations = {
            "file_scan": 30,
            "code_analysis": 120,
            "implementation": 300,
            "testing": 180,
            "research": 240,
            "documentation": 150,
            "validation": 60
        }
        
        complexity_multipliers = {
            "simple": 0.5,
            "medium": 1.0,
            "complex": 2.0,
            "very_complex": 3.0
        }
        
        base_time = base_durations.get(step_type, 120)
        multiplier = complexity_multipliers.get(complexity, 1.0)
        
        return base_time * multiplier


class DepthFirstStrategy(BaseInvestigationStrategy):
    """Depth-first investigation strategy - dive deep into specific areas."""
    
    def __init__(self):
        super().__init__(InvestigationStrategy.DEPTH_FIRST)
    
    def create_investigation_plan(self, query: str, context: Dict[str, Any]) -> InvestigationPlan:
        """Create depth-first investigation plan."""
        investigation_id = f"depth_first_{int(time.time())}"
        steps = []
        
        # Analyze query to identify primary focus area
        focus_area = self._identify_focus_area(query)
        
        # Create deep investigation steps for the focus area
        if focus_area == "code":
            steps.extend(self._create_code_investigation_steps(query))
        elif focus_area == "file_system":
            steps.extend(self._create_file_investigation_steps(query))
        elif focus_area == "implementation":
            steps.extend(self._create_implementation_investigation_steps(query))
        elif focus_area == "debugging":
            steps.extend(self._create_debugging_investigation_steps(query))
        else:
            steps.extend(self._create_general_investigation_steps(query))
        
        total_duration = sum(step.estimated_duration for step in steps)
        
        return InvestigationPlan(
            investigation_id=investigation_id,
            query=query,
            strategy=InvestigationStrategy.DEPTH_FIRST,
            steps=steps,
            total_estimated_duration=total_duration,
            success_criteria=[
                "Primary focus area thoroughly analyzed",
                "All related components identified",
                "Detailed understanding achieved",
                "Actionable insights generated"
            ],
            fallback_strategies=[InvestigationStrategy.BREADTH_FIRST, InvestigationStrategy.TARGETED]
        )
    
    def _identify_focus_area(self, query: str) -> str:
        """Identify the primary focus area from the query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["implement", "create", "build", "develop"]):
            return "implementation"
        elif any(word in query_lower for word in ["debug", "error", "fix", "issue", "problem"]):
            return "debugging"
        elif any(word in query_lower for word in ["file", "directory", "folder", "structure"]):
            return "file_system"
        elif any(word in query_lower for word in ["code", "function", "class", "method"]):
            return "code"
        else:
            return "general"
    
    def _create_code_investigation_steps(self, query: str) -> List[InvestigationStep]:
        """Create steps for deep code investigation."""
        steps = []
        
        # Step 1: Identify relevant code files
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("identify_code", 1),
            description="Identify all relevant code files and modules",
            strategy=self.strategy_type,
            priority=InvestigationPriority.CRITICAL,
            estimated_duration=self._estimate_duration("file_scan"),
            required_models=["qwen2.5:7b-instruct-q4_K_M"],
            expected_outputs=["list_of_relevant_files", "file_relationships"],
            validation_criteria=["Files are relevant to query", "No major files missed"]
        ))
        
        # Step 2: Analyze code structure
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("analyze_structure", 2),
            description="Analyze code structure and dependencies",
            strategy=self.strategy_type,
            priority=InvestigationPriority.HIGH,
            estimated_duration=self._estimate_duration("code_analysis", "complex"),
            required_models=["qwen2.5-coder:7b", "phi3:small", "llama3:8b"],  # Use models with verified tool support
            dependencies=[steps[0].step_id],
            expected_outputs=["dependency_graph", "code_structure_analysis"],
            validation_criteria=["All dependencies identified", "Structure is clear"]
        ))
        
        # Step 3: Deep dive into implementation details
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("deep_analysis", 3),
            description="Deep analysis of implementation details",
            strategy=self.strategy_type,
            priority=InvestigationPriority.HIGH,
            estimated_duration=self._estimate_duration("code_analysis", "very_complex"),
            required_models=["deepseek-coder:6.7b-instruct-q4_K_M"],
            dependencies=[steps[1].step_id],
            expected_outputs=["detailed_code_analysis", "potential_improvements"],
            validation_criteria=["Implementation understood", "Issues identified"]
        ))
        
        return steps
    
    def _create_file_investigation_steps(self, query: str) -> List[InvestigationStep]:
        """Create steps for deep file system investigation."""
        steps = []
        
        # Step 1: Map file structure
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("map_structure", 1),
            description="Map complete file system structure",
            strategy=self.strategy_type,
            priority=InvestigationPriority.CRITICAL,
            estimated_duration=self._estimate_duration("file_scan", "complex"),
            required_models=["qwen2.5:7b-instruct-q4_K_M"],
            expected_outputs=["complete_file_tree", "file_categorization"],
            validation_criteria=["All directories mapped", "File types identified"]
        ))
        
        # Step 2: Analyze file relationships
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("analyze_relationships", 2),
            description="Analyze relationships between files",
            strategy=self.strategy_type,
            priority=InvestigationPriority.HIGH,
            estimated_duration=self._estimate_duration("code_analysis"),
            required_models=["gemma:7b-instruct-q4_K_M"],
            dependencies=[steps[0].step_id],
            expected_outputs=["file_dependency_map", "import_analysis"],
            validation_criteria=["Dependencies clear", "No circular references"]
        ))
        
        return steps
    
    def _create_implementation_investigation_steps(self, query: str) -> List[InvestigationStep]:
        """Create steps for implementation investigation."""
        steps = []
        
        # Step 1: Requirement analysis
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("analyze_requirements", 1),
            description="Analyze implementation requirements",
            strategy=self.strategy_type,
            priority=InvestigationPriority.CRITICAL,
            estimated_duration=self._estimate_duration("research"),
            required_models=["llama3.2:latest"],
            expected_outputs=["requirement_specification", "constraint_analysis"],
            validation_criteria=["Requirements clear", "Constraints identified"]
        ))
        
        # Step 2: Design architecture
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("design_architecture", 2),
            description="Design implementation architecture",
            strategy=self.strategy_type,
            priority=InvestigationPriority.HIGH,
            estimated_duration=self._estimate_duration("implementation", "complex"),
            required_models=["qwen2.5-coder:7b", "phi3:small"],  # Models with verified tool support
            dependencies=[steps[0].step_id],
            expected_outputs=["architecture_design", "component_specification"],
            validation_criteria=["Architecture is sound", "Components well-defined"]
        ))
        
        return steps
    
    def _create_debugging_investigation_steps(self, query: str) -> List[InvestigationStep]:
        """Create steps for debugging investigation."""
        steps = []
        
        # Step 1: Identify error sources
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("identify_errors", 1),
            description="Identify all error sources and symptoms",
            strategy=self.strategy_type,
            priority=InvestigationPriority.CRITICAL,
            estimated_duration=self._estimate_duration("code_analysis"),
            required_models=["deepseek-coder:6.7b-instruct-q4_K_M"],
            expected_outputs=["error_catalog", "symptom_analysis"],
            validation_criteria=["All errors identified", "Symptoms documented"]
        ))
        
        # Step 2: Trace error propagation
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("trace_errors", 2),
            description="Trace error propagation through code",
            strategy=self.strategy_type,
            priority=InvestigationPriority.HIGH,
            estimated_duration=self._estimate_duration("code_analysis", "complex"),
            required_models=["qwen2.5-coder:7b", "phi3:small"],  # Models with verified tool support
            dependencies=[steps[0].step_id],
            expected_outputs=["error_flow_analysis", "root_cause_identification"],
            validation_criteria=["Error flow understood", "Root causes found"]
        ))
        
        return steps
    
    def _create_general_investigation_steps(self, query: str) -> List[InvestigationStep]:
        """Create steps for general investigation."""
        steps = []
        
        # Step 1: Initial exploration
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("initial_exploration", 1),
            description="Initial exploration of the topic",
            strategy=self.strategy_type,
            priority=InvestigationPriority.HIGH,
            estimated_duration=self._estimate_duration("research"),
            required_models=["qwen2.5:7b-instruct-q4_K_M"],
            expected_outputs=["initial_findings", "investigation_direction"],
            validation_criteria=["Topic understood", "Direction identified"]
        ))
        
        return steps
    
    def adapt_plan(self, plan: InvestigationPlan, 
                   execution_results: List[Dict[str, Any]]) -> InvestigationPlan:
        """Adapt depth-first plan based on results."""
        # Analyze results to determine if we need to go deeper
        new_steps = list(plan.steps)
        
        for result in execution_results:
            if result.get("confidence_level") == "low":
                # Add deeper investigation steps
                deeper_step = InvestigationStep(
                    step_id=self._generate_step_id("deeper_analysis", len(new_steps)),
                    description=f"Deeper analysis based on {result.get('step_id')}",
                    strategy=self.strategy_type,
                    priority=InvestigationPriority.HIGH,
                    estimated_duration=self._estimate_duration("code_analysis", "complex"),
                    required_models=["qwen2.5-coder:7b", "phi3:small"],  # Models with verified tool support
                    dependencies=[result.get("step_id")],
                    expected_outputs=["enhanced_understanding"],
                    validation_criteria=["Deeper insight achieved"]
                )
                new_steps.append(deeper_step)
        
        return InvestigationPlan(
            investigation_id=plan.investigation_id + "_adapted",
            query=plan.query,
            strategy=plan.strategy,
            steps=new_steps,
            total_estimated_duration=sum(step.estimated_duration for step in new_steps),
            success_criteria=plan.success_criteria,
            fallback_strategies=plan.fallback_strategies
        )


class BreadthFirstStrategy(BaseInvestigationStrategy):
    """Breadth-first investigation strategy - explore all areas broadly first."""
    
    def __init__(self):
        super().__init__(InvestigationStrategy.BREADTH_FIRST)
    
    def create_investigation_plan(self, query: str, context: Dict[str, Any]) -> InvestigationPlan:
        """Create breadth-first investigation plan."""
        investigation_id = f"breadth_first_{int(time.time())}"
        steps = []
        
        # Create broad overview steps first
        steps.extend(self._create_overview_steps(query))
        
        # Then create specific area investigations
        areas = self._identify_investigation_areas(query)
        for area in areas:
            steps.extend(self._create_area_specific_steps(area, query))
        
        total_duration = sum(step.estimated_duration for step in steps)
        
        return InvestigationPlan(
            investigation_id=investigation_id,
            query=query,
            strategy=InvestigationStrategy.BREADTH_FIRST,
            steps=steps,
            total_estimated_duration=total_duration,
            success_criteria=[
                "All relevant areas identified",
                "Broad understanding achieved",
                "Priority areas determined",
                "Comprehensive overview complete"
            ],
            fallback_strategies=[InvestigationStrategy.TARGETED, InvestigationStrategy.DEPTH_FIRST]
        )
    
    def _create_overview_steps(self, query: str) -> List[InvestigationStep]:
        """Create broad overview steps."""
        steps = []
        
        # Step 1: High-level scan
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("high_level_scan", 1),
            description="High-level scan of the entire project",
            strategy=self.strategy_type,
            priority=InvestigationPriority.CRITICAL,
            estimated_duration=self._estimate_duration("file_scan"),
            required_models=["qwen2.5:7b-instruct-q4_K_M"],
            expected_outputs=["project_overview", "component_identification"],
            validation_criteria=["All major components identified"]
        ))
        
        # Step 2: Quick assessment
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("quick_assessment", 2),
            description="Quick assessment of all identified components",
            strategy=self.strategy_type,
            priority=InvestigationPriority.HIGH,
            estimated_duration=self._estimate_duration("code_analysis"),
            required_models=["gemma:7b-instruct-q4_K_M"],
            dependencies=[steps[0].step_id],
            expected_outputs=["component_assessment", "priority_ranking"],
            validation_criteria=["All components assessed", "Priorities clear"]
        ))
        
        return steps
    
    def _identify_investigation_areas(self, query: str) -> List[str]:
        """Identify different areas that need investigation."""
        areas = ["file_system", "code_structure", "dependencies"]
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["implement", "create", "build"]):
            areas.extend(["requirements", "architecture", "testing"])
        
        if any(word in query_lower for word in ["debug", "error", "fix"]):
            areas.extend(["error_analysis", "logging", "validation"])
        
        if any(word in query_lower for word in ["performance", "optimize"]):
            areas.extend(["performance_analysis", "bottlenecks"])
        
        return list(set(areas))  # Remove duplicates
    
    def _create_area_specific_steps(self, area: str, query: str) -> List[InvestigationStep]:
        """Create steps for a specific investigation area."""
        steps = []
        
        step = InvestigationStep(
            step_id=self._generate_step_id(f"investigate_{area}", 1),
            description=f"Investigate {area.replace('_', ' ')} aspects",
            strategy=self.strategy_type,
            priority=InvestigationPriority.MEDIUM,
            estimated_duration=self._estimate_duration("code_analysis"),
            required_models=self._get_optimal_model_for_area(area),
            expected_outputs=[f"{area}_analysis"],
            validation_criteria=[f"{area.replace('_', ' ')} thoroughly analyzed"]
        )
        
        steps.append(step)
        return steps
    
    def _get_optimal_model_for_area(self, area: str) -> List[str]:
        """Get optimal model for investigating a specific area."""
        # Use models we know support tools for code-related tasks
        model_mapping = {
            "file_system": ["qwen2.5:7b-instruct-q4_K_M"],
            "code_structure": ["qwen2.5-coder:7b", "phi3:small", "llama3:8b"],
            "dependencies": ["qwen2.5-coder:7b", "phi3:small"],
            "requirements": ["qwen2.5:7b-instruct-q4_K_M"],
            "architecture": ["qwen2.5-coder:7b", "phi3:small", "llama3:8b"],
            "testing": ["qwen2.5-coder:7b", "phi3:small"],
            "error_analysis": ["qwen2.5-coder:7b", "phi3:small"],
            "performance_analysis": ["qwen2.5-coder:7b", "phi3:small"]
        }
        
        return model_mapping.get(area, ["qwen2.5:7b-instruct-q4_K_M"])
    
    def adapt_plan(self, plan: InvestigationPlan, 
                   execution_results: List[Dict[str, Any]]) -> InvestigationPlan:
        """Adapt breadth-first plan based on results."""
        # Analyze which areas need more detailed investigation
        new_steps = list(plan.steps)
        
        high_priority_areas = []
        for result in execution_results:
            if result.get("found_issues") or result.get("requires_deeper_analysis"):
                high_priority_areas.append(result.get("investigation_area"))
        
        # Add detailed investigation steps for high-priority areas
        for area in high_priority_areas:
            detailed_step = InvestigationStep(
                step_id=self._generate_step_id(f"detailed_{area}", len(new_steps)),
                description=f"Detailed investigation of {area}",
                strategy=self.strategy_type,
                priority=InvestigationPriority.HIGH,
                estimated_duration=self._estimate_duration("code_analysis", "complex"),
                required_models=self._get_optimal_model_for_area(area),
                expected_outputs=[f"detailed_{area}_analysis"],
                validation_criteria=[f"Detailed understanding of {area}"]
            )
            new_steps.append(detailed_step)
        
        return InvestigationPlan(
            investigation_id=plan.investigation_id + "_adapted",
            query=plan.query,
            strategy=plan.strategy,
            steps=new_steps,
            total_estimated_duration=sum(step.estimated_duration for step in new_steps),
            success_criteria=plan.success_criteria,
            fallback_strategies=plan.fallback_strategies
        )


class TargetedStrategy(BaseInvestigationStrategy):
    """Targeted investigation strategy - focus on specific objectives."""
    
    def __init__(self):
        super().__init__(InvestigationStrategy.TARGETED)
    
    def create_investigation_plan(self, query: str, context: Dict[str, Any]) -> InvestigationPlan:
        """Create targeted investigation plan."""
        investigation_id = f"targeted_{int(time.time())}"
        
        # Extract specific targets from the query
        targets = self._extract_targets(query)
        steps = []
        
        for target in targets:
            steps.extend(self._create_target_specific_steps(target, query))
        
        # Add validation step
        if steps:
            validation_step = InvestigationStep(
                step_id=self._generate_step_id("validate_targets", len(steps)),
                description="Validate that all targets have been addressed",
                strategy=self.strategy_type,
                priority=InvestigationPriority.HIGH,
                estimated_duration=self._estimate_duration("validation"),
                required_models=["gemma:7b-instruct-q4_K_M"],
                dependencies=[step.step_id for step in steps],
                expected_outputs=["validation_report"],
                validation_criteria=["All targets addressed", "Objectives met"]
            )
            steps.append(validation_step)
        
        total_duration = sum(step.estimated_duration for step in steps)
        
        return InvestigationPlan(
            investigation_id=investigation_id,
            query=query,
            strategy=InvestigationStrategy.TARGETED,
            steps=steps,
            total_estimated_duration=total_duration,
            success_criteria=[
                "All specified targets investigated",
                "Specific objectives achieved",
                "Targeted results obtained"
            ],
            fallback_strategies=[InvestigationStrategy.BREADTH_FIRST, InvestigationStrategy.EXPLORATORY]
        )
    
    def _extract_targets(self, query: str) -> List[str]:
        """Extract specific targets from the query."""
        targets = []
        
        # Look for specific file mentions
        file_patterns = re.findall(r'[\w/\\]+\.\w+', query)
        targets.extend(file_patterns)
        
        # Look for specific function/class mentions
        code_patterns = re.findall(r'\b[A-Z][a-zA-Z0-9_]*\b', query)
        targets.extend(code_patterns)
        
        # Look for specific keywords that indicate targets
        target_keywords = {
            "implement": "implementation",
            "fix": "bug_fix",
            "optimize": "optimization",
            "test": "testing",
            "document": "documentation",
            "analyze": "analysis",
            "debug": "debugging"
        }
        
        query_lower = query.lower()
        for keyword, target in target_keywords.items():
            if keyword in query_lower:
                targets.append(target)
        
        return list(set(targets))  # Remove duplicates
    
    def _create_target_specific_steps(self, target: str, query: str) -> List[InvestigationStep]:
        """Create steps for investigating a specific target."""
        steps = []
        
        # Determine the type of target and create appropriate steps
        if target.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.c')):
            # File target
            steps.append(InvestigationStep(
                step_id=self._generate_step_id(f"analyze_file_{target.replace('.', '_')}", 1),
                description=f"Analyze file {target}",
                strategy=self.strategy_type,
                priority=InvestigationPriority.HIGH,
                estimated_duration=self._estimate_duration("code_analysis"),
                required_models=["qwen2.5-coder:7b", "phi3:small"],  # Models with verified tool support
                expected_outputs=[f"file_analysis_{target}"],
                validation_criteria=[f"File {target} thoroughly analyzed"]
            ))
        
        elif target in ["implementation", "bug_fix", "optimization", "testing", "documentation"]:
            # Task target
            steps.append(InvestigationStep(
                step_id=self._generate_step_id(f"execute_{target}", 1),
                description=f"Execute {target} task",
                strategy=self.strategy_type,
                priority=InvestigationPriority.CRITICAL,
                estimated_duration=self._estimate_duration(target.replace("_", ""), "complex"),
                required_models=self._get_model_for_task(target),
                expected_outputs=[f"{target}_result"],
                validation_criteria=[f"{target.replace('_', ' ')} completed successfully"]
            ))
        
        else:
            # Generic target
            steps.append(InvestigationStep(
                step_id=self._generate_step_id(f"investigate_{target}", 1),
                description=f"Investigate {target}",
                strategy=self.strategy_type,
                priority=InvestigationPriority.MEDIUM,
                estimated_duration=self._estimate_duration("research"),
                required_models=["qwen2.5:7b-instruct-q4_K_M"],
                expected_outputs=[f"{target}_investigation"],
                validation_criteria=[f"{target} investigated thoroughly"]
            ))
        
        return steps
    
    def _get_model_for_task(self, task: str) -> List[str]:
        """Get optimal model for a specific task."""
        # Use models we know support tools for code-related tasks
        task_models = {
            "implementation": ["qwen2.5-coder:7b", "phi3:small", "llama3:8b"],
            "bug_fix": ["qwen2.5-coder:7b", "phi3:small"],
            "optimization": ["qwen2.5-coder:7b", "phi3:small"],
            "testing": ["qwen2.5-coder:7b", "phi3:small"],
            "documentation": ["qwen2.5:7b-instruct-q4_K_M"],
            "analysis": ["qwen2.5:7b-instruct-q4_K_M"],
            "debugging": ["qwen2.5-coder:7b", "phi3:small", "llama3:8b"]
        }
        
        return task_models.get(task, ["qwen2.5:7b-instruct-q4_K_M"])
    
    def adapt_plan(self, plan: InvestigationPlan, 
                   execution_results: List[Dict[str, Any]]) -> InvestigationPlan:
        """Adapt targeted plan based on results."""
        new_steps = list(plan.steps)
        
        # Add follow-up steps for targets that need more work
        for result in execution_results:
            if not result.get("target_achieved"):
                follow_up_step = InvestigationStep(
                    step_id=self._generate_step_id("follow_up", len(new_steps)),
                    description=f"Follow up on {result.get('target')}",
                    strategy=self.strategy_type,
                    priority=InvestigationPriority.HIGH,
                    estimated_duration=self._estimate_duration("research"),
                    required_models=["qwen2.5-coder:7b", "phi3:small"],  # Models with verified tool support
                    dependencies=[result.get("step_id")],
                    expected_outputs=["follow_up_results"],
                    validation_criteria=["Target achieved"]
                )
                new_steps.append(follow_up_step)
        
        return InvestigationPlan(
            investigation_id=plan.investigation_id + "_adapted",
            query=plan.query,
            strategy=plan.strategy,
            steps=new_steps,
            total_estimated_duration=sum(step.estimated_duration for step in new_steps),
            success_criteria=plan.success_criteria,
            fallback_strategies=plan.fallback_strategies
        )


class InvestigationStrategyManager:
    """Manages different investigation strategies and selects optimal approaches."""
    
    def __init__(self, context_manager: Optional[ContextManager] = None):
        """Initialize the strategy manager.
        
        Args:
            context_manager: Context manager instance (optional)
        """
        self.context_manager = context_manager or get_context_manager()
        self.logger = get_logger()
        
        # Initialize strategy implementations
        self.strategies = {
            InvestigationStrategy.DEPTH_FIRST: DepthFirstStrategy(),
            InvestigationStrategy.BREADTH_FIRST: BreadthFirstStrategy(),
            InvestigationStrategy.TARGETED: TargetedStrategy(),
            # Note: Other strategies can be added here as needed
        }
        
        self.strategy_performance = {}  # Track performance of different strategies
    
    def select_optimal_strategy(self, query: str, context: Dict[str, Any]) -> InvestigationStrategy:
        """Select the optimal investigation strategy for a query.
        
        Args:
            query: User query
            context: Additional context information
            
        Returns:
            Optimal investigation strategy
        """
        query_lower = query.lower()
        
        # Rule-based strategy selection
        if any(word in query_lower for word in ["specific", "particular", "exactly", "precisely"]):
            return InvestigationStrategy.TARGETED
        
        elif any(word in query_lower for word in ["overview", "summary", "all", "everything"]):
            return InvestigationStrategy.BREADTH_FIRST
        
        elif any(word in query_lower for word in ["deep", "detailed", "thorough", "comprehensive"]):
            return InvestigationStrategy.DEPTH_FIRST
        
        elif any(word in query_lower for word in ["explore", "discover", "find", "search"]):
            return InvestigationStrategy.BREADTH_FIRST
        
        else:
            # Default to breadth-first for general queries
            return InvestigationStrategy.BREADTH_FIRST
    
    def create_investigation_plan(self, query: str, 
                                strategy: Optional[InvestigationStrategy] = None,
                                context: Optional[Dict[str, Any]] = None) -> InvestigationPlan:
        """Create an investigation plan using the specified or optimal strategy.
        
        Args:
            query: User query
            strategy: Investigation strategy (optional, will be auto-selected)
            context: Additional context (optional)
            
        Returns:
            Investigation plan
        """
        if strategy is None:
            strategy = self.select_optimal_strategy(query, context or {})
        
        if strategy not in self.strategies:
            self.logger.warning(f"Strategy {strategy} not implemented, using breadth-first")
            strategy = InvestigationStrategy.BREADTH_FIRST
        
        strategy_impl = self.strategies[strategy]
        plan = strategy_impl.create_investigation_plan(query, context or {})
        
        self.logger.info(f"Created {strategy.value} investigation plan with "
                        f"{len(plan.steps)} steps, estimated duration: "
                        f"{plan.total_estimated_duration:.0f} seconds")
        
        return plan
    
    def adapt_investigation_plan(self, plan: InvestigationPlan,
                               execution_results: List[Dict[str, Any]]) -> InvestigationPlan:
        """Adapt an investigation plan based on execution results.
        
        Args:
            plan: Original investigation plan
            execution_results: Results from executed steps
            
        Returns:
            Adapted investigation plan
        """
        if plan.strategy not in self.strategies:
            self.logger.error(f"Cannot adapt plan with unknown strategy {plan.strategy}")
            return plan
        
        strategy_impl = self.strategies[plan.strategy]
        adapted_plan = strategy_impl.adapt_plan(plan, execution_results)
        
        self.logger.info(f"Adapted {plan.strategy.value} plan: "
                        f"{len(plan.steps)} -> {len(adapted_plan.steps)} steps")
        
        return adapted_plan
    
    def record_strategy_performance(self, strategy: InvestigationStrategy,
                                  execution_time: float, success_rate: float,
                                  user_satisfaction: float) -> None:
        """Record performance metrics for a strategy.
        
        Args:
            strategy: Investigation strategy
            execution_time: Total execution time
            success_rate: Success rate (0-1)
            user_satisfaction: User satisfaction score (0-1)
        """
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = []
        
        self.strategy_performance[strategy].append({
            "execution_time": execution_time,
            "success_rate": success_rate,
            "user_satisfaction": user_satisfaction,
            "timestamp": time.time()
        })
        
        # Keep only recent performance data
        if len(self.strategy_performance[strategy]) > 10:
            self.strategy_performance[strategy] = self.strategy_performance[strategy][-10:]
    
    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for all strategies."""
        stats = {}
        
        for strategy, performance_data in self.strategy_performance.items():
            if performance_data:
                avg_time = sum(p["execution_time"] for p in performance_data) / len(performance_data)
                avg_success = sum(p["success_rate"] for p in performance_data) / len(performance_data)
                avg_satisfaction = sum(p["user_satisfaction"] for p in performance_data) / len(performance_data)
                
                stats[strategy.value] = {
                    "average_execution_time": avg_time,
                    "average_success_rate": avg_success,
                    "average_user_satisfaction": avg_satisfaction,
                    "total_uses": len(performance_data)
                }
        
        return stats


__all__ = [
    "InvestigationStrategyManager", "InvestigationStrategy", "InvestigationPlan", 
    "InvestigationStep", "InvestigationPriority", "DepthFirstStrategy", 
    "BreadthFirstStrategy", "TargetedStrategy"
]
