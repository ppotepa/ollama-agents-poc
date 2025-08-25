"""Improved investigation strategies for better prompt recognition and model selection."""

import re
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum, auto
from src.utils.enhanced_logging import get_logger

# Import original classes
from src.core.investigation_strategies import (
    InvestigationStrategy, 
    InvestigationStep, 
    InvestigationPlan,
    InvestigationStrategyManager,
    InvestigationPriority,
    TaskAnalyzer
)

# Import for model capability checking
from src.core.model_capability_checker import get_capability_checker
from src.core.model_tool_support import test_model_tool_support

class TaskType(Enum):
    """Expanded task type classification for better routing."""
    GENERAL_QA = auto()
    CODE_GENERATION = auto()
    CODE_ANALYSIS = auto()
    CODE_OPTIMIZATION = auto()
    CODE_DEBUGGING = auto()
    FILE_OPERATIONS = auto()
    DEPENDENCY_MANAGEMENT = auto()
    ARCHITECTURE_DESIGN = auto()
    TESTING = auto()
    DOCUMENTATION = auto()
    SYSTEM_OPERATIONS = auto()
    RESEARCH = auto()
    UNKNOWN = auto()

class TaskComplexity(Enum):
    """Task complexity levels for better model assignment."""
    SIMPLE = auto()
    MEDIUM = auto()
    COMPLEX = auto()
    VERY_COMPLEX = auto()

class PromptPatterns:
    """Common patterns in prompts to help identify intent and complexity."""
    
    # Task type patterns
    CODE_PATTERNS = [
        r'(write|implement|code|create|develop)\s+(a|an|the)?\s+(function|class|method|script)',
        r'(fix|debug|solve)\s+(this|the|a|an)?\s+(bug|issue|error|problem)',
        r'(optimize|improve|refactor)\s+(this|the|a|an)?\s+(code|function|method|class)',
        r'(analyze|review|examine)\s+(this|the|a|an)?\s+(code|codebase|repository)',
    ]
    
    FILE_PATTERNS = [
        r'(create|make|write|generate)\s+(a|an|the)?\s+(file|directory)',
        r'(read|open|load)\s+(from|the|a|an)?\s+(file|directory|folder)',
        r'(list|show|display)\s+(the|all|files|directories)',
        r'(move|rename|copy|delete)\s+(the|a|an|this)?\s+(file|directory|folder)',
    ]
    
    ARCHITECTURE_PATTERNS = [
        r'(design|architect|structure)\s+(a|an|the)?\s+(system|application|service)',
        r'(create|make|design)\s+(a|an|the)?\s+(architecture|structure|diagram)',
    ]
    
    DEPENDENCY_PATTERNS = [
        r'(install|add|update|upgrade|use)\s+(a|an|the)?\s+(package|library|dependency|module)',
        r'(manage|handle|resolve)\s+(the|a|an|these)?\s+(dependencies|packages|libraries)',
    ]
    
    TESTING_PATTERNS = [
        r'(write|create|implement)\s+(a|an|the)?\s+(test|tests|unit test|integration test)',
        r'(test|verify|validate|check)\s+(this|the|a|an)?\s+(code|function|class|method|system)',
    ]
    
    DOCUMENTATION_PATTERNS = [
        r'(document|write|create)\s+(a|an|the)?\s+(documentation|readme|comments)',
        r'(explain|describe|elaborate)\s+(on|about|the|this|how)?\s+(code|system|function)',
    ]
    
    SYSTEM_PATTERNS = [
        r'(deploy|run|execute|start|stop)\s+(the|a|an|this)?\s+(application|service|server|container)',
        r'(setup|configure|install)\s+(a|an|the)?\s+(environment|system|server)',
    ]
    
    RESEARCH_PATTERNS = [
        r'(research|investigate|explore|study|learn)\s+(about|on|the|this)?\s+(\w+)',
        r'(find|search\s+for|look\s+up)\s+(information|details|examples)\s+(about|on|of)',
    ]
    
    # Complexity patterns
    COMPLEX_INDICATORS = [
        r'complex',
        r'complicated',
        r'advanced',
        r'sophisticated',
        r'comprehensive',
        r'end-to-end',
        r'complete\s+system',
        r'full\s+implementation',
        r'multiple\s+(steps|parts|components)',
        r'integrate\s+with',
    ]
    
    MULTI_TASK_INDICATORS = [
        r'(\d+)\s+steps?',
        r'(first|second|third|lastly|finally|then|next)',
        r'(and|also|additionally|moreover)\s+(.*?)(create|implement|write|design)',
        r'(several|multiple|many)\s+(tasks|things|actions|operations)',
    ]

class ImprovedTaskAnalyzer(TaskAnalyzer):
    """Enhanced task analyzer with better prompt recognition."""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger()
    
    def analyze_task(self, query: str) -> Dict[str, Any]:
        """Analyze a user query to determine its characteristics.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with task characteristics
        """
        # Start with original analysis
        original_analysis = super().analyze_task(query)
        
        # Enhanced analysis
        task_type = self._detect_task_type(query)
        complexity = self._assess_complexity(query)
        subtasks = self._identify_subtasks(query)
        tools_required = self._requires_tools(query, task_type)
        
        # Combine with original analysis and enhance
        return {
            **original_analysis,
            "task_type": task_type.name,
            "complexity": complexity.name,
            "subtasks": subtasks,
            "tools_required": tools_required,
            "estimated_steps": len(subtasks) if subtasks else self._estimate_steps(query, complexity)
        }
    
    def _detect_task_type(self, query: str) -> TaskType:
        """Detect the type of task based on the query using regex patterns."""
        query = query.lower()
        
        # Check code patterns
        for pattern in PromptPatterns.CODE_PATTERNS:
            if re.search(pattern, query):
                if any(term in query for term in ["bug", "issue", "error", "fix", "debug", "solve"]):
                    return TaskType.CODE_DEBUGGING
                elif any(term in query for term in ["optimize", "improve", "refactor", "performance"]):
                    return TaskType.CODE_OPTIMIZATION
                elif any(term in query for term in ["analyze", "review", "examine", "understand"]):
                    return TaskType.CODE_ANALYSIS
                else:
                    return TaskType.CODE_GENERATION
        
        # Check file patterns
        for pattern in PromptPatterns.FILE_PATTERNS:
            if re.search(pattern, query):
                return TaskType.FILE_OPERATIONS
        
        # Check architecture patterns
        for pattern in PromptPatterns.ARCHITECTURE_PATTERNS:
            if re.search(pattern, query):
                return TaskType.ARCHITECTURE_DESIGN
        
        # Check dependency patterns
        for pattern in PromptPatterns.DEPENDENCY_PATTERNS:
            if re.search(pattern, query):
                return TaskType.DEPENDENCY_MANAGEMENT
        
        # Check testing patterns
        for pattern in PromptPatterns.TESTING_PATTERNS:
            if re.search(pattern, query):
                return TaskType.TESTING
        
        # Check documentation patterns
        for pattern in PromptPatterns.DOCUMENTATION_PATTERNS:
            if re.search(pattern, query):
                return TaskType.DOCUMENTATION
        
        # Check system patterns
        for pattern in PromptPatterns.SYSTEM_PATTERNS:
            if re.search(pattern, query):
                return TaskType.SYSTEM_OPERATIONS
        
        # Check research patterns
        for pattern in PromptPatterns.RESEARCH_PATTERNS:
            if re.search(pattern, query):
                return TaskType.RESEARCH
        
        # Look for code keywords if no specific patterns matched
        code_keywords = ["code", "program", "function", "script", "class", "method", "variable"]
        if any(keyword in query for keyword in code_keywords):
            return TaskType.CODE_ANALYSIS
        
        # Default to general QA if no specific type identified
        return TaskType.GENERAL_QA
    
    def _assess_complexity(self, query: str) -> TaskComplexity:
        """Assess the complexity of the task."""
        query = query.lower()
        
        # Check for explicit complexity indicators
        for pattern in PromptPatterns.COMPLEX_INDICATORS:
            if re.search(pattern, query):
                return TaskComplexity.COMPLEX
        
        # Check for multi-task indicators
        for pattern in PromptPatterns.MULTI_TASK_INDICATORS:
            if re.search(pattern, query):
                return TaskComplexity.COMPLEX
        
        # Length-based complexity
        words = query.split()
        if len(words) > 100:
            return TaskComplexity.COMPLEX
        elif len(words) > 50:
            return TaskComplexity.MEDIUM
        
        # Default to simple
        return TaskComplexity.SIMPLE
    
    def _identify_subtasks(self, query: str) -> List[str]:
        """Identify potential subtasks in a complex query."""
        subtasks = []
        
        # Look for numbered steps
        numbered_steps = re.findall(r'(\d+)\.\s*([^.!?]*[.!?])', query)
        if numbered_steps:
            subtasks = [step[1].strip() for step in numbered_steps]
        
        # Look for transition words indicating steps
        if not subtasks:
            step_indicators = ["first", "second", "third", "finally", "lastly", "then", "next"]
            for indicator in step_indicators:
                matches = re.findall(f"{indicator}[,:]?\s*([^.!?]*[.!?])", query, re.IGNORECASE)
                subtasks.extend(match.strip() for match in matches)
        
        return subtasks
    
    def _requires_tools(self, query: str, task_type: TaskType) -> bool:
        """Determine if the task is likely to require tools."""
        # Some task types almost always require tools
        if task_type in [
            TaskType.FILE_OPERATIONS, 
            TaskType.CODE_DEBUGGING,
            TaskType.SYSTEM_OPERATIONS,
            TaskType.DEPENDENCY_MANAGEMENT
        ]:
            return True
            
        # Look for specific tool indicators
        tool_indicators = [
            "file", "directory", "folder", "read", "write", "create", 
            "modify", "delete", "install", "run", "execute", "terminal",
            "command", "search", "find in", "code from", "repository"
        ]
        
        query = query.lower()
        for indicator in tool_indicators:
            if indicator in query:
                return True
                
        return False
    
    def _estimate_steps(self, query: str, complexity: TaskComplexity) -> int:
        """Estimate the number of steps needed for the task."""
        base_steps = {
            TaskComplexity.SIMPLE: 1,
            TaskComplexity.MEDIUM: 2,
            TaskComplexity.COMPLEX: 4,
            TaskComplexity.VERY_COMPLEX: 8
        }
        
        # Start with base steps for the complexity level
        steps = base_steps[complexity]
        
        # Adjust based on query length
        words = query.split()
        steps += len(words) // 50  # Add one step per 50 words
        
        # Add steps for specific operations mentioned
        operations = [
            "install", "configure", "implement", "test", "optimize",
            "debug", "document", "deploy", "analyze", "design"
        ]
        
        query_lower = query.lower()
        for op in operations:
            if op in query_lower:
                steps += 1
                
        return steps

class ModelScoringCache:
    """Cache to store and learn from model performance for different task types."""
    
    def __init__(self):
        self.scores = {}  # Format: {task_type: {model_name: {success: count, failure: count}}}
        self.logger = get_logger()
    
    def record_success(self, task_type: str, model_name: str):
        """Record a successful use of a model for a task type."""
        if task_type not in self.scores:
            self.scores[task_type] = {}
        if model_name not in self.scores[task_type]:
            self.scores[task_type][model_name] = {"success": 0, "failure": 0}
        
        self.scores[task_type][model_name]["success"] += 1
    
    def record_failure(self, task_type: str, model_name: str):
        """Record a failed use of a model for a task type."""
        if task_type not in self.scores:
            self.scores[task_type] = {}
        if model_name not in self.scores[task_type]:
            self.scores[task_type][model_name] = {"success": 0, "failure": 0}
        
        self.scores[task_type][model_name]["failure"] += 1
    
    def get_success_rate(self, task_type: str, model_name: str) -> float:
        """Get the success rate of a model for a task type."""
        if task_type not in self.scores or model_name not in self.scores[task_type]:
            return 0.0
        
        stats = self.scores[task_type][model_name]
        total = stats["success"] + stats["failure"]
        
        if total == 0:
            return 0.0
        
        return stats["success"] / total
    
    def get_best_models(self, task_type: str, limit: int = 3) -> List[str]:
        """Get the best performing models for a task type."""
        if task_type not in self.scores:
            return []
        
        model_scores = []
        for model_name, stats in self.scores[task_type].items():
            total = stats["success"] + stats["failure"]
            if total > 0:  # Only consider models with some history
                success_rate = stats["success"] / total
                model_scores.append((model_name, success_rate, total))
        
        # Sort by success rate (desc) and then by number of samples (desc)
        model_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        return [model[0] for model in model_scores[:limit]]

# Initialize global scoring cache
model_scoring_cache = ModelScoringCache()

class ImprovedInvestigationStrategy(InvestigationStrategy):
    """Base class for improved investigation strategies."""
    
    def __init__(self, context_manager):
        super().__init__(context_manager)
        self.task_analyzer = ImprovedTaskAnalyzer()
        self.logger = get_logger()
        
    def _get_model_for_task_type(self, task_type: TaskType, requires_tools: bool) -> List[str]:
        """Get optimal models for a specific task type with adaptive selection."""
        capability_checker = get_capability_checker()
        
        # Get task type string representation for the model checker
        task_type_str = self._map_task_type_to_checker_format(task_type)
        
        # Try to get models with verified capabilities first
        best_model = capability_checker.get_best_model_for_task(task_type_str, requires_tools)
        if best_model:
            return [best_model]
        
        # Check if we have successful models from the cache
        cache_key = task_type.name  # Use enum name as key
        cached_models = model_scoring_cache.get_best_models(cache_key)
        if cached_models:
            # Filter for tools support if needed
            if requires_tools:
                tool_supporting_models = [
                    model for model in cached_models 
                    if test_model_tool_support(model)
                ]
                if tool_supporting_models:
                    return tool_supporting_models
            else:
                return cached_models
        
        # Fall back to improved hardcoded mappings
        model_mapping = self._get_improved_model_mapping(task_type, requires_tools)
        return model_mapping
        
    def _map_task_type_to_checker_format(self, task_type: TaskType) -> str:
        """Map task type enum to string format used by capability checker."""
        mapping = {
            TaskType.GENERAL_QA: "general_qa",
            TaskType.CODE_GENERATION: "coding",
            TaskType.CODE_ANALYSIS: "code_analysis",
            TaskType.CODE_OPTIMIZATION: "optimization",
            TaskType.CODE_DEBUGGING: "debugging",
            TaskType.FILE_OPERATIONS: "file_operations",
            TaskType.DEPENDENCY_MANAGEMENT: "dependencies",
            TaskType.ARCHITECTURE_DESIGN: "architecture",
            TaskType.TESTING: "testing",
            TaskType.DOCUMENTATION: "documentation",
            TaskType.SYSTEM_OPERATIONS: "system_operations",
            TaskType.RESEARCH: "research",
            TaskType.UNKNOWN: "general_qa"
        }
        return mapping.get(task_type, "general_qa")
        
    def _get_improved_model_mapping(self, task_type: TaskType, requires_tools: bool) -> List[str]:
        """Get improved model mappings based on task type and tool requirements."""
        # Models known to excel at specific tasks with good tool support
        if requires_tools:
            task_models = {
                TaskType.GENERAL_QA: ["qwen2.5:7b-instruct-q4_K_M"],
                TaskType.CODE_GENERATION: ["qwen2.5-coder:7b", "phi3:small", "llama3:8b"],
                TaskType.CODE_ANALYSIS: ["qwen2.5-coder:7b", "phi3:small"],
                TaskType.CODE_OPTIMIZATION: ["qwen2.5-coder:7b", "phi3:small"],
                TaskType.CODE_DEBUGGING: ["qwen2.5-coder:7b", "phi3:small", "llama3:8b"],
                TaskType.FILE_OPERATIONS: ["qwen2.5:7b-instruct-q4_K_M", "qwen2.5-coder:7b"],
                TaskType.DEPENDENCY_MANAGEMENT: ["qwen2.5-coder:7b", "phi3:small"],
                TaskType.ARCHITECTURE_DESIGN: ["qwen2.5-coder:7b", "llama3:8b"],
                TaskType.TESTING: ["qwen2.5-coder:7b", "phi3:small"],
                TaskType.DOCUMENTATION: ["qwen2.5:7b-instruct-q4_K_M"],
                TaskType.SYSTEM_OPERATIONS: ["qwen2.5-coder:7b", "phi3:small"],
                TaskType.RESEARCH: ["qwen2.5:7b-instruct-q4_K_M", "llama3:8b"],
                TaskType.UNKNOWN: ["qwen2.5:7b-instruct-q4_K_M"]
            }
        else:
            # For tasks not requiring tools, we can use a broader set of models
            task_models = {
                TaskType.GENERAL_QA: ["qwen2.5:7b-instruct-q4_K_M", "llama3:8b"],
                TaskType.CODE_GENERATION: ["qwen2.5-coder:7b", "phi3:small", "llama3:8b"],
                TaskType.CODE_ANALYSIS: ["qwen2.5-coder:7b", "phi3:small", "llama3:8b"],
                TaskType.CODE_OPTIMIZATION: ["qwen2.5-coder:7b", "phi3:small"],
                TaskType.CODE_DEBUGGING: ["qwen2.5-coder:7b", "phi3:small", "llama3:8b"],
                TaskType.FILE_OPERATIONS: ["qwen2.5:7b-instruct-q4_K_M", "qwen2.5-coder:7b"],
                TaskType.DEPENDENCY_MANAGEMENT: ["qwen2.5-coder:7b", "phi3:small"],
                TaskType.ARCHITECTURE_DESIGN: ["qwen2.5-coder:7b", "llama3:8b"],
                TaskType.TESTING: ["qwen2.5-coder:7b", "phi3:small"],
                TaskType.DOCUMENTATION: ["qwen2.5:7b-instruct-q4_K_M", "llama3:8b"],
                TaskType.SYSTEM_OPERATIONS: ["qwen2.5-coder:7b", "phi3:small"],
                TaskType.RESEARCH: ["qwen2.5:7b-instruct-q4_K_M", "llama3:8b"],
                TaskType.UNKNOWN: ["qwen2.5:7b-instruct-q4_K_M", "llama3:8b"]
            }
        
        return task_models.get(task_type, ["qwen2.5:7b-instruct-q4_K_M"])

    def _get_model_for_complexity(self, complexity: TaskComplexity, requires_tools: bool) -> List[str]:
        """Get optimal model based on task complexity."""
        if requires_tools:
            complexity_models = {
                TaskComplexity.SIMPLE: ["qwen2.5:7b-instruct-q4_K_M"],
                TaskComplexity.MEDIUM: ["qwen2.5-coder:7b", "phi3:small"],
                TaskComplexity.COMPLEX: ["qwen2.5-coder:7b"],
                TaskComplexity.VERY_COMPLEX: ["qwen2.5-coder:7b"]
            }
        else:
            complexity_models = {
                TaskComplexity.SIMPLE: ["qwen2.5:7b-instruct-q4_K_M", "phi3:small"],
                TaskComplexity.MEDIUM: ["qwen2.5:7b-instruct-q4_K_M", "llama3:8b"],
                TaskComplexity.COMPLEX: ["qwen2.5-coder:7b", "llama3:8b"],
                TaskComplexity.VERY_COMPLEX: ["qwen2.5-coder:7b"]
            }
        
        return complexity_models.get(complexity, ["qwen2.5:7b-instruct-q4_K_M"])
    
    def _select_models_for_query(self, query: str) -> List[str]:
        """Select models based on query analysis using enhanced methods."""
        # Analyze the query
        analysis = self.task_analyzer.analyze_task(query)
        
        # Get task type and complexity
        task_type = TaskType[analysis["task_type"]]
        complexity = TaskComplexity[analysis["complexity"]]
        requires_tools = analysis["tools_required"]
        
        # Get models for task type and complexity
        task_models = self._get_model_for_task_type(task_type, requires_tools)
        complexity_models = self._get_model_for_complexity(complexity, requires_tools)
        
        # Combine and prioritize models
        combined_models = []
        
        # First add models that appear in both lists (highest confidence)
        for model in task_models:
            if model in complexity_models:
                combined_models.append(model)
        
        # Then add remaining task models
        for model in task_models:
            if model not in combined_models:
                combined_models.append(model)
        
        # Then add remaining complexity models
        for model in complexity_models:
            if model not in combined_models:
                combined_models.append(model)
                
        # Add default fallback
        if "qwen2.5:7b-instruct-q4_K_M" not in combined_models:
            combined_models.append("qwen2.5:7b-instruct-q4_K_M")
            
        return combined_models

class ImprovedBreadthFirstStrategy(ImprovedInvestigationStrategy):
    """Enhanced breadth-first investigation strategy."""
    
    def __init__(self, context_manager):
        super().__init__(context_manager)
        self.strategy_type = "breadth-first"
    
    def create_plan(self, query: str) -> InvestigationPlan:
        """Create a breadth-first investigation plan with improved model selection."""
        # Analyze the query
        analysis = self.task_analyzer.analyze_task(query)
        
        # Extract task information
        task_type = TaskType[analysis["task_type"]]
        complexity = TaskComplexity[analysis["complexity"]]
        requires_tools = analysis["tools_required"]
        
        # Determine areas to investigate based on task type
        investigation_areas = self._determine_investigation_areas(task_type)
        
        # Create steps for each area
        steps = []
        for i, area in enumerate(investigation_areas):
            # Select models optimized for this area
            models = self._get_model_for_task_type(task_type, requires_tools)
            
            steps.append(InvestigationStep(
                step_id=self._generate_step_id(area, i),
                description=f"Investigate {area.lower().replace('_', ' ')}",
                strategy=self.strategy_type,
                priority=InvestigationPriority.MEDIUM,
                estimated_duration=self._estimate_duration("investigation"),
                required_models=models,
                expected_outputs=[f"{area.lower()}_findings"],
                validation_criteria=[f"{area.lower()} investigated thoroughly"]
            ))
        
        return InvestigationPlan(
            plan_id=str(uuid.uuid4()),
            steps=steps,
            original_query=query,
            strategy=self.strategy_type,
            creation_time=time.time()
        )
    
    def _determine_investigation_areas(self, task_type: TaskType) -> List[str]:
        """Determine relevant investigation areas based on task type."""
        # Map task types to investigation areas
        type_to_areas = {
            TaskType.GENERAL_QA: ["CONTEXT", "REQUIREMENTS", "INFORMATION_GATHERING"],
            TaskType.CODE_GENERATION: ["REQUIREMENTS", "CODE_STRUCTURE", "ARCHITECTURE", "IMPLEMENTATION"],
            TaskType.CODE_ANALYSIS: ["CODE_STRUCTURE", "ARCHITECTURE", "PERFORMANCE_ANALYSIS"],
            TaskType.CODE_OPTIMIZATION: ["CODE_STRUCTURE", "PERFORMANCE_ANALYSIS", "IMPLEMENTATION"],
            TaskType.CODE_DEBUGGING: ["CODE_STRUCTURE", "ERROR_ANALYSIS", "TESTING"],
            TaskType.FILE_OPERATIONS: ["FILE_SYSTEM", "REQUIREMENTS"],
            TaskType.DEPENDENCY_MANAGEMENT: ["DEPENDENCIES", "REQUIREMENTS"],
            TaskType.ARCHITECTURE_DESIGN: ["REQUIREMENTS", "ARCHITECTURE", "CODE_STRUCTURE"],
            TaskType.TESTING: ["CODE_STRUCTURE", "TESTING", "REQUIREMENTS"],
            TaskType.DOCUMENTATION: ["CODE_STRUCTURE", "REQUIREMENTS", "ARCHITECTURE"],
            TaskType.SYSTEM_OPERATIONS: ["REQUIREMENTS", "ARCHITECTURE", "IMPLEMENTATION"],
            TaskType.RESEARCH: ["INFORMATION_GATHERING", "CONTEXT", "REQUIREMENTS"],
            TaskType.UNKNOWN: ["CONTEXT", "REQUIREMENTS", "CODE_STRUCTURE", "ARCHITECTURE"]
        }
        
        return type_to_areas.get(task_type, ["CONTEXT", "REQUIREMENTS", "CODE_STRUCTURE"])

class ImprovedDepthFirstStrategy(ImprovedInvestigationStrategy):
    """Enhanced depth-first investigation strategy."""
    
    def __init__(self, context_manager):
        super().__init__(context_manager)
        self.strategy_type = "depth-first"
    
    def create_plan(self, query: str) -> InvestigationPlan:
        """Create a depth-first investigation plan with improved model selection."""
        # Analyze the query
        analysis = self.task_analyzer.analyze_task(query)
        
        # Extract task information
        task_type = TaskType[analysis["task_type"]]
        complexity = TaskComplexity[analysis["complexity"]]
        requires_tools = analysis["tools_required"]
        
        # Identify key investigation areas based on task type
        main_area = self._identify_main_area(task_type)
        
        # Create a progressive investigation with increasing depth
        steps = []
        
        # Step 1: Initial exploration
        models = self._get_model_for_task_type(task_type, requires_tools)
        steps.append(InvestigationStep(
            step_id=self._generate_step_id(main_area, 0),
            description=f"Initial exploration of {main_area.lower().replace('_', ' ')}",
            strategy=self.strategy_type,
            priority=InvestigationPriority.HIGH,
            estimated_duration=self._estimate_duration("investigation"),
            required_models=models,
            expected_outputs=["initial_findings"],
            validation_criteria=["Initial understanding established"]
        ))
        
        # Step 2: Detailed analysis
        steps.append(InvestigationStep(
            step_id=self._generate_step_id(main_area, 1),
            description=f"Detailed analysis of {main_area.lower().replace('_', ' ')}",
            strategy=self.strategy_type,
            priority=InvestigationPriority.HIGH,
            estimated_duration=self._estimate_duration("analysis"),
            required_models=models,
            dependencies=[steps[0].step_id],
            expected_outputs=["detailed_analysis"],
            validation_criteria=["Comprehensive understanding achieved"]
        ))
        
        # Step 3: Specific insights
        steps.append(InvestigationStep(
            step_id=self._generate_step_id(main_area, 2),
            description=f"Extract specific insights from {main_area.lower().replace('_', ' ')}",
            strategy=self.strategy_type,
            priority=InvestigationPriority.MEDIUM,
            estimated_duration=self._estimate_duration("analysis"),
            required_models=models,
            dependencies=[steps[1].step_id],
            expected_outputs=["specific_insights"],
            validation_criteria=["Key insights extracted"]
        ))
        
        # Step 4: Application
        steps.append(InvestigationStep(
            step_id=self._generate_step_id("APPLICATION", 3),
            description="Apply findings to original query",
            strategy=self.strategy_type,
            priority=InvestigationPriority.HIGH,
            estimated_duration=self._estimate_duration("application"),
            required_models=models,
            dependencies=[steps[2].step_id],
            expected_outputs=["application_results"],
            validation_criteria=["Findings successfully applied"]
        ))
        
        return InvestigationPlan(
            plan_id=str(uuid.uuid4()),
            steps=steps,
            original_query=query,
            strategy=self.strategy_type,
            creation_time=time.time()
        )
    
    def _identify_main_area(self, task_type: TaskType) -> str:
        """Identify the main investigation area for a task type."""
        type_to_main_area = {
            TaskType.GENERAL_QA: "CONTEXT",
            TaskType.CODE_GENERATION: "IMPLEMENTATION",
            TaskType.CODE_ANALYSIS: "CODE_STRUCTURE",
            TaskType.CODE_OPTIMIZATION: "PERFORMANCE_ANALYSIS",
            TaskType.CODE_DEBUGGING: "ERROR_ANALYSIS",
            TaskType.FILE_OPERATIONS: "FILE_SYSTEM",
            TaskType.DEPENDENCY_MANAGEMENT: "DEPENDENCIES",
            TaskType.ARCHITECTURE_DESIGN: "ARCHITECTURE",
            TaskType.TESTING: "TESTING",
            TaskType.DOCUMENTATION: "DOCUMENTATION",
            TaskType.SYSTEM_OPERATIONS: "SYSTEM",
            TaskType.RESEARCH: "INFORMATION_GATHERING",
            TaskType.UNKNOWN: "CONTEXT"
        }
        
        return type_to_main_area.get(task_type, "CONTEXT")

# Update with other strategy implementations as needed...

# Make sure to import necessary modules at the top
import time
import uuid
