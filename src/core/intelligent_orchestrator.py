"""Central Orchestration Layer - Coordinates intelligent investigation with model switching."""
from __future__ import annotations

import asyncio
import uuid
import time
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

from src.utils.enhanced_logging import get_logger
from src.core.context_manager import ContextManager, ExecutionContext, get_context_manager
from src.core.task_decomposer import TaskDecomposer, TaskDecomposition, TaskType, Subtask, TaskPriority
from src.core.execution_planner import ExecutionPlanner, ExecutionPlan, ExecutionStep
from src.core.reflection_system import ReflectionSystem, ReflectionResult, ReflectionTrigger, ConfidenceLevel
from src.core.investigation_strategies import InvestigationStrategyManager, InvestigationStrategy, InvestigationPlan
from src.core.model_capability_checker import get_capability_checker
from src.core.helpers import get_agent_instance
from src.agents.universal.agent import UniversalAgent
from src.core.model_discovery import model_exists, ensure_model_available, get_available_models


class OrchestrationState(Enum):
    """States of the orchestration process."""
    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    ADAPTING = "adapting"
    COMPLETED = "completed"
    ERROR = "error"
    PAUSED = "paused"


class ExecutionMode(Enum):
    """Execution modes for the orchestrator."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    SMART_ROUTING = "smart_routing"


@dataclass
class OrchestrationSession:
    """Represents an active orchestration session."""
    session_id: str
    original_query: str
    current_state: OrchestrationState
    execution_mode: ExecutionMode
    investigation_plan: Optional[InvestigationPlan]
    execution_plan: Optional[ExecutionPlan]
    task_decomposition: Optional[TaskDecomposition] = None
    current_step: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    model_switches: int = 0
    total_execution_time: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "original_query": self.original_query,
            "current_state": self.current_state.value,
            "execution_mode": self.execution_mode.value,
            "current_step": self.current_step,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "model_switches": self.model_switches,
            "total_execution_time": self.total_execution_time,
            "created_at": self.created_at,
            "last_activity": self.last_activity
        }


class IntelligentOrchestrator:
    """Central orchestrator for intelligent investigation with dynamic model switching."""
    
    def __init__(self, 
                 context_manager: Optional[ContextManager] = None,
                 enable_streaming: bool = True,
                 max_concurrent_steps: int = 3):
        """Initialize the intelligent orchestrator.
        
        Args:
            context_manager: Context manager instance (optional)
            enable_streaming: Whether to enable streaming mode for agents
            max_concurrent_steps: Maximum concurrent steps in parallel execution
        """
        self.context_manager = context_manager or get_context_manager()
        self.enable_streaming = enable_streaming
        self.max_concurrent_steps = max_concurrent_steps
        
        self.logger = get_logger()
        
        # Initialize core components
        self.task_decomposer = TaskDecomposer()
        self.execution_planner = ExecutionPlanner()
        self.reflection_system = ReflectionSystem(self.context_manager)
        self.strategy_manager = InvestigationStrategyManager(self.context_manager)
        
        # Session management
        self.active_sessions: Dict[str, OrchestrationSession] = {}
        self.agent_cache: Dict[str, UniversalAgent] = {}
        
        # Configuration
        self.config = {
            "max_session_duration": 3600,  # 1 hour
            "auto_reflection_interval": 300,  # 5 minutes
            "confidence_threshold": 0.6,
            "max_model_switches": 10,
            "enable_adaptive_planning": True,
            "enable_parallel_execution": True
        }
    
    async def start_investigation(self, 
                                query: str,
                                execution_mode: ExecutionMode = ExecutionMode.ADAPTIVE,
                                strategy: Optional[InvestigationStrategy] = None,
                                context: Optional[Dict[str, Any]] = None) -> str:
        """Start a new intelligent investigation.
        
        Args:
            query: User query to investigate
            execution_mode: How to execute the investigation
            strategy: Investigation strategy (optional, will be auto-selected)
            context: Additional context information
            
        Returns:
            Session ID for tracking the investigation
        """
        session_id = str(uuid.uuid4())
        
        # Create orchestration session
        session = OrchestrationSession(
            session_id=session_id,
            original_query=query,
            current_state=OrchestrationState.INITIALIZING,
            execution_mode=execution_mode,
            investigation_plan=None,  # Will be set after planning
            execution_plan=None,     # Will be set after planning
            current_step=None        # Will be set when execution starts
        )
        
        self.active_sessions[session_id] = session
        
        # Create execution context
        execution_context = self.context_manager.create_context(
            session_id=session_id,
            original_query=query
        )
        
        self.logger.info(f"Started investigation session {session_id} for query: {query[:100]}...")
        
        try:
            # Phase 1: Planning
            await self._planning_phase(session, strategy, context or {})
            
            # Phase 2: Execution
            await self._execution_phase(session)
            
        except Exception as e:
            self.logger.error(f"Investigation session {session_id} failed: {e}")
            session.current_state = OrchestrationState.ERROR
            self.context_manager.add_context_data(
                session_id, "error", str(e), "orchestrator"
            )
        
        return session_id
    
    async def _planning_phase(self, session: OrchestrationSession, 
                            strategy: Optional[InvestigationStrategy],
                            context: Dict[str, Any]) -> None:
        """Execute the planning phase of investigation.
        
        Args:
            session: Orchestration session
            strategy: Investigation strategy (optional)
            context: Additional context
        """
        session.current_state = OrchestrationState.PLANNING
        session.update_activity()
        
        self.logger.info(f"Planning phase for session {session.session_id}")
        
        # Step 1: Task Decomposition
        task_decomposition = await self._decompose_task(session.original_query, context)
        
        # Store decomposition in session and context
        session.task_decomposition = task_decomposition
        self.context_manager.add_context_data(
            session.session_id,
            "task_decomposition", 
            task_decomposition.to_dict(),
            "task_decomposer"
        )
        
        # Step 2: Investigation Strategy Selection
        if strategy is None:
            strategy = self.strategy_manager.select_optimal_strategy(
                session.original_query, context
            )
        
        # Step 3: Create Investigation Plan
        session.investigation_plan = self.strategy_manager.create_investigation_plan(
            session.original_query, strategy, context
        )
        
        # Store investigation plan
        self.context_manager.add_context_data(
            session.session_id,
            "investigation_plan",
            session.investigation_plan.to_dict(),
            "strategy_manager"
        )
        
        # Step 4: Create Execution Plan
        session.execution_plan = await self._create_execution_plan(
            task_decomposition, session.investigation_plan
        )
        
        # Store execution plan
        self.context_manager.add_context_data(
            session.session_id,
            "execution_plan",
            session.execution_plan.to_dict(),
            "execution_planner"
        )
        
        self.logger.info(f"Planning complete for session {session.session_id}: "
                        f"{len(session.execution_plan.execution_steps)} execution steps planned")
    
    async def _execution_phase(self, session: OrchestrationSession) -> None:
        """Execute the investigation according to the plan.
        
        Args:
            session: Orchestration session
        """
        session.current_state = OrchestrationState.EXECUTING
        session.update_activity()
        
        self.logger.info(f"Execution phase for session {session.session_id}")
        
        execution_results = []
        
        while not self._is_execution_complete(session):
            # Get next executable steps
            next_steps = self._get_next_executable_steps(session)
            
            if not next_steps:
                self.logger.warning(f"No executable steps found for session {session.session_id}")
                break
            
            # Execute steps based on execution mode
            if session.execution_mode == ExecutionMode.SEQUENTIAL:
                for step in next_steps[:1]:  # Execute one at a time
                    result = await self._execute_step(session, step)
                    execution_results.append(result)
                    
                    # Reflection after each step
                    await self._reflection_checkpoint(session, step, result)
            
            elif session.execution_mode == ExecutionMode.PARALLEL:
                # Execute multiple steps in parallel
                parallel_steps = next_steps[:self.max_concurrent_steps]
                tasks = [self._execute_step(session, step) for step in parallel_steps]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for step, result in zip(parallel_steps, results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Step {step.id} failed: {result}")
                        session.failed_steps.append(step.id)
                    else:
                        execution_results.append(result)
                        await self._reflection_checkpoint(session, step, result)
            
            elif session.execution_mode == ExecutionMode.ADAPTIVE:
                # Adaptive execution based on step characteristics
                await self._adaptive_execution(session, next_steps, execution_results)
            
            elif session.execution_mode == ExecutionMode.SMART_ROUTING:
                # Smart routing with optimal model selection
                await self._smart_routing_execution(session, next_steps, execution_results)
            
            # Check for plan adaptation needs
            if self.config["enable_adaptive_planning"]:
                await self._check_plan_adaptation(session, execution_results)
            
            session.update_activity()
        
        session.current_state = OrchestrationState.COMPLETED
        self.logger.info(f"Execution completed for session {session.session_id}")
    
    async def _decompose_task(self, query: str, context: Dict[str, Any]) -> TaskDecomposition:
        """Decompose the task using the task decomposer.
        
        Args:
            query: User query
            context: Additional context
            
        Returns:
            Task decomposition result
        """
        self.logger.debug(f"Decomposing task: {query[:100]}...")
        
        # Add context information to the decomposition
        enhanced_context = context.copy()
        enhanced_context.update({
            "enable_streaming": self.enable_streaming,
            "orchestration_mode": "intelligent_investigation"
        })
        
        return await asyncio.to_thread(
            self.task_decomposer.decompose, 
            query, 
            enhanced_context
        )
    
    async def _create_execution_plan(self, 
                                   task_decomposition: TaskDecomposition,
                                   investigation_plan: InvestigationPlan) -> ExecutionPlan:
        """Create execution plan combining task decomposition and investigation strategy.
        
        Args:
            task_decomposition: Result from task decomposer
            investigation_plan: Investigation plan from strategy manager
            
        Returns:
            Execution plan
        """
        self.logger.debug("Creating execution plan...")
        
        # Combine subtasks from decomposition with investigation steps
        combined_steps = []
        
        # Convert investigation steps to execution steps
        for inv_step in investigation_plan.steps:
            # Create a Subtask object from InvestigationStep
            subtask = Subtask(
                id=inv_step.step_id,
                description=inv_step.description,
                task_type=TaskType.GENERAL_QA,  # Default, can be refined
                priority=TaskPriority.MEDIUM,  # Convert from investigation priority
                estimated_complexity=min(inv_step.estimated_duration / 60.0, 1.0),  # Convert duration to complexity
                required_capabilities=["general_qa"],
                dependencies=inv_step.dependencies,
                preferred_models=inv_step.required_models if inv_step.required_models else [],
                context_needed=[],
                expected_output="; ".join(inv_step.expected_outputs) if inv_step.expected_outputs else ""
            )
            
            # Use capability checker to ensure tool-supporting model assignment
            capability_checker = get_capability_checker()
            default_model = "qwen2.5:7b-instruct-q4_K_M"
            
            # Determine if this step requires tools
            requires_tools = any(keyword in inv_step.description.lower() 
                               for keyword in ["file", "read", "write", "analyze", "execute", "run", "create"])
            
            if inv_step.required_models:
                # Validate that preferred models support tools if needed
                suitable_models = []
                from src.core.model_tool_support import test_model_tool_support
                
                for model in inv_step.required_models:
                    if requires_tools:
                        # Explicitly test if the model supports tools
                        if test_model_tool_support(model):
                            self.logger.info(f"Verified tool support for model {model}")
                            suitable_models.append(model)
                        else:
                            self.logger.warning(f"Model {model} doesn't support tools for step {inv_step.step_id}, finding alternative")
                            # Get an alternative that we know supports tools
                            alternative = capability_checker.get_best_model_for_task("coding", requires_tools=True)
                            if alternative:
                                self.logger.info(f"Using alternative model with verified tool support: {alternative}")
                                suitable_models.append(alternative)
                    else:
                        # Tools not required, can use as is
                        suitable_models.append(model)
                
                assigned_model = suitable_models[0] if suitable_models else default_model
            else:
                # Get best model for this type of task
                task_type = "general_qa"
                if "file" in inv_step.description.lower() or "read" in inv_step.description.lower():
                    task_type = "file_operations"
                elif "code" in inv_step.description.lower() or "analyze" in inv_step.description.lower():
                    task_type = "coding"
                
                # Use the capability checker to get a model with verified tool support
                from src.core.model_tool_support import test_model_tool_support
                
                # Get the best model for this task type with verified tool support if needed
                assigned_model = capability_checker.get_best_model_for_task(task_type, requires_tools)
                
                # Double-check tool support if required
                if requires_tools and assigned_model and not test_model_tool_support(assigned_model):
                    self.logger.warning(f"Selected model {assigned_model} failed tool support verification, using default")
                    # Try our hardcoded reliable models
                    for reliable_model in ["qwen2.5-coder:7b", "phi3:small", "llama3:8b"]:
                        if test_model_tool_support(reliable_model):
                            assigned_model = reliable_model
                            self.logger.info(f"Using reliable model with verified tool support: {assigned_model}")
                            break
                    else:
                        assigned_model = default_model
                
                # Fallback to default model if needed
                if not assigned_model:
                    assigned_model = default_model
            
            exec_step = ExecutionStep(
                id=inv_step.step_id,
                subtask=subtask,
                assigned_model=assigned_model,
                model_confidence=0.7,  # Default confidence
                context_input={
                    "investigation_strategy": inv_step.strategy.value,
                    "expected_outputs": inv_step.expected_outputs,
                    "validation_criteria": inv_step.validation_criteria
                }
            )
            combined_steps.append(exec_step)
        
        # Add steps from task decomposition
        for subtask in task_decomposition.subtasks:
            # Use capability checker for model assignment
            capability_checker = get_capability_checker()
            
            # Determine if subtask requires tools
            requires_tools = any(cap in ["file_operations", "system_operations", "code_execution"] 
                               for cap in subtask.required_capabilities)
            
            if subtask.preferred_models:
                # Validate preferred models support tools if needed
                suitable_models = []
                for model in subtask.preferred_models:
                    if not requires_tools or capability_checker.supports_tools(model):
                        suitable_models.append(model)
                    else:
                        self.logger.warning(f"Model {model} doesn't support tools for subtask {subtask.id}, finding alternative")
                        alternative = capability_checker.get_alternative_model(model, requires_tools=True)
                        if alternative:
                            suitable_models.append(alternative)
                
                assigned_model = suitable_models[0] if suitable_models else "qwen2.5:7b-instruct-q4_K_M"
            else:
                task_type_str = subtask.task_type.value if hasattr(subtask.task_type, 'value') else str(subtask.task_type)
                assigned_model = capability_checker.get_best_model_for_task(task_type_str, requires_tools) or "qwen2.5:7b-instruct-q4_K_M"
            
            exec_step = ExecutionStep(
                id=f"subtask_{subtask.id}",
                subtask=subtask,
                assigned_model=assigned_model,
                model_confidence=0.8,  # Higher confidence for decomposed tasks
                context_input={
                    "source": "task_decomposition",
                    "original_subtask_id": subtask.id
                }
            )
            combined_steps.append(exec_step)
        
        return ExecutionPlan(
            task_decomposition=task_decomposition,
            execution_steps=combined_steps,
            execution_order=[step.id for step in combined_steps],
            total_estimated_time=sum(step.subtask.estimated_complexity * 60 for step in combined_steps),
            model_transitions=[],  # Will be populated during execution
            metadata={
                "created_at": time.time(),
                "investigation_plan_id": investigation_plan.investigation_id,
                "combined_approach": True
            }
        )
    
    async def _execute_step(self, session: OrchestrationSession, 
                          step: ExecutionStep) -> Dict[str, Any]:
        """Execute a single step with the assigned model.
        
        Args:
            session: Orchestration session
            step: Execution step to run
            
        Returns:
            Execution result
        """
        start_time = time.time()
        
        self.logger.info(f"Executing step {step.id} with model {step.assigned_model}")
        
        try:
            # Get or create agent for the assigned model
            agent = await self._get_agent(step.assigned_model)
            
            # Prepare context for the step
            step_context = self._prepare_step_context(session, step)
            
            # Execute the step
            if self.enable_streaming:
                result = await self._execute_streaming_step(agent, step, step_context)
            else:
                result = await self._execute_non_streaming_step(agent, step, step_context)
            
            execution_time = time.time() - start_time
            
            # Record successful execution
            session.completed_steps.append(step.id)
            session.total_execution_time += execution_time
            
            # Record in context
            self.context_manager.record_execution(
                session.session_id,
                step.id,
                step.assigned_model,
                str(result)[:1000],  # Truncate long results
                execution_time,
                {"success": True, "step_metadata": step.metadata}
            )
            
            return {
                "step_id": step.id,
                "model": step.assigned_model,
                "result": result,
                "execution_time": execution_time,
                "success": True,
                "timestamp": time.time()
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Step {step.id} failed: {e}")
            
            session.failed_steps.append(step.id)
            
            # Record failed execution
            self.context_manager.record_execution(
                session.session_id,
                step.id,
                step.assigned_model,
                f"ERROR: {str(e)}",
                execution_time,
                {"success": False, "error": str(e)}
            )
            
            return {
                "step_id": step.id,
                "model": step.assigned_model,
                "result": None,
                "execution_time": execution_time,
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def _get_agent(self, model_name: str) -> UniversalAgent:
        """Get or create an agent for the specified model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Universal agent instance
        """
        # Check if we already have this agent cached
        if model_name in self.agent_cache:
            return self.agent_cache[model_name]
            
        self.logger.debug(f"Creating new agent for model {model_name}")
        
        # Check if the model exists and try to auto-pull if it doesn't
        model_available = model_exists(model_name)
        
        if not model_available:
            self.logger.info(f"Model {model_name} not found locally, attempting to pull...")
            try:
                # Try to pull the model
                model_available = ensure_model_available(model_name)
                if model_available:
                    self.logger.info(f"Successfully pulled model {model_name}")
                else:
                    self.logger.warning(f"Failed to pull model {model_name}, will try fallback models")
                    # Get a fallback model that exists
                    capability_checker = get_capability_checker()
                    fallback_model = capability_checker.get_default_model()
                    
                    # Make sure the fallback model exists
                    if not model_exists(fallback_model):
                        # As a last resort, get the first available model
                        from src.core.model_discovery import get_available_models
                        available_models = get_available_models()
                        if available_models:
                            fallback_model = available_models[0]
                        else:
                            self.logger.error("No models are available in the system")
                            raise RuntimeError("No models are available in the system")
                    
                    self.logger.info(f"Using fallback model {fallback_model} instead of {model_name}")
                    model_name = fallback_model
            except Exception as e:
                self.logger.error(f"Error pulling model {model_name}: {str(e)}")
                # Get a fallback model that exists
                capability_checker = get_capability_checker()
                fallback_model = capability_checker.get_default_model()
                
                # Make sure the fallback model exists
                if not model_exists(fallback_model):
                    # As a last resort, get the first available model
                    from src.core.model_discovery import get_available_models
                    available_models = get_available_models()
                    if available_models:
                        fallback_model = available_models[0]
                    else:
                        self.logger.error("No models are available in the system")
                        raise RuntimeError("No models are available in the system")
                
                self.logger.info(f"Using fallback model {fallback_model} instead of {model_name}")
                model_name = fallback_model
                
        # Create agent with streaming configuration
        agent = await asyncio.to_thread(
            get_agent_instance,
            model_name,
            streaming=self.enable_streaming
        )
        
        self.agent_cache[model_name] = agent
        return agent
    
    def _prepare_step_context(self, session: OrchestrationSession, 
                            step: ExecutionStep) -> Dict[str, Any]:
        """Prepare context information for step execution.
        
        Args:
            session: Orchestration session
            step: Execution step
            
        Returns:
            Context dictionary for the step
        """
        # Get relevant context from context manager
        context = self.context_manager.get_context(session.session_id)
        if context:
            relevant_context = context.get_relevant_context()
        else:
            relevant_context = {}
        
        # Add step-specific information
        step_context = {
            "session_id": session.session_id,
            "original_query": session.original_query,
            "current_step": step.id,
            "step_description": step.subtask.description,
            "completed_steps": session.completed_steps,
            "step_metadata": step.metadata,
            **relevant_context
        }
        
        return step_context
    
    async def _execute_streaming_step(self, agent: UniversalAgent, 
                                    step: ExecutionStep, 
                                    context: Dict[str, Any]) -> Any:
        """Execute a step with streaming output.
        
        Args:
            agent: Universal agent instance
            step: Execution step
            context: Step context
            
        Returns:
            Step execution result
        """
        # Prepare the prompt for the step
        prompt = self._build_step_prompt(step, context)
        
        # Execute with streaming
        result = ""
        
        # Define on_token callback
        def on_token(token: str):
            nonlocal result
            result += token
            # Could add real-time processing here
        
        # Call stream with the on_token callback
        await asyncio.to_thread(agent.stream, prompt, on_token)
        
        return result
    
    async def _execute_non_streaming_step(self, agent: UniversalAgent,
                                        step: ExecutionStep,
                                        context: Dict[str, Any]) -> Any:
        """Execute a step without streaming.
        
        Args:
            agent: Universal agent instance
            step: Execution step
            context: Step context
            
        Returns:
            Step execution result
        """
        prompt = self._build_step_prompt(step, context)
        
        # Execute without streaming
        result = await asyncio.to_thread(agent.run, prompt)
        return result
    
    def _build_step_prompt(self, step: ExecutionStep, context: Dict[str, Any]) -> str:
        """Build the prompt for step execution.
        
        Args:
            step: Execution step
            context: Step context
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            f"Task: {step.subtask.description}",
            f"Original Query: {context.get('original_query', 'N/A')}",
            ""
        ]
        
        # Add context information
        if context.get("completed_steps"):
            prompt_parts.append("Previously completed steps:")
            for completed_step in context["completed_steps"][-3:]:  # Last 3 steps
                prompt_parts.append(f"- {completed_step}")
            prompt_parts.append("")
        
        # Add relevant context entries
        if context.get("context_entries"):
            prompt_parts.append("Relevant context:")
            for key, value in list(context["context_entries"].items())[:5]:  # Top 5 entries
                prompt_parts.append(f"- {key}: {str(value)[:200]}...")
            prompt_parts.append("")
        
        # Add step-specific metadata
        if step.metadata:
            if step.metadata.get("expected_outputs"):
                prompt_parts.append("Expected outputs:")
                for output in step.metadata["expected_outputs"]:
                    prompt_parts.append(f"- {output}")
                prompt_parts.append("")
            
            if step.metadata.get("validation_criteria"):
                prompt_parts.append("Validation criteria:")
                for criterion in step.metadata["validation_criteria"]:
                    prompt_parts.append(f"- {criterion}")
                prompt_parts.append("")
        
        prompt_parts.append("Please execute this task thoroughly and provide detailed results.")
        
        return "\n".join(prompt_parts)
    
    async def _reflection_checkpoint(self, session: OrchestrationSession,
                                   step: ExecutionStep, 
                                   result: Dict[str, Any]) -> None:
        """Execute a reflection checkpoint after step completion.
        
        Args:
            session: Orchestration session
            step: Completed execution step
            result: Step execution result
        """
        session.current_state = OrchestrationState.REFLECTING
        
        # Perform reflection evaluation
        reflection_result = self.reflection_system.evaluate_step(
            session.session_id,
            step.id,
            step.assigned_model,
            result.get("result"),
            result.get("execution_time", 0),
            ReflectionTrigger.STEP_COMPLETION
        )
        
        # Check if model swap is recommended
        if reflection_result.should_swap and reflection_result.recommended_model:
            await self._handle_model_swap(session, step, reflection_result)
        
        session.current_state = OrchestrationState.EXECUTING
    
    async def _handle_model_swap(self, session: OrchestrationSession,
                               step: ExecutionStep,
                               reflection_result: ReflectionResult) -> None:
        """Handle model swap recommendation.
        
        Args:
            session: Orchestration session
            step: Current execution step
            reflection_result: Reflection evaluation result
        """
        if session.model_switches >= self.config["max_model_switches"]:
            self.logger.warning(f"Max model switches reached for session {session.session_id}")
            return
        
        old_model = step.assigned_model
        new_model = reflection_result.recommended_model
        
        # Validate that the new model supports required capabilities
        capability_checker = get_capability_checker()
        
        # Check if new model supports tools if current step requires them
        requires_tools = self._step_requires_tools(step)
        
        if requires_tools:
            # Use our direct tool support test to verify the model
            from src.core.model_tool_support import test_model_tool_support, get_verified_tool_supporting_models
            
            # Check if we have cached verification results for this model
            if not test_model_tool_support(new_model):
                self.logger.warning(f"Recommended model {new_model} doesn't support tools, finding alternative")
                
                # Get cached list of verified tool-supporting models
                verified_models = get_verified_tool_supporting_models()
                
                # Try these verified models first if available
                if verified_models:
                    for verified_model in verified_models:
                        if verified_model != new_model:  # Don't try the same model we just tested
                            self.logger.info(f"Trying verified tool-supporting model: {verified_model}")
                            new_model = verified_model
                            break
                else:
                    # Fallback to reliable models we've hardcoded
                    for reliable_model in ["qwen2.5-coder:7b", "phi3:small", "llama3:8b", "qwen2.5:7b-instruct-q4_K_M"]:
                        if test_model_tool_support(reliable_model):
                            new_model = reliable_model
                            self.logger.info(f"Using reliable model with verified tool support: {new_model}")
                            break
                    else:
                        # If no reliable models work, try the capability checker as fallback
                        alternative = capability_checker.get_alternative_model(new_model, requires_tools=True)
                        if alternative and alternative != new_model and test_model_tool_support(alternative):
                            new_model = alternative
                            self.logger.info(f"Using capability checker recommended model: {new_model}")
                        else:
                            self.logger.warning(f"Could not find a suitable tool-supporting model, using qwen2.5-coder:7b as last resort")
                            new_model = "qwen2.5-coder:7b"
                            
                self.logger.info(f"Using tool-supporting alternative: {new_model}")
        
        self.logger.info(f"Model swap recommended: {old_model} -> {new_model} "
                        f"(Reason: {reflection_result.reasoning})")
        
        # Update remaining steps to use new model
        self._update_execution_plan_model(session.execution_plan, old_model, new_model)
        
        # Prepare transition context
        transition_context = self.context_manager.prepare_model_transition_context(
            session.session_id, old_model, new_model
        )
        
        # Record the swap
        session.model_switches += 1
        self.context_manager.add_context_data(
            session.session_id,
            f"model_swap_{session.model_switches}",
            {
                "from_model": old_model,
                "to_model": new_model,
                "reason": reflection_result.reasoning,
                "timestamp": time.time()
            },
            "orchestrator"
        )
    
    def _update_execution_plan_model(self, execution_plan: ExecutionPlan,
                                   old_model: str, new_model: str) -> None:
        """Update execution plan to use new model for remaining steps.
        
        Args:
            execution_plan: Execution plan to update
            old_model: Model to replace
            new_model: New model to use
        """
        for step in execution_plan.execution_steps:
            if step.assigned_model == old_model and step.id not in self.active_sessions:
                step.assigned_model = new_model
                self.logger.debug(f"Updated step {step.id} to use model {new_model}")
    
    def _get_next_executable_steps(self, session: OrchestrationSession) -> List[ExecutionStep]:
        """Get steps that can be executed next.
        
        Args:
            session: Orchestration session
            
        Returns:
            List of executable steps
        """
        if not session.execution_plan:
            return []
        
        executable_steps = []
        
        for step in session.execution_plan.execution_steps:
            # Skip completed or failed steps
            if step.id in session.completed_steps or step.id in session.failed_steps:
                continue
            
            # Check if all dependencies are completed
            if all(dep in session.completed_steps for dep in step.subtask.dependencies):
                executable_steps.append(step)
        
        # Sort by priority (lower number = higher priority)
        executable_steps.sort(key=lambda x: x.subtask.priority.value if hasattr(x.subtask.priority, 'value') else 2)
        
        return executable_steps
    
    def _is_execution_complete(self, session: OrchestrationSession) -> bool:
        """Check if execution is complete.
        
        Args:
            session: Orchestration session
            
        Returns:
            True if execution is complete
        """
        if not session.execution_plan:
            return True
        
        total_steps = len(session.execution_plan.execution_steps)
        completed_steps = len(session.completed_steps)
        failed_steps = len(session.failed_steps)
        
        # Execution is complete if all steps are either completed or failed
        return (completed_steps + failed_steps) >= total_steps
    
    async def _adaptive_execution(self, session: OrchestrationSession,
                                next_steps: List[ExecutionStep],
                                execution_results: List[Dict[str, Any]]) -> None:
        """Execute steps adaptively based on characteristics and results.
        
        Args:
            session: Orchestration session
            next_steps: Available steps to execute
            execution_results: Previous execution results
        """
        # Analyze recent performance to decide execution strategy
        recent_failures = sum(1 for r in execution_results[-5:] if not r.get("success", True))
        
        if recent_failures > 2:
            # Use sequential execution if recent failures
            for step in next_steps[:1]:
                result = await self._execute_step(session, step)
                execution_results.append(result)
                await self._reflection_checkpoint(session, step, result)
        else:
            # Use parallel execution for independent steps
            independent_steps = [step for step in next_steps if not step.subtask.dependencies][:2]
            if independent_steps:
                tasks = [self._execute_step(session, step) for step in independent_steps]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for step, result in zip(independent_steps, results):
                    if not isinstance(result, Exception):
                        execution_results.append(result)
                        await self._reflection_checkpoint(session, step, result)
    
    async def _smart_routing_execution(self, session: OrchestrationSession,
                                     next_steps: List[ExecutionStep],
                                     execution_results: List[Dict[str, Any]]) -> None:
        """Execute steps with smart model routing based on task characteristics.
        
        Args:
            session: Orchestration session
            next_steps: Available steps to execute
            execution_results: Previous execution results
        """
        # Group steps by optimal model
        model_groups = {}
        for step in next_steps:
            model = step.assigned_model
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(step)
        
        # Execute steps grouped by model to minimize switching overhead
        for model, steps in model_groups.items():
            for step in steps[:1]:  # Execute one step per model per iteration
                result = await self._execute_step(session, step)
                execution_results.append(result)
                await self._reflection_checkpoint(session, step, result)
    
    async def _check_plan_adaptation(self, session: OrchestrationSession,
                                   execution_results: List[Dict[str, Any]]) -> None:
        """Check if plan adaptation is needed based on execution results.
        
        Args:
            session: Orchestration session
            execution_results: Execution results so far
        """
        # Check if adaptation is needed
        recent_results = execution_results[-5:]  # Last 5 results
        failure_rate = sum(1 for r in recent_results if not r.get("success", True)) / len(recent_results) if recent_results else 0
        
        if failure_rate > 0.6:  # More than 60% failure rate
            session.current_state = OrchestrationState.ADAPTING
            
            self.logger.info(f"High failure rate detected, adapting plan for session {session.session_id}")
            
            # Adapt investigation plan
            if session.investigation_plan:
                adapted_plan = self.strategy_manager.adapt_investigation_plan(
                    session.investigation_plan, execution_results
                )
                session.investigation_plan = adapted_plan
                
                # Update execution plan accordingly
                session.execution_plan = await self._create_execution_plan(
                    session.task_decomposition,  # Use existing task decomposition
                    adapted_plan
                )
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of an investigation session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session status information
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return None
        
        # Get reflection insights
        reflection_insights = self.reflection_system.get_reflection_insights(session_id)
        
        # Get context statistics
        context_stats = self.context_manager.get_session_statistics(session_id)
        
        return {
            "session": session.to_dict(),
            "reflection_insights": reflection_insights,
            "context_statistics": context_stats,
            "progress": {
                "completed_steps": len(session.completed_steps),
                "failed_steps": len(session.failed_steps),
                "total_steps": len(session.execution_plan.execution_steps) if session.execution_plan else 0,
                "completion_percentage": (len(session.completed_steps) / len(session.execution_plan.execution_steps) * 100) if session.execution_plan and session.execution_plan.execution_steps else 0
            }
        }
    
    async def pause_session(self, session_id: str) -> bool:
        """Pause an active investigation session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if paused successfully
        """
        session = self.active_sessions.get(session_id)
        if session and session.current_state != OrchestrationState.COMPLETED:
            session.current_state = OrchestrationState.PAUSED
            self.logger.info(f"Paused session {session_id}")
            return True
        return False
    
    async def resume_session(self, session_id: str) -> bool:
        """Resume a paused investigation session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if resumed successfully
        """
        session = self.active_sessions.get(session_id)
        if session and session.current_state == OrchestrationState.PAUSED:
            session.current_state = OrchestrationState.EXECUTING
            session.update_activity()
            self.logger.info(f"Resumed session {session_id}")
            
            # Continue execution
            await self._execution_phase(session)
            return True
        return False
    
    def _step_requires_tools(self, step: ExecutionStep) -> bool:
        """Determine if an execution step requires tool support."""
        # Check if the subtask has tool-requiring capabilities
        tool_requiring_capabilities = [
            "file_operations", "system_operations", "code_execution", 
            "web_operations", "project_operations", "repository", "tool_usage",
            "data_analysis", "automation"
        ]
        
        # Check for explicit tool requirement flag if it exists
        if hasattr(step, 'requires_tools') and step.requires_tools:
            return True
            
        # Check subtask if available
        if hasattr(step, 'subtask') and step.subtask:
            # Check explicit flag in subtask if it exists
            if hasattr(step.subtask, 'requires_tools') and step.subtask.requires_tools:
                return True
                
            # Check required capabilities
            for capability in step.subtask.required_capabilities:
                if any(tool_cap in capability.lower() for tool_cap in tool_requiring_capabilities):
                    return True
            
            # Check task type - these types always need tools
            if step.subtask.task_type in ['CODING', 'SYSTEM_OPERATION', 'REPOSITORY_MANAGEMENT', 'DATA_PROCESSING']:
                return True
                
            # Check if the subtask has tools defined or required
            if hasattr(step.subtask, 'available_tools') and step.subtask.available_tools:
                return True
        
        # Check step description for tool-indicating keywords
        if hasattr(step, 'description') and step.description:
            # Expanded list of tool-related keywords
            tool_keywords = [
                "file", "code", "execute", "run", "install", "create", "modify", "analyze",
                "directory", "folder", "repository", "git", "clone", "commit", "tool", "command",
                "terminal", "script", "compile", "build", "deploy", "api", "database", "query",
                "search", "fetch", "download", "upload", "write"
            ]
            step_text = step.description.lower()
            if any(keyword in step_text for keyword in tool_keywords):
                return True
                
        # Check if step properties indicate tool requirements
        if hasattr(step, 'properties') and step.properties:
            if step.properties.get('tools_required', False):
                return True
            if step.properties.get('file_operations', False):
                return True
            if step.properties.get('system_access', False):
                return True
                
        return False
    
    def cleanup_completed_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up completed sessions.
        
        Args:
            max_age_hours: Maximum age for completed sessions
            
        Returns:
            Number of sessions cleaned up
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        sessions_to_remove = []
        for session_id, session in self.active_sessions.items():
            if (session.current_state == OrchestrationState.COMPLETED and
                current_time - session.last_activity > max_age_seconds):
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
            self.logger.debug(f"Cleaned up completed session {session_id}")
        
        return len(sessions_to_remove)


# Global orchestrator instance
_global_orchestrator: Optional[IntelligentOrchestrator] = None


def get_orchestrator(enable_streaming: bool = True,
                    max_concurrent_steps: int = 3) -> IntelligentOrchestrator:
    """Get the global orchestrator instance."""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = IntelligentOrchestrator(
            enable_streaming=enable_streaming,
            max_concurrent_steps=max_concurrent_steps
        )
    
    return _global_orchestrator


__all__ = [
    "IntelligentOrchestrator", "OrchestrationSession", "OrchestrationState", 
    "ExecutionMode", "get_orchestrator"
]
