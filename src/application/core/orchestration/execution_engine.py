"""Execution engine for orchestration steps."""

import asyncio
from typing import Any, Dict, List, Optional

from src.agents.universal.agent import UniversalAgent
from src.core.context_manager import ContextManager
from src.core.execution_planner import ExecutionPlan, ExecutionStep
from src.utils.enhanced_logging import get_logger

from .session_manager import OrchestrationSession, OrchestrationState


class ExecutionEngine:
    """Handles execution of orchestration steps."""

    def __init__(self, 
                 context_manager: ContextManager,
                 enable_streaming: bool = True,
                 max_concurrent_steps: int = 3):
        """Initialize execution engine.
        
        Args:
            context_manager: Context manager instance
            enable_streaming: Whether to enable streaming mode
            max_concurrent_steps: Maximum concurrent steps in parallel execution
        """
        self.context_manager = context_manager
        self.enable_streaming = enable_streaming
        self.max_concurrent_steps = max_concurrent_steps
        self.logger = get_logger()

    async def execute_step(self, 
                          session: OrchestrationSession,
                          step: ExecutionStep,
                          agent: UniversalAgent) -> Dict[str, Any]:
        """Execute a single orchestration step.
        
        Args:
            session: Orchestration session
            step: Execution step to run
            agent: Agent to execute the step
            
        Returns:
            Execution result
        """
        session.update_activity()
        session.current_step = step.step_id

        self.logger.info(f"Executing step '{step.step_id}' in session {session.session_id}")

        # Prepare context for this step
        step_context = self._prepare_step_context(session, step)
        
        # Build the prompt for this step
        prompt = self._build_step_prompt(step, step_context)
        
        # Execute based on streaming preference
        if self.enable_streaming:
            result = await self._execute_streaming_step(agent, prompt, step)
        else:
            result = await self._execute_non_streaming_step(agent, prompt, step)

        # Store result in context
        self.context_manager.add_context_data(
            session.session_id,
            f"step_{step.step_id}_result",
            result,
            "execution_engine"
        )

        # Update session tracking
        if result.get("success", False):
            session.completed_steps.append(step.step_id)
        else:
            session.failed_steps.append(step.step_id)

        return result

    async def execute_parallel_steps(self,
                                   session: OrchestrationSession,
                                   steps: List[ExecutionStep],
                                   agents: Dict[str, UniversalAgent]) -> List[Dict[str, Any]]:
        """Execute multiple steps in parallel.
        
        Args:
            session: Orchestration session
            steps: List of steps to execute
            agents: Dictionary mapping step IDs to agents
            
        Returns:
            List of execution results
        """
        # Limit concurrent execution
        semaphore = asyncio.Semaphore(self.max_concurrent_steps)
        
        async def execute_with_semaphore(step: ExecutionStep) -> Dict[str, Any]:
            async with semaphore:
                agent = agents.get(step.step_id)
                if not agent:
                    return {"success": False, "error": f"No agent available for step {step.step_id}"}
                return await self.execute_step(session, step, agent)

        # Execute all steps concurrently
        results = await asyncio.gather(
            *[execute_with_semaphore(step) for step in steps],
            return_exceptions=True
        )

        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append({
                    "success": False,
                    "error": str(result),
                    "step_id": steps[i].step_id
                })
            else:
                final_results.append(result)

        return final_results

    def _prepare_step_context(self, 
                             session: OrchestrationSession,
                             step: ExecutionStep) -> Dict[str, Any]:
        """Prepare context data for step execution.
        
        Args:
            session: Orchestration session
            step: Execution step
            
        Returns:
            Context dictionary
        """
        # Get session context
        session_context = self.context_manager.get_context(session.session_id) or {}
        
        # Build step-specific context
        step_context = {
            "original_query": session.original_query,
            "session_id": session.session_id,
            "step_id": step.step_id,
            "step_description": step.description,
            "execution_mode": session.execution_mode.value,
            "completed_steps": session.completed_steps,
            "failed_steps": session.failed_steps,
            "session_context": session_context
        }

        # Add results from prerequisite steps
        if step.prerequisites:
            prerequisite_results = {}
            for prereq_id in step.prerequisites:
                result_key = f"step_{prereq_id}_result"
                if result_key in session_context:
                    prerequisite_results[prereq_id] = session_context[result_key]
            step_context["prerequisite_results"] = prerequisite_results

        return step_context

    async def _execute_streaming_step(self,
                                     agent: UniversalAgent,
                                     prompt: str,
                                     step: ExecutionStep) -> Dict[str, Any]:
        """Execute step with streaming response.
        
        Args:
            agent: Agent to execute with
            prompt: Prompt to send
            step: Execution step
            
        Returns:
            Execution result
        """
        try:
            response_parts = []
            
            async for chunk in agent.stream_query(prompt):
                if chunk and 'content' in chunk:
                    response_parts.append(chunk['content'])
            
            full_response = ''.join(response_parts)
            
            return {
                "success": True,
                "response": full_response,
                "step_id": step.step_id,
                "agent_model": agent.model_id,
                "execution_type": "streaming"
            }
            
        except Exception as e:
            self.logger.error(f"Streaming step execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "step_id": step.step_id,
                "execution_type": "streaming"
            }

    async def _execute_non_streaming_step(self,
                                         agent: UniversalAgent,
                                         prompt: str,
                                         step: ExecutionStep) -> Dict[str, Any]:
        """Execute step with non-streaming response.
        
        Args:
            agent: Agent to execute with
            prompt: Prompt to send
            step: Execution step
            
        Returns:
            Execution result
        """
        try:
            response = await agent.query(prompt)
            
            return {
                "success": True,
                "response": response,
                "step_id": step.step_id,
                "agent_model": agent.model_id,
                "execution_type": "non_streaming"
            }
            
        except Exception as e:
            self.logger.error(f"Non-streaming step execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "step_id": step.step_id,
                "execution_type": "non_streaming"
            }

    def _build_step_prompt(self, step: ExecutionStep, context: Dict[str, Any]) -> str:
        """Build the prompt for a specific step.
        
        Args:
            step: Execution step
            context: Step context
            
        Returns:
            Formatted prompt
        """
        prompt_parts = []
        
        # Add step description
        prompt_parts.append(f"**Task**: {step.description}")
        
        # Add original query context
        if context.get("original_query"):
            prompt_parts.append(f"**Original Query**: {context['original_query']}")
        
        # Add prerequisite results if available
        if context.get("prerequisite_results"):
            prompt_parts.append("**Previous Step Results**:")
            for prereq_id, result in context["prerequisite_results"].items():
                if isinstance(result, dict) and result.get("response"):
                    prompt_parts.append(f"- Step {prereq_id}: {result['response'][:200]}...")
        
        # Add step-specific instructions
        if step.instructions:
            prompt_parts.append(f"**Instructions**: {step.instructions}")
        
        # Add expected output format
        if step.expected_output:
            prompt_parts.append(f"**Expected Output**: {step.expected_output}")
        
        return "\n\n".join(prompt_parts)

    def get_next_executable_steps(self, session: OrchestrationSession) -> List[ExecutionStep]:
        """Get the next steps that can be executed.
        
        Args:
            session: Orchestration session
            
        Returns:
            List of executable steps
        """
        if not session.execution_plan:
            return []
        
        executable_steps = []
        
        for step in session.execution_plan.steps:
            # Skip already completed or failed steps
            if step.step_id in session.completed_steps or step.step_id in session.failed_steps:
                continue
            
            # Check if all prerequisites are completed
            if step.prerequisites:
                prerequisites_met = all(
                    prereq_id in session.completed_steps 
                    for prereq_id in step.prerequisites
                )
                if not prerequisites_met:
                    continue
            
            executable_steps.append(step)
        
        return executable_steps

    def is_execution_complete(self, session: OrchestrationSession) -> bool:
        """Check if execution is complete.
        
        Args:
            session: Orchestration session
            
        Returns:
            True if execution is complete
        """
        if not session.execution_plan:
            return True
        
        total_steps = len(session.execution_plan.steps)
        completed_or_failed = len(session.completed_steps) + len(session.failed_steps)
        
        return completed_or_failed >= total_steps
