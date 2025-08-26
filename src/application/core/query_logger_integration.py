"""Query Logger Integration - Wrapper for integrating logging into existing systems."""
from __future__ import annotations

import functools
import time
from typing import Any, Callable

from src.core.query_logger import get_query_logger
from src.utils.enhanced_logging import get_logger


def log_query_execution(execution_mode: str = "unknown"):
    """Decorator to automatically log query execution."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract query from arguments
            query = None
            if args:
                query = str(args[0])  # First argument is usually the query
            elif 'query' in kwargs:
                query = str(kwargs['query'])

            if not query:
                # No query found, execute without logging
                return func(*args, **kwargs)

            query_logger = get_query_logger()
            get_logger()

            # Start logging session
            query_logger.start_query_session(query, execution_mode)

            try:
                # Log function start
                query_logger.start_execution_step(
                    "function_execution",
                    func.__name__,
                    f"Executing {func.__name__} with query"
                )

                # Execute function
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Log completion
                query_logger.end_execution_step(
                    output_generated=str(result)[:1000] if result else None,
                    execution_time=execution_time
                )

                # End session successfully
                query_logger.end_query_session(
                    final_answer=str(result)[:1000] if result else None,
                    success=True
                )

                return result

            except Exception as e:
                # Log error
                error_msg = str(e)
                query_logger.end_query_session(
                    final_answer=None,
                    success=False,
                    error_messages=[error_msg]
                )
                raise

        return wrapper
    return decorator


class LoggingCollaborativeWrapper:
    """Wrapper for collaborative system with integrated logging."""

    def __init__(self, collaborative_system):
        """Initialize wrapper with existing collaborative system."""
        self.system = collaborative_system
        self.query_logger = get_query_logger()
        self.logger = get_logger()

    def collaborative_execution_with_logging(self, query: str, working_directory: str = ".", max_steps: int = None) -> dict[str, Any]:
        """Execute collaborative query with comprehensive logging."""
        # Start query logging session
        self.query_logger.start_query_session(query, "collaborative")

        # Log initial context
        self.query_logger.log_context_usage(
            source="collaborative_init",
            content=f"Query: {query}\nWorking directory: {working_directory}",
            metadata={"max_steps": max_steps or self.system.max_iterations}
        )

        try:
            # Start main execution step
            self.query_logger.start_execution_step(
                "collaborative_execution",
                getattr(self.system.main_agent, '_model_id', 'unknown'),
                f"Collaborative execution with {max_steps or self.system.max_iterations} max steps"
            )

            # Patch methods to add logging
            original_get_recommendations = self.system._get_interceptor_recommendations
            original_execute_command = self.system._execute_command
            original_switch_agent = self.system._switch_main_agent

            def logged_get_recommendations(*args, **kwargs):
                time.time()
                result = original_get_recommendations(*args, **kwargs)

                # Log interceptor recommendations
                self.query_logger.log_context_usage(
                    source="interceptor_recommendations",
                    content=str(result),
                    metadata={"recommendation_count": len(result) if isinstance(result, list) else 1}
                )
                return result

            def logged_execute_command(command_name: str, args: dict[str, Any], *other_args, **other_kwargs):
                tool_start = time.time()
                try:
                    result = original_execute_command(command_name, args, *other_args, **other_kwargs)
                    tool_time = time.time() - tool_start

                    # Log tool execution
                    self.query_logger.log_tool_execution(
                        tool_name=command_name,
                        args=args,
                        execution_time=tool_time,
                        success=True,
                        output=str(result)
                    )
                    return result
                except Exception as e:
                    tool_time = time.time() - tool_start
                    self.query_logger.log_tool_execution(
                        tool_name=command_name,
                        args=args,
                        execution_time=tool_time,
                        success=False,
                        output=str(e)
                    )
                    raise

            def logged_switch_agent(new_agent_id: str, *other_args, **other_kwargs):
                old_agent = getattr(self.system.main_agent, '_model_id', 'unknown')

                # Get context before switch
                context = self.system._create_context_summary(
                    getattr(self.system, 'discovered_files', set()),
                    getattr(self.system, 'executed_commands', []),
                    getattr(self.system, 'current_step', 0)
                )

                result = original_switch_agent(new_agent_id, *other_args, **other_kwargs)

                # Log agent switch
                self.query_logger.log_agent_switch(
                    from_agent=old_agent,
                    to_agent=new_agent_id,
                    reason="Collaborative system agent switch",
                    context_preserved=context,
                    success=result
                )
                return result

            # Apply patches
            self.system._get_interceptor_recommendations = logged_get_recommendations
            self.system._execute_command = logged_execute_command
            self.system._switch_main_agent = logged_switch_agent

            try:
                # Execute the collaborative process
                result = self.system.collaborative_execution(query, working_directory, max_steps)

                # Log successful completion
                self.query_logger.end_execution_step(
                    output_generated=str(result).get('final_answer', '')[:1000] if isinstance(result, dict) else str(result)[:1000]
                )

                self.query_logger.end_query_session(
                    final_answer=str(result).get('final_answer', '') if isinstance(result, dict) else str(result),
                    success=result.get('success', True) if isinstance(result, dict) else True
                )

                return result

            finally:
                # Restore original methods
                self.system._get_interceptor_recommendations = original_get_recommendations
                self.system._execute_command = original_execute_command
                self.system._switch_main_agent = original_switch_agent

        except Exception as e:
            # Log error
            self.query_logger.end_query_session(
                final_answer=None,
                success=False,
                error_messages=[str(e)]
            )
            raise


class LoggingUniversalAgentWrapper:
    """Wrapper for Universal Multi-Agent with integrated logging."""

    def __init__(self, universal_agent):
        """Initialize wrapper with existing universal agent."""
        self.agent = universal_agent
        self.query_logger = get_query_logger()
        self.logger = get_logger()

    def process_request_with_logging(self, prompt: str, context: dict[str, Any] = None) -> str:
        """Process request with comprehensive logging."""
        context = context or {}

        # Start query logging session
        self.query_logger.start_query_session(prompt, "universal_multi_agent")

        try:
            # Log initial context
            self.query_logger.log_context_usage(
                source="user_prompt",
                content=prompt,
                metadata=context
            )

            # Start execution step
            self.query_logger.start_execution_step(
                "universal_agent_processing",
                getattr(self.agent, '_current_model_id', 'universal'),
                "Universal Multi-Agent processing with dynamic model selection"
            )

            # Analyze task type
            task_type = self.agent._analyze_task_requirements(prompt)
            optimal_model = self.agent._select_optimal_model(task_type, context)

            # Log task analysis
            self.query_logger.log_context_usage(
                source="task_analysis",
                content=f"Task type: {task_type.value}, Optimal model: {optimal_model}",
                metadata={"task_type": task_type.value, "optimal_model": optimal_model}
            )

            # Track model switches
            if self.agent._current_model_id != optimal_model:
                old_model = self.agent._current_model_id

                # Execute model switch
                start_time = time.time()
                result = self.agent.process_request(prompt, context)
                execution_time = time.time() - start_time

                # Log the switch that occurred during processing
                self.query_logger.log_agent_switch(
                    from_agent=old_model,
                    to_agent=self.agent._current_model_id,
                    reason=f"Optimal for {task_type.value} task",
                    context_preserved=f"Task analysis and context for {task_type.value}",
                    success=True
                )
            else:
                # No switch needed
                start_time = time.time()
                result = self.agent.process_request(prompt, context)
                execution_time = time.time() - start_time

            # Log final result
            self.query_logger.end_execution_step(
                output_generated=result[:1000] if result else None,
                execution_time=execution_time
            )

            self.query_logger.end_query_session(
                final_answer=result,
                success=True
            )

            return result

        except Exception as e:
            # Log error
            self.query_logger.end_query_session(
                final_answer=None,
                success=False,
                error_messages=[str(e)]
            )
            raise


def patch_system_with_logging():
    """Patch the existing system components with logging capabilities."""
    logger = get_logger()

    try:
        # Patch collaborative system.  The refactored collaborative module exports
        # ``CollaborativeAgentSystem`` rather than ``CollaborativeSystem``.  We
        # attempt to import the modern name and fall back gracefully.
        try:
            from src.core.collaborative_system import CollaborativeAgentSystem as _CollaborativeSystem
        except ImportError:
            from src.core.collaborative_system import CollaborativeAgentSystem as _CollaborativeSystem  # pragma: no cover

        if not hasattr(_CollaborativeSystem, '_original_collaborative_execution'):
            _CollaborativeSystem._original_collaborative_execution = _CollaborativeSystem.collaborative_execution

            def logged_collaborative_execution(self, query: str, working_directory: str = ".", max_steps: int = None):
                wrapper = LoggingCollaborativeWrapper(self)
                return wrapper.collaborative_execution_with_logging(query, working_directory, max_steps)

            _CollaborativeSystem.collaborative_execution = logged_collaborative_execution
            logger.info("✅ Patched CollaborativeAgentSystem with logging")

        # Patch Universal Multi-Agent
        from src.core.universal_multi_agent import UniversalMultiAgent

        if not hasattr(UniversalMultiAgent, '_original_process_request'):
            UniversalMultiAgent._original_process_request = UniversalMultiAgent.process_request

            def logged_process_request(self, prompt: str, context: dict[str, Any] = None):
                wrapper = LoggingUniversalAgentWrapper(self)
                return wrapper.process_request_with_logging(prompt, context)

            UniversalMultiAgent.process_request = logged_process_request
            logger.info("✅ Patched UniversalMultiAgent with logging")

        return True

    except Exception as e:
        logger.error(f"❌ Failed to patch system with logging: {e}")
        return False


__all__ = [
    "log_query_execution", "LoggingCollaborativeWrapper", "LoggingUniversalAgentWrapper",
    "patch_system_with_logging"
]
