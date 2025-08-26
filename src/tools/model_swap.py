"""Model swap tool - Allows agents to request switching to a different model."""
from __future__ import annotations

from src.utils.enhanced_logging import get_logger

try:
    from langchain.tools import StructuredTool

    from src.tools.registry import register_tool
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False
    from src.tools.registry import register_tool


def request_model_swap(
    reason: str,
    preferred_model: str | None = None,
    task_type: str | None = None
) -> str:
    """Request switching to a different model that might be better suited for the current task.

    Args:
        reason: Explanation of why a model swap is needed
        preferred_model: Specific model to switch to (optional)
        task_type: Type of task (coding, analysis, creative, etc.) to help auto-select

    Returns:
        Status message about the swap request
    """
    logger = get_logger()

    logger.info(f"Model swap requested: {reason}")
    if preferred_model:
        logger.info(f"Preferred model: {preferred_model}")
    if task_type:
        logger.info(f"Task type: {task_type}")

    # This will be handled by the collaborative system
    swap_request = {
        'action': 'model_swap_request',
        'reason': reason,
        'preferred_model': preferred_model,
        'task_type': task_type,
        'timestamp': __import__('time').time()
    }

    # Store the request in a way that the collaborative system can pick it up
    import json
    request_str = json.dumps(swap_request)

    return f"""🔄 MODEL SWAP REQUEST INITIATED

Reason: {reason}
Preferred model: {preferred_model or 'Auto-select based on task'}
Task type: {task_type or 'General'}

The system will evaluate this request and potentially switch to a more suitable model.

SWAP_REQUEST_DATA: {request_str}"""


def suggest_better_model(current_task: str, current_model: str) -> str:
    """Suggest a better model for the current task.

    Args:
        current_task: Description of the current task
        current_model: Current model being used

    Returns:
        Suggestion for a better model
    """
    logger = get_logger()

    # Simple heuristics for model selection
    task_lower = current_task.lower()

    suggestions = []

    # Coding tasks
    if any(keyword in task_lower for keyword in ['code', 'programming', 'function', 'script', 'debug', 'algorithm']):
        if 'coder' not in current_model.lower():
            suggestions.append("qwen2.5-coder:7b - Specialized for coding tasks")
        if 'deepcoder' not in current_model.lower():
            suggestions.append("deepcoder:14b - Advanced coding assistant")

    # Analysis tasks
    if any(keyword in task_lower for keyword in ['analyze', 'analysis', 'examine', 'review', 'evaluate']) and 'qwen' not in current_model.lower():
        suggestions.append("qwen2.5:7b-instruct-q4_K_M - Strong analytical capabilities")

    # File operations
    if any(keyword in task_lower for keyword in ['file', 'directory', 'folder', 'list', 'read', 'write']) and 'deepcoder' not in current_model.lower():
        suggestions.append("deepcoder:14b - Has comprehensive file operation tools")

    if not suggestions:
        return f"✅ Current model ({current_model}) seems appropriate for this task."

    suggestion_text = f"""🎯 BETTER MODEL SUGGESTIONS for current task:

Current model: {current_model}
Task: {current_task}

Recommended alternatives:
""" + "\n".join(f"• {suggestion}" for suggestion in suggestions) + """

To request a model swap, use: request_model_swap("reason", "preferred_model_id")"""

    logger.info(f"Suggested better models for '{current_task}': {suggestions}")
    return suggestion_text


# Register the tools
if LANGCHAIN_AVAILABLE:
    for fn in [request_model_swap, suggest_better_model]:
        register_tool(StructuredTool.from_function(fn, name=fn.__name__, description=fn.__doc__ or fn.__name__))
else:
    # Use the standard ToolWrapper from registry
    from src.tools.registry import ToolWrapper
    for _f in [request_model_swap, suggest_better_model]:
        register_tool(ToolWrapper(_f))


__all__ = ["request_model_swap", "suggest_better_model"]
