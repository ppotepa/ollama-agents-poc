"""Code execution tools (safe sandboxed approximations)."""
from __future__ import annotations

import contextlib
import io
import textwrap

try:
    from langchain.tools import StructuredTool

    from src.tools.registry import register_tool
    LANGCHAIN_AVAILABLE = True
except Exception:  # pragma: no cover
    LANGCHAIN_AVAILABLE = False
    from src.tools.registry import register_tool  # provide direct registration


def run_python(code: str) -> str:
    """Execute a small Python snippet and return stdout/stderr.

    Limitations:
      - No external package installation
      - Restricted builtins for safety
    """
    if len(code) > 4000:
        return "❌ Code too long (limit 4000 chars)."
    code = textwrap.dedent(code).strip()
    allowed_builtins = {"print": print, "range": range, "len": len, "enumerate": enumerate}
    local_env = {}
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(compile(code, "<user_code>", "exec"), {"__builtins__": allowed_builtins}, local_env)
    except Exception as e:  # pragma: no cover
        return f"❌ Execution error: {e}\n---\n{stdout.getvalue()}".strip()
    output = stdout.getvalue().strip()
    if not output:
        last_expr = local_env.get("_")
        if last_expr is not None:
            output = repr(last_expr)
    return output or "✅ (no output)"


if LANGCHAIN_AVAILABLE:
    register_tool(StructuredTool.from_function(run_python, name="run_python", description="Execute a short Python snippet (sandboxed)."))
else:  # fallback simple wrapper with name attr
    class _Tool:
        name = "run_python"
        description = "Execute a short Python snippet (sandboxed)."
        def __call__(self, code: str):  # pragma: no cover
            return run_python(code)
    register_tool(_Tool())

__all__ = ["run_python"]
