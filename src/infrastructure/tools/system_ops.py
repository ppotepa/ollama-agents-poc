"""System tools."""
from __future__ import annotations

import os
import platform
import shutil

try:
    from langchain.tools import StructuredTool

    from src.tools.registry import register_tool
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False
    from src.tools.registry import register_tool


def get_system_info() -> str:
    info = [
        f"💻 System: {platform.system()} {platform.release()}",
        f"🏗️ Arch: {platform.machine()}",
        f"🐍 Python: {platform.python_version()}",
        f"📍 CWD: {os.getcwd()}",
    ]
    try:
        total, used, free = shutil.disk_usage(os.getcwd())
        info.append(f"💾 Disk Free: {free // (2**30)} GB")
    except Exception:
        pass
    return "\n".join(info)


if LANGCHAIN_AVAILABLE:
    register_tool(StructuredTool.from_function(get_system_info, name="get_system_info", description="System diagnostics."))
else:
    class _Tool:
        name = "get_system_info"
        description = "System diagnostics."
        def __call__(self):  # pragma: no cover
            return get_system_info()
    register_tool(_Tool())

__all__ = ["get_system_info"]
