"""File operation tools."""
from __future__ import annotations

import os, glob, shutil

try:
    from langchain.tools import StructuredTool
    from src.tools.registry import register_tool
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False
    from src.tools.registry import register_tool


def write_file(filepath: str, contents: str) -> str:
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(contents)
    return f"✅ File written: {filepath} ({len(contents)} bytes)"


def read_file(filepath: str) -> str:
    if not os.path.exists(filepath):
        return f"❌ File {filepath} does not exist"
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def append_file(filepath: str, contents: str) -> str:
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(contents)
    return f"✅ Appended to {filepath} ({len(contents)} bytes)"


def list_files(directory: str = ".", pattern: str = "*") -> str:
    if not os.path.exists(directory):
        return f"❌ Directory {directory} does not exist"
    items = []
    for item in glob.glob(os.path.join(directory, pattern)):
        if os.path.isfile(item):
            items.append(os.path.relpath(item))
        elif os.path.isdir(item):
            items.append(os.path.relpath(item) + "/")
    return "\n".join(items) if items else "📂 Empty"


def delete_file(filepath: str) -> str:
    if os.path.isfile(filepath):
        os.remove(filepath)
        return f"🗑️ Deleted file: {filepath}"
    if os.path.isdir(filepath):
        shutil.rmtree(filepath)
        return f"🗑️ Deleted directory: {filepath}"
    return f"❌ Path {filepath} does not exist"


def copy_file(source: str, destination: str) -> str:
    if os.path.isfile(source):
        parent = os.path.dirname(destination)
        if parent:
            os.makedirs(parent, exist_ok=True)
        shutil.copy2(source, destination)
        return f"📋 Copied file: {source} → {destination}"
    if os.path.isdir(source):
        shutil.copytree(source, destination, dirs_exist_ok=True)
        return f"📋 Copied directory: {source} → {destination}"
    return f"❌ Source {source} does not exist"


if LANGCHAIN_AVAILABLE:
    for fn in [write_file, read_file, append_file, list_files, delete_file, copy_file]:
        register_tool(StructuredTool.from_function(fn, name=fn.__name__, description=fn.__doc__ or fn.__name__))
else:
    class _Wrap:
        def __init__(self, fn):
            self._fn = fn
            self.name = fn.__name__
            self.description = fn.__doc__ or fn.__name__
        def __call__(self, *a, **kw):  # pragma: no cover
            return self._fn(*a, **kw)
    for _f in [write_file, read_file, append_file, list_files, delete_file, copy_file]:
        register_tool(_Wrap(_f))

__all__ = ["write_file", "read_file", "append_file", "list_files", "delete_file", "copy_file"]
