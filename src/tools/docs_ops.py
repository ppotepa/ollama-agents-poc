"""Documentation tools."""
from __future__ import annotations

import os, glob, re

try:
    from langchain.tools import StructuredTool
    from src.tools.registry import register_tool
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False
    from src.tools.registry import register_tool


def generate_documentation(directory: str = ".", output_file: str = "API_DOCS.md") -> str:
    docs = ["# API Documentation\n"]
    for fp in glob.glob(os.path.join(directory, "**", "*.py"), recursive=True):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                content = f.read()
            functions = re.findall(r'^def\s+(\w+)\(', content, re.MULTILINE)
            classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
            if functions or classes:
                docs.append(f"\n## {fp}\n")
                if classes:
                    docs.append("### Classes")
                    docs.extend([f"- {c}" for c in classes])
                if functions:
                    docs.append("### Functions")
                    docs.extend([f"- {fn}()" for fn in functions])
        except Exception:
            continue
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(docs))
    return f"📚 Generated documentation: {output_file}"


if LANGCHAIN_AVAILABLE:
    register_tool(StructuredTool.from_function(generate_documentation, name="generate_documentation", description="Generate simple docs."))
else:
    class _Tool:
        name = "generate_documentation"
        description = "Generate simple docs."
        def __call__(self, directory: str = ".", output_file: str = "API_DOCS.md"):  # pragma: no cover
            return generate_documentation(directory=directory, output_file=output_file)
    register_tool(_Tool())

__all__ = ["generate_documentation"]
