"""Aggregate tool submodules so registration side-effects occur.

Importing these modules populates the central registry via register_tool.
"""

# Import order not critical; ignore failures gracefully
from importlib import import_module

_MODULES = [
	"src.tools.file_ops",
	"src.tools.project_ops",
	"src.tools.system_ops",
	"src.tools.docs_ops",
	"src.tools.exec_ops",
	"src.tools.web_ops",
	"src.tools.repository_context_tool",
]

for _m in _MODULES:  # pragma: no cover
	try:
		import_module(_m)
	except Exception:
		pass

del import_module, _MODULES
