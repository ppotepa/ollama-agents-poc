"""Tool call processing and normalization."""

import json
import re
from typing import Any, Dict, List, Optional

from src.utils.enhanced_logging import get_logger


class ToolCallProcessor:
    """Processes and normalizes tool calls for Continue integration."""

    def __init__(self):
        """Initialize the tool call processor."""
        self.logger = get_logger()
        
        # Tool name mapping for normalization
        self.tool_name_mapping = {
            "create_new_file": "files:create",
            "edit_file": "files:edit", 
            "read_file": "files:read",
            "write_file": "files:write",
            "list_files": "files:list",
            "delete_file": "files:delete",
            "run_command": "terminal:run",
            "execute_command": "terminal:run",
            "search_files": "search:files",
            "search_code": "search:code",
            "get_directory_structure": "files:tree"
        }

    def parse_user_tool_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON tool calls from user text.
        
        Args:
            text: Text potentially containing JSON tool calls
            
        Returns:
            Parsed tool call dictionary or None
        """
        if not text or not isinstance(text, str):
            return None

        try:
            # Try direct JSON parse first
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Look for JSON-like patterns in text
        json_patterns = [
            r'\{[^{}]*"name"\s*:\s*"[^"]*"[^{}]*\}',
            r'\{[^{}]*"function"\s*:\s*\{[^{}]*\}[^{}]*\}',
            r'\{[^{}]*"tool_calls"\s*:\s*\[[^\]]*\][^{}]*\}'
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        return None

    def normalize_tool_call(self, tool: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize tool call names and structure.
        
        Args:
            tool: Raw tool call dictionary
            
        Returns:
            Normalized tool call or None if invalid
        """
        if not isinstance(tool, dict):
            return None

        try:
            # Extract tool name from various possible structures
            tool_name = None
            arguments = {}

            if "function" in tool:
                func = tool["function"]
                if isinstance(func, dict):
                    tool_name = func.get("name")
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            arguments = json.loads(args)
                        except json.JSONDecodeError:
                            arguments = {"raw_args": args}
                    else:
                        arguments = args
            elif "name" in tool:
                tool_name = tool["name"]
                arguments = tool.get("arguments", {})

            if not tool_name:
                return None

            # Normalize tool name
            normalized_name = self.tool_name_mapping.get(tool_name, tool_name)

            # Create normalized tool call
            normalized = {
                "id": tool.get("id", f"call_{hash(tool_name) % 10000:04d}"),
                "type": "function",
                "function": {
                    "name": normalized_name,
                    "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)
                }
            }

            self.logger.debug(f"Normalized tool call: {tool_name} -> {normalized_name}")
            return normalized

        except Exception as e:
            self.logger.error(f"Error normalizing tool call: {e}")
            return None

    def extract_tool_calls_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls from text content.
        
        Args:
            text: Text to search for tool calls
            
        Returns:
            List of extracted and normalized tool calls
        """
        tool_calls = []
        
        if not text:
            return tool_calls

        # Look for explicit tool call patterns
        patterns = [
            r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
            r'```json\s*(\{.*?"name"\s*:.*?\})\s*```',
            r'Tool:\s*(\{.*?\})',
            r'Function:\s*(\{.*?\})'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                tool_data = self.parse_user_tool_json(match)
                if tool_data:
                    normalized = self.normalize_tool_call(tool_data)
                    if normalized:
                        tool_calls.append(normalized)

        return tool_calls

    def validate_tool_call(self, tool_call: Dict[str, Any]) -> bool:
        """Validate tool call structure.
        
        Args:
            tool_call: Tool call to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            required_fields = ["id", "type", "function"]
            if not all(field in tool_call for field in required_fields):
                return False

            function = tool_call["function"]
            if not isinstance(function, dict) or "name" not in function:
                return False

            return True

        except Exception:
            return False

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools for Continue integration.
        
        Returns:
            List of tool definitions
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "files:create",
                    "description": "Create a new file with content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path"},
                            "content": {"type": "string", "description": "File content"}
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "files:edit",
                    "description": "Edit an existing file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path"},
                            "content": {"type": "string", "description": "New content"}
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "files:read",
                    "description": "Read file content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "terminal:run",
                    "description": "Execute terminal command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Command to execute"}
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search:files",
                    "description": "Search for files",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "path": {"type": "string", "description": "Search path"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
