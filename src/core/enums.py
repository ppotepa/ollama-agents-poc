"""Core enumerations for the agent system."""

from enum import Enum, auto, Flag

class ModelFamily(Enum):
    """Model architecture families."""
    LLAMA = auto()
    CODELLAMA = auto()
    TINYLLAMA = auto()
    QWEN = auto()
    QWEN_CODER = auto() 
    MISTRAL = auto()
    GEMMA = auto()
    PHI = auto()
    DEEPSEEK = auto()
    DEEPSEEK_CODER = auto()
    
    @classmethod
    def from_name(cls, name: str) -> "ModelFamily":
        """Parse model family from name string."""
        name = name.lower()
        if "llama" in name:
            return cls.LLAMA if "code" not in name else cls.CODELLAMA
        elif "qwen" in name:
            return cls.QWEN_CODER if "coder" in name else cls.QWEN
        elif "mistral" in name:
            return cls.MISTRAL
        elif "gemma" in name:
            return cls.GEMMA
        elif "phi" in name:
            return cls.PHI
        elif "deepseek" in name:
            return cls.DEEPSEEK_CODER if "coder" in name else cls.DEEPSEEK
        elif "tiny" in name:
            return cls.TINYLLAMA
        return cls.LLAMA  # Default
    
class Domain(Enum):
    """Model domain specialization."""
    GENERAL = auto()    # General purpose
    CODING = auto()     # Code generation
    CHAT = auto()       # Chat optimized
    MATH = auto()       # Mathematics
    SCIENCE = auto()    # Scientific reasoning
    
class AgentCapability(Flag):
    """Agent capabilities as flag enum (can be combined)."""
    NONE = 0
    STREAMING = auto()        # Supports streaming output
    FUNCTION_CALLS = auto()   # Supports tool/function calling
    CODE = auto()             # Code generation
    FILES = auto()            # File operations
    SYSTEM = auto()           # System operations
    INTERNET = auto()         # Web search/fetch
    REPO_ANALYSIS = auto()    # Repository analysis
    
    # Common combinations
    BASIC = STREAMING
    CODER = STREAMING | FUNCTION_CALLS | CODE | FILES | REPO_ANALYSIS
    
    @classmethod
    def from_strings(cls, capability_strings):
        """Convert string capabilities to enum flags."""
        result = cls.NONE
        mapping = {
            "streaming": cls.STREAMING,
            "function_calls": cls.FUNCTION_CALLS, 
            "tool_calls": cls.FUNCTION_CALLS,
            "code": cls.CODE,
            "coding": cls.CODE,
            "files": cls.FILES,
            "file_operations": cls.FILES,
            "system": cls.SYSTEM,
            "web": cls.INTERNET,
            "internet": cls.INTERNET,
            "repo": cls.REPO_ANALYSIS,
            "repository": cls.REPO_ANALYSIS
        }
        
        # Handle both list and dict formats
        if isinstance(capability_strings, dict):
            for cap, enabled in capability_strings.items():
                if enabled and cap.lower() in mapping:
                    result |= mapping[cap.lower()]
        elif isinstance(capability_strings, list):
            for cap in capability_strings:
                cap_lower = cap.lower()
                if cap_lower in mapping:
                    result |= mapping[cap_lower]
        
        return result
    
    def to_strings(self):
        """Convert flags to list of capability strings."""
        result = []
        if self & self.STREAMING:
            result.append("streaming")
        if self & self.FUNCTION_CALLS:
            result.append("function_calls")
        if self & self.CODE:
            result.append("code")
        if self & self.FILES:
            result.append("files")
        if self & self.SYSTEM:
            result.append("system")
        if self & self.INTERNET:
            result.append("internet")
        if self & self.REPO_ANALYSIS:
            result.append("repository")
        return result

class ToolType(Enum):
    """Tool categories."""
    FILE = auto()           # File operations
    CODE = auto()           # Code execution
    SYSTEM = auto()         # System operations
    WEB = auto()            # Web operations
    REPO = auto()           # Repository operations
    UTILITY = auto()        # General utilities
    
    @classmethod
    def for_tool(cls, tool_name: str) -> "ToolType":
        """Get tool type from name."""
        tool_name = tool_name.lower()
        if any(x in tool_name for x in ["file", "read", "write", "append", "delete"]):
            return cls.FILE
        elif any(x in tool_name for x in ["run", "exec", "python"]):
            return cls.CODE
        elif any(x in tool_name for x in ["system", "info"]):
            return cls.SYSTEM
        elif any(x in tool_name for x in ["web", "http", "fetch", "search"]):
            return cls.WEB
        elif any(x in tool_name for x in ["repo", "git", "analyze"]):
            return cls.REPO
        return cls.UTILITY
