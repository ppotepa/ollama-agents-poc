# DeepCoder Modular Architecture Implementation Guideline

## 🎯 **Current State Analysis**

### **Existing Architecture**
- **Single-file implementation**: `deepcoder_interactive_stream.py` (1,198 lines)
- **Basic structure**: Organized in sections with clear separation markers
- **Mixed responsibilities**: All components in one file (streaming, tools, UI, model config)
- **Plugin potential**: Well-structured sections that can be extracted into modules

### **Current Components Identified from deepcoder_interactive_stream.py**
1. **Streaming System** (Lines 62-232):
   - `stream_text()`, `stream_multiline()`, `show_thinking_animation()`
   - `StreamingCallbackHandler` class with LangChain callbacks
   - `RealTimeCapture` class for output processing

2. **Tool System** (Lines 235-692):
   - **File Operations**: `write_file()`, `read_file()`, `append_file()`, `list_files()`
   - **Code Execution**: `run_command()`, `run_python_code()`
   - **Project Management**: `create_project()`, `analyze_project()`
   - **Web & Documentation**: `open_documentation()`, `generate_documentation()`

3. **System Information** (Lines 661-692):
   - `get_system_info()` with platform detection and resource monitoring

4. **Model Configuration** (Lines 781-863):
   - ChatOllama setup with streaming enabled
   - Agent initialization with tool registration
   - System message and prompt engineering

5. **UI System** (Lines 864-1198):
   - REPL interface with command shortcuts
   - Help system and error handling
   - Progressive reveal animations

### **Dependencies Identified**
- **Core**: `langchain_ollama`, `langchain.agents`, `langchain.tools`
- **Optional**: `psutil`, `rich`, `colorama` (with graceful fallbacks)
- **System**: Standard library modules for file ops, subprocess, etc.

---

## 🏗️ **Proposed Modular Architecture**

### **Directory Structure**
```
deepcoder/
├── main.py                    # Entry point and CLI
├── config/
│   ├── __init__.py
│   ├── settings.py           # Global configuration
│   └── models.yaml           # Model definitions
├── core/
│   ├── __init__.py
│   ├── base_agent.py         # Abstract agent interface
│   ├── streaming.py          # Extracted streaming utilities
│   ├── tool_manager.py       # Tool registration and management
│   └── ui.py                 # User interface components
├── models/
│   ├── __init__.py
│   ├── base_model.py         # Abstract model interface
│   ├── deepcoder_basic/
│   │   ├── __init__.py
│   │   └── agent.py          # Basic DeepCoder implementation
│   ├── llama/
│   │   ├── __init__.py
│   │   └── agent.py          # Llama model implementation
│   └── codellama/
│       ├── __init__.py
│       └── agent.py          # CodeLlama implementation
├── tools/
│   ├── __init__.py
│   ├── file_ops.py           # File operations (write_file, read_file, etc.)
│   ├── system.py             # System commands and info
│   ├── project.py            # Project management tools
│   ├── code_execution.py     # Code running and execution
│   └── web.py                # Web and documentation tools
├── utils/
│   ├── __init__.py
│   ├── animations.py         # UI animations and progressive reveal
│   ├── system_info.py        # System information gathering
│   └── helpers.py            # General utilities
└── requirements.txt
```

---

## 🔧 **Component Extraction Plan**

### **1. Core Streaming Module** (`core/streaming.py`)
Extract from lines 62-232:
```python
# Functions to extract:
- stream_text()
- stream_multiline() 
- show_thinking_animation()
- progressive_reveal()
- StreamingCallbackHandler class
- RealTimeCapture class
```

### **2. Tools Modules** (`tools/`)
Extract from lines 235-692:

**file_ops.py**:
```python
- write_file(), read_file(), append_file()
- list_files(), search_files(), delete_file(), copy_file()
```

**system.py**:
```python
- run_command()
- get_system_info()
```

**code_execution.py**:
```python
- run_python_code()
```

**project.py**:
```python
- create_project()
- analyze_project()
```

**web.py**:
```python
- open_documentation()
- generate_documentation()
```

### **3. UI Components** (`core/ui.py`)
Extract from lines 864-1198:
```python
- print_help()
- handle_shortcut()
- Main REPL logic
- Error handling and user interaction
```

### **4. Model Implementation** (`models/deepcoder_basic/agent.py`)
Extract from lines 781-863:
```python
- ChatOllama configuration
- Agent initialization
- System message and prompts
- Tool registration logic
```

---

## 📋 **Base Interfaces for Plugin Architecture**

### **1. Base Model Interface**
```python
# models/base_model.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseModel(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', 'unknown')
        self.model_id = config.get('model_id', '')
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the model and return success status"""
        pass
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response from the model"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, bool]:
        """Return model capabilities"""
        pass
    
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Check if model supports streaming"""
        pass
    
    @abstractmethod
    def get_available_tools(self) -> List[str]:
        """Get list of tools this model supports"""
        pass
```

### **2. Tool Registration System**
```python
# core/tool_manager.py
from typing import Dict, List, Callable, Any
from functools import wraps

# Global tool registry
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}

def register_tool(name: str, description: str = "", category: str = "general"):
    """Decorator to register tools for plugin system"""
    def decorator(func: Callable):
        TOOL_REGISTRY[name] = {
            'function': func,
            'description': description,
            'category': category,
            'name': name
        }
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

def get_tools_by_category(category: str) -> List[str]:
    """Get all tools in a specific category"""
    return [name for name, tool in TOOL_REGISTRY.items() 
            if tool['category'] == category]

def get_all_tools() -> Dict[str, Any]:
    """Get all registered tools"""
    return TOOL_REGISTRY
```

---

## 🚀 **Implementation Phases**

### **Phase 1: Setup Infrastructure** (Week 1)
1. **Create directory structure**
2. **Extract streaming utilities** to `core/streaming.py`
3. **Create tool registration system** in `core/tool_manager.py`
4. **Setup base interfaces** (`BaseModel`, `BaseAgent`)

### **Phase 2: Extract Components** (Week 2)
1. **Move tool functions** to `tools/` modules with `@register_tool` decorators
2. **Extract UI components** to `core/ui.py`
3. **Create utility modules** in `utils/`
4. **Setup configuration system** with `models.yaml`

### **Phase 3: Create Model System** (Week 3)
1. **Implement DeepCoder model** in `models/deepcoder_basic/agent.py`
2. **Create model factory** for dynamic loading
3. **Add CLI interface** for model selection
4. **Implement runtime model switching**

### **Phase 4: Plugin Architecture** (Week 4)
1. **Add support for custom tools**
2. **Create plugin discovery system**
3. **Add model plugin support**
4. **Documentation and examples**

---

## 🎛️ **Configuration System**

### **models.yaml**
```yaml
models:
  deepcoder_basic:
    name: "DeepCoder Basic"
    model_id: "deepcoder:14b"
    provider: "ollama"
    capabilities:
      coding: true
      general_qa: true
      file_operations: true
      streaming: true
    parameters:
      temperature: 0.1
      num_ctx: 16384
      num_predict: 2048
      repeat_penalty: 1.1
      top_k: 40
      top_p: 0.9
    tools:
      - file_ops
      - system
      - project
      - code_execution
      - web
    system_message: |
      You are DeepCoder, an advanced AI coding assistant...
    
  llama_basic:
    name: "Llama Basic"
    model_id: "llama3.2:latest"
    provider: "ollama"
    capabilities:
      coding: false
      general_qa: true
      file_operations: false
      streaming: true
    parameters:
      temperature: 0.7
      num_ctx: 8192
      num_predict: 1024
    tools:
      - system
      - web
```

---

## 📊 **Migration Benefits**

### **Immediate Benefits**
- ✅ **Clean separation of concerns**
- ✅ **Reusable components across models**
- ✅ **Easy testing of individual modules**
- ✅ **Plugin-ready architecture**

### **Long-term Benefits**  
- ✅ **Easy addition of new models** (drop-in plugins)
- ✅ **Custom tool development** (register_tool decorator)
- ✅ **Model-specific optimizations**
- ✅ **Community plugin ecosystem**

### **User Experience**
- ✅ **Same familiar interface**
- ✅ **Runtime model switching**: `/switch-model llama`
- ✅ **Model-specific capabilities**: Automatic tool filtering
- ✅ **Backward compatibility**: Existing commands work unchanged

---

## 🎯 **Next Steps**

### **Immediate Actions**
1. **Create directory structure**: `mkdir -p deepcoder/{core,models,tools,utils,config}`
2. **Extract streaming module**: Move streaming functions to `core/streaming.py`
3. **Setup tool registration**: Create `core/tool_manager.py` with decorator system
4. **Begin tool extraction**: Start with `tools/file_ops.py`

### **Success Criteria**
- ✅ Original functionality preserved
- ✅ New models can be added as plugins
- ✅ Tools can be registered dynamically
- ✅ Runtime model switching works
- ✅ Configuration-driven model loading

The modular architecture will transform the current single-file implementation into a flexible, extensible platform while preserving all existing functionality and user experience.
