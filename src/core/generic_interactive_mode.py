"""Generic interactive agent system for interactive sessions."""

import os
import sys
import traceback
import threading
import time
from typing import Optional, Dict, Any
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent
from langchain.tools import StructuredTool

try:
    from langchain.agents import AgentType
    AGENT_TYPE = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
except Exception:
    AGENT_TYPE = "structured-chat-zero-shot-react-description"


class BusyIndicator:
    """Fancy loading indicator for interactive mode."""
    
    def __init__(self, message: str = "Awaiting response"):
        self.message = message
        self.is_running = False
        self.thread = None
    
    def start(self):
        """Start the loading animation."""
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._animate)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the loading animation."""
        if not self.is_running:
            return
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        # Clear the line and show completion
        print(f"\r{' ' * (len(self.message) + 10)}\r", end='', flush=True)
    
    def _animate(self):
        """Animation loop with increasing dots."""
        dots = 0
        while self.is_running:
            dots_str = '.' * (dots % 4)  # 0 to 3 dots
            spaces = ' ' * (3 - len(dots_str))  # Keep consistent width
            print(f"\r💭 {self.message}{dots_str}{spaces}", end='', flush=True)
            time.sleep(0.5)
            dots += 1


class LoadingIndicator:
    """Fancy loading indicator with animated dots."""
    
    def __init__(self, message: str = "Awaiting response"):
        self.message = message
        self.is_running = False
        self.thread = None
        
    def start(self):
        """Start the loading animation."""
        self.is_running = True
        self.thread = threading.Thread(target=self._animate)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop the loading animation."""
        self.is_running = False
        if self.thread:
            self.thread.join()
        # Clear the line
        print("\r" + " " * (len(self.message) + 10) + "\r", end="", flush=True)
        
    def _animate(self):
        """Animate the loading dots."""
        dots = 0
        while self.is_running:
            dots_str = "." * ((dots % 4) + 1)
            print(f"\r🤖 {self.message}{dots_str:<4}", end="", flush=True)
            time.sleep(0.5)
            dots += 1


class GenericInteractiveAgent:
    """Generic interactive agent that can work with any model configuration."""
    
    def __init__(self, agent_name: str):
        """Initialize the interactive agent with the specified agent configuration."""
        self.agent_name = agent_name
        self.agent = None
        self.model_config = None
        
        # Show loading while initializing
        loader = LoadingIndicator("Initializing agent")
        loader.start()
        
        try:
            self._load_agent_config()
            self._setup_agent()
            loader.stop()
            print(f"✅ Agent '{agent_name}' ready!")
        except Exception as e:
            loader.stop()
            print(f"❌ Failed to initialize agent: {e}")
            raise
    
    def _load_agent_config(self):
        """Load agent configuration from YAML."""
        try:
            # Add parent directory to path for imports
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            
            from integrations.model_config_reader import ModelConfigReader
            
            config_reader = ModelConfigReader('src/config/models.yaml')
            self.model_config = config_reader.get_model(self.agent_name)
            
            if not self.model_config:
                # Try to find by model ID or partial match
                for model in config_reader.get_all_models():
                    if (self.agent_name in model.model_id or 
                        model.model_id.startswith(self.agent_name) or
                        self.agent_name in model.short_name):
                        self.model_config = model
                        break
                        
            if not self.model_config:
                raise ValueError(f"No configuration found for agent '{self.agent_name}'")
                
        except Exception as e:
            print(f"⚠️  Warning: Could not load agent config: {e}")
            # Create a basic fallback config
            self.model_config = type('ModelConfig', (), {
                'model_id': self.agent_name,
                'short_name': self.agent_name,
                'name': self.agent_name.title(),
                'supports_coding': True,
                'parameters': {'temperature': 0.7, 'num_ctx': 8192},
                'tools': ['file_ops', 'system'],
                'system_message': None
            })()
    
    def _get_tools(self):
        """Get tools based on agent configuration."""
        tools = []
        
        # Basic file operations (if agent supports file operations)
        if hasattr(self.model_config, 'tools') and 'file_ops' in self.model_config.tools:
            tools.extend([
                StructuredTool.from_function(
                    func=self._write_file,
                    name="write_file",
                    description="Write/create a file. Args: filepath (path), contents (full file content)."
                ),
                StructuredTool.from_function(
                    func=self._read_file,
                    name="read_file", 
                    description="Read file contents. Args: filepath (path)."
                ),
                StructuredTool.from_function(
                    func=self._list_files,
                    name="list_files",
                    description="List files in directory. Args: directory (path, default: current)."
                )
            ])
        
        # System operations
        if hasattr(self.model_config, 'tools') and 'system' in self.model_config.tools:
            tools.append(
                StructuredTool.from_function(
                    func=self._run_command,
                    name="run_command",
                    description="Run system command. Args: command (shell command to execute)."
                )
            )
        
        # Project operations (for coding agents)
        if (hasattr(self.model_config, 'tools') and 'project' in self.model_config.tools and 
            hasattr(self.model_config, 'supports_coding') and self.model_config.supports_coding):
            tools.append(
                StructuredTool.from_function(
                    func=self._analyze_project,
                    name="analyze_project",
                    description="Analyze project structure and files. Args: path (optional project path)."
                )
            )
        
        return tools
    
    def _write_file(self, filepath: str, contents: str) -> str:
        """Write file tool implementation."""
        try:
            parent = os.path.dirname(filepath)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(contents)
            return f"✅ File written: {filepath} ({len(contents)} bytes)"
        except Exception as e:
            return f"❌ Error writing file {filepath}: {e}"
    
    def _read_file(self, filepath: str) -> str:
        """Read file tool implementation."""
        try:
            if not os.path.exists(filepath):
                return f"❌ File does not exist: {filepath}"
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            return f"📄 Content of {filepath}:\n{content}"
        except Exception as e:
            return f"❌ Error reading file {filepath}: {e}"
    
    def _list_files(self, directory: str = ".") -> str:
        """List files tool implementation."""
        try:
            path = Path(directory)
            if not path.exists():
                return f"❌ Directory does not exist: {directory}"
            
            files = []
            dirs = []
            for item in path.iterdir():
                if item.is_file():
                    files.append(item.name)
                elif item.is_dir():
                    dirs.append(f"{item.name}/")
            
            result = f"📁 Contents of {directory}:\n"
            if dirs:
                result += "Directories: " + ", ".join(sorted(dirs)) + "\n"
            if files:
                result += "Files: " + ", ".join(sorted(files))
            
            return result
        except Exception as e:
            return f"❌ Error listing directory {directory}: {e}"
    
    def _run_command(self, command: str) -> str:
        """Run system command tool implementation."""
        try:
            import subprocess
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            if result.returncode == 0:
                return f"✅ Command executed successfully:\n{result.stdout}"
            else:
                return f"❌ Command failed (exit code {result.returncode}):\n{result.stderr}"
        except subprocess.TimeoutExpired:
            return "❌ Command timed out (30s limit)"
        except Exception as e:
            return f"❌ Error executing command: {e}"
    
    def _analyze_project(self, path: str = ".") -> str:
        """Analyze project structure tool implementation."""
        try:
            project_path = Path(path)
            if not project_path.exists():
                return f"❌ Project path does not exist: {path}"
            
            # Basic project analysis
            analysis = f"🔍 Project Analysis: {project_path.absolute()}\n\n"
            
            # Count files by type
            file_counts = {}
            total_files = 0
            
            for file_path in project_path.rglob("*"):
                if file_path.is_file():
                    total_files += 1
                    ext = file_path.suffix.lower()
                    file_counts[ext] = file_counts.get(ext, 0) + 1
            
            analysis += f"📊 Total files: {total_files}\n"
            analysis += "📈 File types:\n"
            for ext, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                ext_name = ext if ext else "(no extension)"
                analysis += f"  {ext_name}: {count}\n"
            
            # Look for common project files
            common_files = [
                "README.md", "README.rst", "README.txt",
                "package.json", "requirements.txt", "Pipfile", "pyproject.toml",
                "Dockerfile", "docker-compose.yml", ".gitignore",
                "Makefile", "CMakeLists.txt"
            ]
            
            found_files = []
            for common_file in common_files:
                if (project_path / common_file).exists():
                    found_files.append(common_file)
            
            if found_files:
                analysis += f"\n🔧 Key project files found: {', '.join(found_files)}"
            
            return analysis
            
        except Exception as e:
            return f"❌ Error analyzing project: {e}"
    
    def _setup_agent(self):
        """Setup the LangChain agent with model and tools."""
        # Start loading indicator for agent setup
        setup_loader = BusyIndicator("Initializing agent")
        setup_loader.start()
        
        try:
            # Get model parameters
            model_id = getattr(self.model_config, 'model_id', self.agent_name)
            params = getattr(self.model_config, 'parameters', {})
            
            # Create the language model
            llm = ChatOllama(
                model=model_id,
                temperature=params.get('temperature', 0.7),
                num_ctx=params.get('num_ctx', 8192),
            )
            
            # Get tools for this agent
            tools = self._get_tools()
            
            # Create system message
            system_message = getattr(self.model_config, 'system_message', None)
            if not system_message:
                if hasattr(self.model_config, 'supports_coding') and self.model_config.supports_coding:
                    system_message = (
                        f"You are {getattr(self.model_config, 'name', self.agent_name)}, "
                        "an AI assistant specialized in software development and coding tasks. "
                        "Use the available tools to help with file operations, code analysis, and project management. "
                        "Always execute actions when requested - don't ask for permission."
                    )
                else:
                    system_message = (
                        f"You are {getattr(self.model_config, 'name', self.agent_name)}, "
                        "a helpful AI assistant. Use the available tools to assist with various tasks. "
                        "Always execute actions when requested - don't ask for permission."
                    )
            
            # Initialize the agent
            self.agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent=AGENT_TYPE,
                verbose=True,
                handle_parsing_errors=True,
                agent_kwargs={"system_message": system_message},
            )
            
            setup_loader.stop()
            print("✅ Agent ready!")
            
        except Exception as e:
            setup_loader.stop()
            print(f"❌ Error setting up agent: {e}")
            raise
    
    def run_query(self, query: str) -> str:
        """Run a single query against the agent."""
        if not self.agent:
            return "❌ Agent not initialized"
        
        # Start loading indicator
        loader = BusyIndicator("Awaiting response")
        loader.start()
        
        try:
            result = self.agent.invoke({"input": query})
            loader.stop()
            return result.get("output", str(result))
        except Exception as e:
            loader.stop()
            return f"❌ Error processing query: {e}"
    
    def run_interactive_session(self):
        """Run an interactive session with the agent."""
        print(f"🤖 {getattr(self.model_config, 'name', self.agent_name)} Interactive Session")
        print("Type 'exit' to quit.\n")
        
        # Handle repository setup for coding agents
        if hasattr(self.model_config, 'supports_coding') and self.model_config.supports_coding:
            self._handle_repository_setup()
        
        # Show agent info
        print(f"💡 Using model: {getattr(self.model_config, 'model_id', self.agent_name)}")
        if hasattr(self.model_config, 'tools'):
            print(f"🔧 Available tools: {', '.join(self.model_config.tools)}")
        print("💡 Use Ctrl+C to exit\n")
        
        # Interactive loop
        while True:
            try:
                cmd = input("📝> ").strip()
                if cmd.lower() in ("exit", "quit"):
                    print("👋 Session ended!")
                    break
                
                if not cmd:
                    continue
                
                # Process the command
                response = self.run_query(cmd)
                print(f"\n💡 Response:\n{response}\n{'-'*60}\n")
                
            except KeyboardInterrupt:
                print("\n👋 Session interrupted.")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                if hasattr(e, '__traceback__'):
                    traceback.print_exc()
    
    def _handle_repository_setup(self):
        """Handle repository setup for coding agents with repository selection."""
        try:
            from core.helpers import display_repository_selection, validate_repository_requirement
            
            print(f"🔍 Agent '{self.agent_name}' is a coding agent and works best with a repository.")
            
            # Show repository selection menu
            selected_url = display_repository_selection()
            
            if selected_url:
                try:
                    original_cwd = os.getcwd()
                    data_path = os.path.join(original_cwd, "data")
                    
                    # Show loading while setting up repository
                    loader = LoadingIndicator("Setting up repository")
                    loader.start()
                    
                    validation_passed, working_dir = validate_repository_requirement(
                        self.agent_name, ".", selected_url, data_path
                    )
                    
                    loader.stop()
                    
                    if working_dir != ".":
                        print(f"✅ Repository cloned and validated: {working_dir}")
                        os.chdir(working_dir)
                        print(f"✅ Changed working directory to: {working_dir}")
                    else:
                        print(f"✅ Repository validation passed")
                        
                except Exception as e:
                    if 'loader' in locals():
                        loader.stop()
                    print(f"❌ Error setting up repository: {e}")
                    print("⚠️  Continuing without repository - some features may be limited.\n")
            else:
                print("⚠️  Continuing without repository - some features may be limited.\n")
                    
        except Exception as e:
            print(f"⚠️  Warning: Could not handle repository setup: {e}")


# Public functions for main.py compatibility
def run_interactive_session(agent_name: str):
    """Run an interactive session with any agent."""
    try:
        agent = GenericInteractiveAgent(agent_name)
        agent.run_interactive_session()
    except Exception as e:
        print(f"❌ Error starting interactive session for '{agent_name}': {e}")
        if hasattr(e, '__traceback__'):
            traceback.print_exc()


# Direct execution support
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        agent_name = sys.argv[1]
        run_interactive_session(agent_name)
    else:
        print("Usage: python generic_interactive_mode.py <agent_name>")
