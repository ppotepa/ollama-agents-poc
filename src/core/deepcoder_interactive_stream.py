#!/usr/bin/env python3
# DeepCoder Enhanced Agent - Comprehensive AI Coding Assistant
# Single-file implementation with maximum capabilities

import sys
import os

# Add graceful dependency handling
try:
    from langchain_ollama import ChatOllama
    from langchain.agents import initialize_agent
    try:
        from langchain.agents import AgentType
        AGENT_TYPE = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
    except Exception:
        AGENT_TYPE = "structured-chat-zero-shot-react-description"
    from langchain.tools import StructuredTool
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ LangChain dependencies not found: {e}")
    print("💡 Please install with: pip install -r requirements.txt")
    print("🔧 Running in standalone mode with limited functionality...")
    LANGCHAIN_AVAILABLE = False

import subprocess
import traceback
import json
import shutil
import glob
import re
import time
import platform
from pathlib import Path
from typing import List, Dict, Any, Optional
import threading
import webbrowser
from urllib.parse import quote

# Optional dependencies with graceful fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import track
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import colorama
    colorama.init()
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

# ---------- STREAMING OUTPUT UTILITIES ----------

def stream_text(text: str, delay: float = 0.03, newline: bool = True):
    """Stream text character by character to simulate real-time generation."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    if newline:
        print()

def stream_multiline(text: str, line_delay: float = 0.1, char_delay: float = 0.02):
    """Stream multiline text with delays between lines and characters."""
    lines = text.split('\n')
    for i, line in enumerate(lines):
        stream_text(line, delay=char_delay, newline=True)
        if i < len(lines) - 1:  # Don't delay after the last line
            time.sleep(line_delay)

def show_thinking_animation(duration: float = 2.0, message: str = "🔄 Thinking"):
    """Show a thinking animation for a specified duration."""
    animation = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    end_time = time.time() + duration
    i = 0
    
    while time.time() < end_time:
        print(f"\r{message} {animation[i % len(animation)]}", end='', flush=True)
        time.sleep(0.1)
        i += 1
    
    print(f"\r{' ' * (len(message) + 5)}\r", end='', flush=True)  # Clear the line

def progressive_reveal(sections: List[tuple], section_delay: float = 0.5):
    """Progressively reveal sections of text with animations."""
    for title, content in sections:
        stream_text(f"\n{title}", delay=0.05)
        time.sleep(section_delay)
        stream_multiline(content, line_delay=0.1, char_delay=0.01)
        time.sleep(section_delay)

# ---------- STREAMING CALLBACK HANDLER ----------

class StreamingCallbackHandler:
    """Custom callback handler for real-time LLM output streaming."""
    
    def __init__(self):
        self.tokens = []
        self.is_thinking = False
        self.ignore_llm = False
        self.ignore_chain = False
        self.ignore_agent = False
        self.raise_error = False
        
    def on_llm_start(self, serialized: dict, prompts: List[str], **kwargs) -> None:
        """Called when LLM starts generating."""
        if not self.ignore_llm:
            print("🧠 AI is thinking...", end='', flush=True)
            self.is_thinking = True
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when a new token is generated."""
        if not self.ignore_llm:
            if self.is_thinking:
                print("\r🤖 AI Response: ", end='', flush=True)
                self.is_thinking = False
            print(token, end='', flush=True)
            self.tokens.append(token)
        
    def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM finishes generating."""
        if not self.ignore_llm:
            print()  # New line after completion
        
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Called when LLM encounters an error."""
        if self.raise_error:
            raise error
        print(f"\n❌ LLM Error: {error}")
        
    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs) -> None:
        """Called when a chain starts."""
        if not self.ignore_chain and serialized is not None:
            chain_name = serialized.get("name", "Unknown chain") if serialized else "Unknown chain"
            if "AgentExecutor" in chain_name:
                stream_text("🚀 Starting AI reasoning...", delay=0.02)
        
    def on_chain_end(self, outputs: dict, **kwargs) -> None:
        """Called when a chain ends."""
        if not self.ignore_chain:
            pass  # Chain completion handled elsewhere
        
    def on_chain_error(self, error: Exception, **kwargs) -> None:
        """Called when a chain encounters an error."""
        if self.raise_error:
            raise error
        print(f"\n❌ Chain Error: {error}")
        
    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        """Called when a tool starts executing."""
        if serialized is not None:
            tool_name = serialized.get("name", "Unknown tool") if serialized else "Unknown tool"
            stream_text(f"\n🔧 Using tool: {tool_name}", delay=0.02)
        
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes executing."""
        # Tool output is already handled by our streaming functions
        pass
        
    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Called when a tool encounters an error."""
        if self.raise_error:
            raise error
        print(f"\n❌ Tool Error: {error}")
        
    def on_agent_action(self, action, **kwargs) -> None:
        """Called when agent takes an action."""
        if not self.ignore_agent:
            stream_text(f"🎯 Action: {action.tool}", delay=0.02)
        
    def on_agent_finish(self, finish, **kwargs) -> None:
        """Called when agent finishes."""
        if not self.ignore_agent:
            stream_text("✅ Task completed!", delay=0.02)
    
    def on_text(self, text: str, **kwargs) -> None:
        """Called when arbitrary text is logged."""
        # Handle general text output
        pass

# Create global callback handler instance
streaming_callback = StreamingCallbackHandler() if LANGCHAIN_AVAILABLE else None

# ---------- REAL-TIME OUTPUT CAPTURE ----------

class RealTimeCapture:
    """Capture and stream output in real-time."""
    
    def __init__(self):
        self.output_buffer = []
        self.is_streaming = False
        
    def start_streaming(self):
        """Start the streaming process."""
        self.is_streaming = True
        self.output_buffer = []
        
    def add_output(self, text: str):
        """Add text to output buffer and stream it."""
        if self.is_streaming:
            # Process the text for better formatting - simplified for cleaner output
            if "Entering new AgentExecutor chain" in text:
                print("🚀 Starting AI reasoning...")
            elif "Action:" in text and not "Action Input:" in text:
                cleaned = text.replace("Action:", "").strip()
                print(f"🔧 Tool: {cleaned}")
            elif "Action Input:" in text:
                print("📝 Processing...")
            elif "Observation:" in text:
                print("👁️ Tool completed")
            elif "Thought:" in text:
                print("💭 Thinking...")
            elif "Final Answer:" in text:
                print("✅ Generating response...")
            
            self.output_buffer.append(text)
            
    def stop_streaming(self):
        """Stop the streaming process."""
        self.is_streaming = False
        return '\n'.join(self.output_buffer)

# Global capture instance
real_time_capture = RealTimeCapture()

# ---------- ENHANCED FILE OPERATIONS ----------

def write_file(filepath: str, contents: str) -> str:
    """Create or overwrite a file with the given contents."""
    try:
        show_thinking_animation(0.5, "📝 Writing file")
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(contents)
        result = f"✅ File written: {filepath} ({len(contents)} bytes)"
        stream_text(result, delay=0.02)
        return result
    except Exception as e:
        error_msg = f"❌ Error writing file {filepath}: {str(e)}"
        stream_text(error_msg, delay=0.02)
        return error_msg

def read_file(filepath: str) -> str:
    """Read the contents of a file."""
    try:
        show_thinking_animation(0.3, "📖 Reading file")
        if not os.path.exists(filepath):
            error_msg = f"❌ File {filepath} does not exist"
            stream_text(error_msg, delay=0.02)
            return error_msg
        
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        header = f"📄 File {filepath} ({len(content)} bytes):"
        stream_text(header, delay=0.02)
        print()  # New line before content
        
        # Stream file content with syntax highlighting effect
        if len(content) > 1000:
            stream_text("📋 Large file detected - showing first 1000 characters...", delay=0.02)
            content_to_show = content[:1000] + "\n... (truncated)"
        else:
            content_to_show = content
        
        stream_multiline(content_to_show, line_delay=0.05, char_delay=0.001)
        return f"{header}\n{content}"
        
    except Exception as e:
        error_msg = f"❌ Error reading file {filepath}: {str(e)}"
        stream_text(error_msg, delay=0.02)
        return error_msg

def append_file(filepath: str, contents: str) -> str:
    """Append contents to an existing file or create new if it doesn't exist."""
    try:
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(contents)
        return f"✅ Appended to {filepath} ({len(contents)} bytes)"
    except Exception as e:
        return f"❌ Error appending to file {filepath}: {str(e)}"

def list_files(directory: str = ".", pattern: str = "*") -> str:
    """List files in a directory with optional pattern matching."""
    try:
        if not os.path.exists(directory):
            return f"❌ Directory {directory} does not exist"
        
        files = []
        for item in glob.glob(os.path.join(directory, pattern)):
            if os.path.isfile(item):
                size = os.path.getsize(item)
                mod_time = time.ctime(os.path.getmtime(item))
                files.append(f"📄 {os.path.relpath(item)} ({size} bytes, {mod_time})")
            elif os.path.isdir(item):
                files.append(f"📁 {os.path.relpath(item)}/")
        
        if not files:
            return f"📂 No files found in {directory} matching '{pattern}'"
        
        return f"📂 Directory listing for {directory}:\n" + "\n".join(files)
    except Exception as e:
        return f"❌ Error listing directory {directory}: {str(e)}"

def search_files(search_term: str, directory: str = ".", file_pattern: str = "*.py") -> str:
    """Search for text within files matching a pattern."""
    try:
        results = []
        for filepath in glob.glob(os.path.join(directory, "**", file_pattern), recursive=True):
            if os.path.isfile(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    for line_num, line in enumerate(lines, 1):
                        if search_term.lower() in line.lower():
                            results.append(f"{filepath}:{line_num}: {line.strip()}")
                except Exception:
                    continue
        
        if not results:
            return f"🔍 No matches found for '{search_term}' in {file_pattern} files"
        
        return f"🔍 Search results for '{search_term}':\n" + "\n".join(results[:20])  # Limit to 20 results
    except Exception as e:
        return f"❌ Error searching files: {str(e)}"

def delete_file(filepath: str) -> str:
    """Delete a file or directory."""
    try:
        if os.path.isfile(filepath):
            os.remove(filepath)
            return f"🗑️ Deleted file: {filepath}"
        elif os.path.isdir(filepath):
            shutil.rmtree(filepath)
            return f"🗑️ Deleted directory: {filepath}"
        else:
            return f"❌ Path {filepath} does not exist"
    except Exception as e:
        return f"❌ Error deleting {filepath}: {str(e)}"

def copy_file(source: str, destination: str) -> str:
    """Copy a file or directory to a new location."""
    try:
        if os.path.isfile(source):
            parent = os.path.dirname(destination)
            if parent:
                os.makedirs(parent, exist_ok=True)
            shutil.copy2(source, destination)
            return f"📋 Copied file: {source} → {destination}"
        elif os.path.isdir(source):
            shutil.copytree(source, destination, dirs_exist_ok=True)
            return f"📋 Copied directory: {source} → {destination}"
        else:
            return f"❌ Source {source} does not exist"
    except Exception as e:
        return f"❌ Error copying {source} to {destination}: {str(e)}"

# ---------- CODE EXECUTION ----------

def run_command(command: str, working_directory: str = ".") -> str:
    """Execute a shell command and return the output."""
    try:
        stream_text(f"🔧 Executing: {command}", delay=0.02)
        show_thinking_animation(1.0, "⚙️ Running command")
        
        result = subprocess.run(
            command,
            shell=True,
            cwd=working_directory,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        # Progressive reveal of results
        sections = []
        
        if result.stdout:
            sections.append(("📤 STDOUT:", result.stdout))
        if result.stderr:
            sections.append(("⚠️ STDERR:", result.stderr))
        
        status = "✅ SUCCESS" if result.returncode == 0 else f"❌ ERROR (exit code: {result.returncode})"
        sections.append(("🎯 Status:", f"{status}\n📍 Directory: {working_directory}"))
        
        progressive_reveal(sections, section_delay=0.3)
        
        output = []
        if result.stdout:
            output.append(f"📤 STDOUT:\n{result.stdout}")
        if result.stderr:
            output.append(f"⚠️ STDERR:\n{result.stderr}")
        
        output.append(f"🔧 Command: {command}")
        output.append(f"📍 Directory: {working_directory}")
        output.append(f"🎯 Status: {status}")
        
        return "\n".join(output)
    except subprocess.TimeoutExpired:
        error_msg = f"⏰ Command timed out after 60 seconds: {command}"
        stream_text(error_msg, delay=0.02)
        return error_msg
    except Exception as e:
        error_msg = f"❌ Error executing command '{command}': {str(e)}"
        stream_text(error_msg, delay=0.02)
        return error_msg

def run_python_code(code: str, save_to_file: bool = False, filename: str = "temp_script.py") -> str:
    """Execute Python code directly or save to file and run."""
    try:
        if save_to_file:
            stream_text(f"📄 Saving code to {filename}...", delay=0.02)
            with open(filename, "w", encoding="utf-8") as f:
                f.write(code)
            return run_command(f"python {filename}")
        else:
            stream_text("🐍 Executing Python code...", delay=0.02)
            show_thinking_animation(0.8, "⚡ Processing")
            
            # Execute code directly using subprocess
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Progressive reveal of results
            sections = []
            
            if result.stdout:
                sections.append(("📤 OUTPUT:", result.stdout))
            if result.stderr:
                sections.append(("⚠️ ERRORS:", result.stderr))
            
            status = "✅ SUCCESS" if result.returncode == 0 else f"❌ ERROR (exit code: {result.returncode})"
            sections.append(("🎯 Status:", status))
            
            progressive_reveal(sections, section_delay=0.3)
            
            output = []
            if result.stdout:
                output.append(f"📤 OUTPUT:\n{result.stdout}")
            if result.stderr:
                output.append(f"⚠️ ERRORS:\n{result.stderr}")
            
            output.append(f"🎯 Status: {status}")
            
            return "\n".join(output)
    except subprocess.TimeoutExpired:
        error_msg = "⏰ Python code execution timed out after 30 seconds"
        stream_text(error_msg, delay=0.02)
        return error_msg
    except Exception as e:
        error_msg = f"❌ Error executing Python code: {str(e)}"
        stream_text(error_msg, delay=0.02)
        return error_msg

# ---------- PROJECT MANAGEMENT ----------

def create_project(project_name: str, project_type: str = "python") -> str:
    """Create a new project with basic structure."""
    try:
        if os.path.exists(project_name):
            return f"❌ Project directory {project_name} already exists"
        
        os.makedirs(project_name)
        
        if project_type.lower() == "python":
            # Create Python project structure
            structure = {
                f"{project_name}/main.py": "#!/usr/bin/env python3\n\nif __name__ == '__main__':\n    print('Hello, World!')\n",
                f"{project_name}/requirements.txt": "# Add your dependencies here\n",
                f"{project_name}/README.md": f"# {project_name}\n\nDescription of your project.\n\n## Installation\n\n```bash\npip install -r requirements.txt\n```\n\n## Usage\n\n```bash\npython main.py\n```\n",
                f"{project_name}/.gitignore": "__pycache__/\n*.pyc\n*.pyo\n*.pyd\n.Python\nbuild/\ndevelop-eggs/\ndist/\ndownloads/\neggs/\n.eggs/\nlib/\nlib64/\nparts/\nsdist/\nvar/\nwheels/\n*.egg-info/\n.installed.cfg\n*.egg\n"
            }
        elif project_type.lower() == "javascript" or project_type.lower() == "node":
            # Create Node.js project structure
            package_json = {
                "name": project_name,
                "version": "1.0.0",
                "description": "",
                "main": "index.js",
                "scripts": {"start": "node index.js"},
                "author": "",
                "license": "ISC"
            }
            structure = {
                f"{project_name}/index.js": "console.log('Hello, World!');\n",
                f"{project_name}/package.json": json.dumps(package_json, indent=2),
                f"{project_name}/README.md": f"# {project_name}\n\nDescription of your project.\n\n## Installation\n\n```bash\nnpm install\n```\n\n## Usage\n\n```bash\nnpm start\n```\n",
                f"{project_name}/.gitignore": "node_modules/\nnpm-debug.log*\n.npm\n.env\n"
            }
        else:
            # Generic project
            structure = {
                f"{project_name}/README.md": f"# {project_name}\n\nDescription of your project.\n"
            }
        
        for filepath, content in structure.items():
            write_file(filepath, content)
        
        return f"🚀 Created {project_type} project: {project_name}"
    except Exception as e:
        return f"❌ Error creating project {project_name}: {str(e)}"

def analyze_project(directory: str = ".") -> str:
    """Analyze project structure and provide insights."""
    try:
        stream_text("🔍 Starting project analysis...", delay=0.02)
        show_thinking_animation(1.5, "📊 Scanning files")
        
        analysis = []
        analysis.append(f"🔍 Project Analysis for: {os.path.abspath(directory)}")
        
        # Count files by extension
        file_counts = {}
        total_size = 0
        
        stream_text("\n📂 Scanning directory structure...", delay=0.02)
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                filepath = os.path.join(root, file)
                if os.path.isfile(filepath):
                    ext = os.path.splitext(file)[1].lower()
                    file_counts[ext] = file_counts.get(ext, 0) + 1
                    total_size += os.path.getsize(filepath)
        
        # Progressive reveal of analysis results
        sections = [
            ("📊 File Statistics:", f"Total files: {sum(file_counts.values())}\nTotal size: {total_size:,} bytes"),
        ]
        
        if file_counts:
            file_types_text = "File types:\n"
            for ext, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                ext_name = ext if ext else "(no extension)"
                file_types_text += f"  {ext_name}: {count} files\n"
            sections.append(("📄 File Types:", file_types_text))
        
        # Detect project type
        project_indicators = {
            "Python": ["requirements.txt", "setup.py", "pyproject.toml", "*.py"],
            "Node.js": ["package.json", "package-lock.json", "*.js", "*.ts"],
            "C#": ["*.csproj", "*.sln", "*.cs"],
            "Java": ["pom.xml", "build.gradle", "*.java"],
            "Go": ["go.mod", "go.sum", "*.go"],
            "Rust": ["Cargo.toml", "Cargo.lock", "*.rs"]
        }
        
        detected_types = []
        for proj_type, indicators in project_indicators.items():
            for indicator in indicators:
                if glob.glob(os.path.join(directory, "**", indicator), recursive=True):
                    detected_types.append(proj_type)
                    break
        
        if detected_types:
            sections.append(("🎯 Detected Technologies:", ", ".join(detected_types)))
        
        progressive_reveal(sections, section_delay=0.5)
        
        analysis.append(f"📊 Total files: {sum(file_counts.values())}")
        analysis.append(f"📏 Total size: {total_size:,} bytes")
        
        if file_counts:
            analysis.append("📄 File types:")
            for ext, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                ext_name = ext if ext else "(no extension)"
                analysis.append(f"  {ext_name}: {count} files")
        
        if detected_types:
            analysis.append(f"🎯 Detected project types: {', '.join(detected_types)}")
        
        return "\n".join(analysis)
    except Exception as e:
        error_msg = f"❌ Error analyzing project: {str(e)}"
        stream_text(error_msg, delay=0.02)
        return error_msg

# ---------- WEB AND DOCUMENTATION ----------

def open_documentation(topic: str) -> str:
    """Open documentation for a specific topic in the default browser."""
    try:
        docs_urls = {
            "python": "https://docs.python.org/3/",
            "javascript": "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
            "react": "https://reactjs.org/docs/",
            "django": "https://docs.djangoproject.com/",
            "flask": "https://flask.palletsprojects.com/",
            "fastapi": "https://fastapi.tiangolo.com/",
            "langchain": "https://python.langchain.com/docs/",
            "git": "https://git-scm.com/doc"
        }
        
        topic_lower = topic.lower()
        if topic_lower in docs_urls:
            webbrowser.open(docs_urls[topic_lower])
            return f"🌐 Opened documentation for {topic}"
        else:
            # Fallback to Google search
            search_url = f"https://www.google.com/search?q={quote(f'{topic} documentation')}"
            webbrowser.open(search_url)
            return f"🔍 Searched for {topic} documentation"
    except Exception as e:
        return f"❌ Error opening documentation: {str(e)}"

def generate_documentation(directory: str = ".", output_file: str = "API_DOCS.md") -> str:
    """Generate basic documentation for Python files in the project."""
    try:
        docs = ["# API Documentation\n\nAuto-generated documentation.\n"]
        
        for filepath in glob.glob(os.path.join(directory, "**", "*.py"), recursive=True):
            if os.path.isfile(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    # Extract functions and classes
                    functions = re.findall(r'^def\s+(\w+)\([^)]*\):', content, re.MULTILINE)
                    classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
                    
                    if functions or classes:
                        docs.append(f"\n## {filepath}\n")
                        
                        if classes:
                            docs.append("### Classes\n")
                            for cls in classes:
                                docs.append(f"- `{cls}`")
                        
                        if functions:
                            docs.append("\n### Functions\n")
                            for func in functions:
                                docs.append(f"- `{func}()`")
                
                except Exception:
                    continue
        
        doc_content = "\n".join(docs)
        write_file(output_file, doc_content)
        return f"📚 Generated documentation: {output_file}"
    except Exception as e:
        return f"❌ Error generating documentation: {str(e)}"

# ---------- SYSTEM INFORMATION ----------

def get_system_info() -> str:
    """Get comprehensive system information."""
    try:
        info = []
        info.append(f"💻 System: {platform.system()} {platform.release()}")
        info.append(f"🏗️ Architecture: {platform.machine()}")
        info.append(f"🐍 Python: {platform.python_version()}")
        info.append(f"📍 Current Directory: {os.getcwd()}")
        info.append(f"👤 User: {os.getenv('USER', os.getenv('USERNAME', 'Unknown'))}")
        
        # Disk space
        try:
            total, used, free = shutil.disk_usage(os.getcwd())
            info.append(f"💾 Disk Space: {free // (2**30)} GB free of {total // (2**30)} GB total")
        except Exception:
            pass
        
        # Environment variables (selected)
        important_env_vars = ['PATH', 'PYTHONPATH', 'HOME', 'USERPROFILE']
        for var in important_env_vars:
            if os.getenv(var):
                value = os.getenv(var)
                if len(value) > 100:
                    value = value[:100] + "..."
                info.append(f"🌍 {var}: {value}")
        
        return "\n".join(info)
    except Exception as e:
        return f"❌ Error getting system info: {str(e)}"

# ---------- TOOL DEFINITIONS ----------

# Create tools list only if LangChain is available
if LANGCHAIN_AVAILABLE:
    tools = [
        # File Operations
        StructuredTool.from_function(
            func=write_file,
            name="write_file",
            description="Create or overwrite a file with given contents. Args: filepath (str), contents (str)",
        ),
        StructuredTool.from_function(
            func=read_file,
            name="read_file",
            description="Read the contents of a file. Args: filepath (str)",
        ),
        StructuredTool.from_function(
            func=append_file,
            name="append_file",
            description="Append contents to a file. Args: filepath (str), contents (str)",
        ),
        StructuredTool.from_function(
            func=list_files,
            name="list_files",
            description="List files in a directory with optional pattern. Args: directory (str, default='.'), pattern (str, default='*')",
        ),
        StructuredTool.from_function(
            func=search_files,
            name="search_files",
            description="Search for text within files. Args: search_term (str), directory (str, default='.'), file_pattern (str, default='*.py')",
        ),
        StructuredTool.from_function(
            func=delete_file,
            name="delete_file",
            description="Delete a file or directory. Args: filepath (str)",
        ),
        StructuredTool.from_function(
            func=copy_file,
            name="copy_file",
            description="Copy a file or directory. Args: source (str), destination (str)",
        ),
        
        # Code Execution
        StructuredTool.from_function(
            func=run_command,
            name="run_command",
            description="Execute a shell command. Args: command (str), working_directory (str, default='.')",
        ),
        StructuredTool.from_function(
            func=run_python_code,
            name="run_python_code",
            description="Execute Python code directly or save to file. Args: code (str), save_to_file (bool, default=False), filename (str, default='temp_script.py')",
        ),
        
        # Project Management
        StructuredTool.from_function(
            func=create_project,
            name="create_project",
            description="Create a new project with basic structure. Args: project_name (str), project_type (str, default='python')",
        ),
        StructuredTool.from_function(
            func=analyze_project,
            name="analyze_project",
            description="Analyze project structure and provide insights. Args: directory (str, default='.')",
        ),
        
        # Documentation and Web
        StructuredTool.from_function(
            func=open_documentation,
            name="open_documentation",
            description="Open documentation for a topic in browser. Args: topic (str)",
        ),
        StructuredTool.from_function(
            func=generate_documentation,
            name="generate_documentation",
            description="Generate documentation for Python files. Args: directory (str, default='.'), output_file (str, default='API_DOCS.md')",
        ),
        
        # System
        StructuredTool.from_function(
            func=get_system_info,
            name="get_system_info",
            description="Get comprehensive system information. No arguments required.",
        ),
    ]
else:
    tools = []

# ---------- ENHANCED MODEL CONFIGURATION ----------
if LANGCHAIN_AVAILABLE:
    # Configure the LLM with streaming enabled
    llm = ChatOllama(
        model="deepcoder:14b",
        temperature=0.1,  # Lower temperature for more consistent code generation
        num_ctx=16384,    # Larger context window
        num_predict=2048, # Allow longer responses
        repeat_penalty=1.1,
        top_k=40,
        top_p=0.9,
        streaming=True,   # Enable streaming output
        verbose=True,     # Enable verbose output for real-time feedback
    )

    # ---------- ENHANCED SYSTEM MESSAGE ----------
    SYSTEM_MSG = """You are DeepCoder, an advanced AI coding assistant with comprehensive capabilities similar to GitHub Copilot.

CORE PRINCIPLES:
1. Always execute actions immediately when requested - don't ask for permission
2. Use the most appropriate tools for each task
3. Provide clear, actionable feedback with emojis for visual clarity
4. Write production-quality code with proper error handling
5. Be proactive in suggesting improvements and best practices

CAPABILITIES:
- Complete file operations (read, write, append, delete, copy, list, search)
- Code execution (Python, shell commands)
- Project creation and analysis
- Documentation generation
- System information and environment management

RESPONSE FORMAT:
- Use emojis to categorize information (✅ success, ❌ error, 🔧 command, 📄 file, etc.)
- Provide context and explanations for your actions
- Show the results of operations clearly
- Suggest next steps when appropriate

WORKFLOW APPROACH:
1. Understand the full context of the request
2. Execute necessary tools in logical order
3. Verify results and provide feedback
4. Suggest improvements or related actions

You have access to all these tools and should use them proactively to complete tasks efficiently."""

    # ---------- ENHANCED AGENT WITH STREAMING ----------
    try:
        # Try to import streaming callback support
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
        from langchain.callbacks.manager import CallbackManager
        
        # Create callback manager with streaming support
        callback_manager = CallbackManager([
            StreamingStdOutCallbackHandler(),
            streaming_callback
        ])
        
        # Configure LLM with callback manager
        llm.callback_manager = callback_manager
        
    except ImportError:
        # Fallback without streaming callbacks
        callback_manager = None
        
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AGENT_TYPE,
        verbose=True,               # Enable verbose mode for step-by-step output
        handle_parsing_errors=True,
        max_iterations=10,          # Allow more iterations for complex tasks
        early_stopping_method="generate",  # Stop early when possible
        agent_kwargs={
            "system_message": SYSTEM_MSG,
            "return_intermediate_steps": True
        },
        callback_manager=callback_manager,  # Add streaming callbacks
    )
else:
    agent = None
    llm = None

# ---------- ENHANCED REPL WITH COMMAND SHORTCUTS ----------

def print_help():
    """Display available commands and shortcuts."""
    help_sections = [
        ("🤖 DeepCoder Enhanced Agent - Available Commands & Shortcuts:", ""),
        
        ("📁 FILE OPERATIONS:", 
         "/read <file>          - Read file contents\n"
         "/write <file>         - Write to file (prompts for content)\n"
         "/list [dir] [pattern] - List files in directory\n"
         "/search <term>        - Search text in files\n"
         "/delete <file>        - Delete file/directory\n"
         "/copy <src> <dst>     - Copy file/directory"),
        
        ("🔧 CODE EXECUTION:",
         "/run <command>        - Execute shell command\n"
         "/python <code>        - Execute Python code\n"
         "/py <file>            - Run Python file"),
        
        ("🚀 PROJECT MANAGEMENT:",
         "/new <name> [type]    - Create new project (python/javascript/node)\n"
         "/analyze [dir]        - Analyze project structure\n"
         "/docs [topic]         - Open documentation\n"
         "/gendocs [dir]        - Generate project documentation"),
        
        ("💻 SYSTEM:",
         "/info                 - System information\n"
         "/pwd                  - Print working directory\n"
         "/cd <dir>             - Change directory\n"
         "/clear                - Clear screen"),
        
        ("⚡ SHORTCUTS:",
         "/help or /h           - Show this help\n"
         "/exit or /quit        - Exit the REPL"),
        
        ("💡 NATURAL LANGUAGE:",
         "Simply describe what you want to do, and I'll execute it using the appropriate tools!\n"
         "\nExamples:\n"
         "- \"Create a Python web scraper that extracts titles from a website\"\n"
         "- \"Analyze the current project and suggest improvements\"\n"
         "- \"Run tests in the current directory\"")
    ]
    
    progressive_reveal(help_sections, section_delay=0.4)

def handle_shortcut(command: str) -> Optional[str]:
    """Handle shortcut commands and return result or None if not a shortcut."""
    parts = command.strip().split()
    if not parts or not parts[0].startswith('/'):
        return None
    
    cmd = parts[0][1:].lower()  # Remove leading '/'
    args = parts[1:] if len(parts) > 1 else []
    
    try:
        if cmd in ['help', 'h']:
            print_help()
            return "📖 Help displayed"
        
        elif cmd in ['exit', 'quit']:
            return "EXIT"
        
        elif cmd == 'clear':
            os.system('cls' if os.name == 'nt' else 'clear')
            return "🧹 Screen cleared"
        
        elif cmd == 'pwd':
            return f"📍 Current directory: {os.getcwd()}"
        
        elif cmd == 'cd':
            if args:
                try:
                    os.chdir(args[0])
                    return f"📁 Changed directory to: {os.getcwd()}"
                except Exception as e:
                    return f"❌ Error changing directory: {str(e)}"
            else:
                return "❌ Usage: /cd <directory>"
        
        elif cmd == 'info':
            return get_system_info()
        
        elif cmd == 'read':
            if args:
                return read_file(args[0])
            else:
                return "❌ Usage: /read <filepath>"
        
        elif cmd == 'write':
            if args:
                print(f"📝 Writing to {args[0]}. Enter content (type 'EOF' on a new line to finish):")
                lines = []
                while True:
                    try:
                        line = input()
                        if line.strip() == 'EOF':
                            break
                        lines.append(line)
                    except KeyboardInterrupt:
                        return "❌ Write operation cancelled"
                content = '\n'.join(lines)
                return write_file(args[0], content)
            else:
                return "❌ Usage: /write <filepath>"
        
        elif cmd == 'list':
            directory = args[0] if args else "."
            pattern = args[1] if len(args) > 1 else "*"
            return list_files(directory, pattern)
        
        elif cmd == 'search':
            if args:
                term = ' '.join(args)
                return search_files(term)
            else:
                return "❌ Usage: /search <search_term>"
        
        elif cmd == 'delete':
            if args:
                return delete_file(args[0])
            else:
                return "❌ Usage: /delete <filepath>"
        
        elif cmd == 'copy':
            if len(args) >= 2:
                return copy_file(args[0], args[1])
            else:
                return "❌ Usage: /copy <source> <destination>"
        
        elif cmd == 'run':
            if args:
                command = ' '.join(args)
                return run_command(command)
            else:
                return "❌ Usage: /run <command>"
        
        elif cmd == 'python':
            if args:
                code = ' '.join(args)
                return run_python_code(code)
            else:
                return "❌ Usage: /python <python_code>"
        
        elif cmd == 'py':
            if args:
                return run_command(f"python {args[0]}")
            else:
                return "❌ Usage: /py <python_file>"
        
        elif cmd == 'new':
            if args:
                project_name = args[0]
                project_type = args[1] if len(args) > 1 else "python"
                return create_project(project_name, project_type)
            else:
                return "❌ Usage: /new <project_name> [type]"
        
        elif cmd == 'analyze':
            directory = args[0] if args else "."
            return analyze_project(directory)
        
        elif cmd == 'docs':
            if args:
                topic = ' '.join(args)
                return open_documentation(topic)
            else:
                return "❌ Usage: /docs <topic>"
        
        elif cmd == 'gendocs':
            directory = args[0] if args else "."
            return generate_documentation(directory)
        
        else:
            return f"❌ Unknown shortcut: /{cmd}. Type /help for available commands."
    
    except Exception as e:
        return f"❌ Error executing shortcut /{cmd}: {str(e)}"

# ---------- ENHANCED MAIN REPL ----------
if __name__ == "__main__":
    # Animated startup sequence
    stream_text("🤖 DeepCoder Enhanced Agent - Your AI Coding Assistant", delay=0.05)
    stream_text("=" * 60, delay=0.01)
    
    if not LANGCHAIN_AVAILABLE:
        sections = [
            ("⚠️ Status:", "Running in STANDALONE MODE (LangChain not available)"),
            ("💡 Setup:", "Install dependencies with: pip install -r requirements.txt"),
            ("🔧 Available:", "File operations, system commands, project tools")
        ]
    else:
        sections = [
            ("🚀 Status:", "Full AI Agent Mode - Natural language processing enabled"),
            ("💡 Usage:", "Type natural language commands or use shortcuts (type /help for shortcuts)"),
            ("🧠 Capabilities:", "I can handle files, run code, create projects, and much more!")
        ]
    
    progressive_reveal(sections, section_delay=0.3)
    
    stream_text("📝 Type '/exit' to quit", delay=0.02)
    stream_text("=" * 60, delay=0.01)
    
    # Show initial system info with animation
    try:
        stream_text("\n🔍 System Information:", delay=0.03)
        show_thinking_animation(1.0, "📊 Gathering system data")
        print("\n" + get_system_info())
    except:
        pass
    
    stream_text("\n" + "=" * 60 + "\n", delay=0.01)
    
    while True:
        try:
            if LANGCHAIN_AVAILABLE:
                prompt = "🤖 DeepCoder> "
            else:
                prompt = "🔧 DeepCoder (Standalone)> "
                
            cmd = input(prompt).strip()
            if not cmd:
                continue
            
            # Handle exit commands
            if cmd.lower() in ("exit", "quit", "/exit", "/quit"):
                stream_text("👋 Goodbye! Happy coding!", delay=0.05)
                break
            
            # Show processing indication
            show_thinking_animation(0.5, "🔄 Processing")
            
            # Try shortcut first
            shortcut_result = handle_shortcut(cmd)
            if shortcut_result:
                if shortcut_result == "EXIT":
                    stream_text("👋 Goodbye! Happy coding!", delay=0.05)
                    break
                else:
                    # Shortcut results are already streamed in their functions
                    if shortcut_result not in ["📖 Help displayed", "🧹 Screen cleared"]:
                        print()  # Add spacing
                    continue
            
            # Handle natural language commands with agent (if available)
            if LANGCHAIN_AVAILABLE and agent:
                # Show initial processing animation
                show_thinking_animation(0.5, "🧠 AI Starting")
                
                # Stream the response header
                stream_text("\n💡 AI Reasoning Process:", delay=0.03)
                stream_text("-" * 50, delay=0.01)
                
                start_time = time.time()
                
                try:
                    # Custom approach: Use a thread to monitor and stream output
                    import sys
                    from io import StringIO
                    import threading
                    
                    # Create custom stdout that streams in real-time
                    class StreamingStdout:
                        def __init__(self, original_stdout):
                            self.original = original_stdout
                            self.buffer = StringIO()
                            
                        def write(self, text):
                            self.original.write(text)  # Keep original output
                            if text.strip():  # Only process non-empty text
                                real_time_capture.add_output(text)
                            
                        def flush(self):
                            self.original.flush()
                    
                    # Start real-time capture
                    real_time_capture.start_streaming()
                    
                    # Replace stdout temporarily
                    original_stdout = sys.stdout
                    streaming_stdout = StreamingStdout(original_stdout)
                    sys.stdout = streaming_stdout
                    
                    try:
                        # Run the agent
                        result = agent.invoke({"input": cmd})
                    finally:
                        # Restore original stdout
                        sys.stdout = original_stdout
                        real_time_capture.stop_streaming()
                    
                    # Get the final output
                    output = result.get("output", result)
                    execution_time = time.time() - start_time
                    
                    # Stream the final answer with tokenized effect
                    stream_text(f"\n🎯 Final Answer ({execution_time:.2f}s):", delay=0.03)
                    stream_text("-" * 30, delay=0.01)
                    
                    # Apply tokenized streaming only to the final response
                    stream_multiline(str(output), line_delay=0.02, char_delay=0.03)
                    
                    stream_text("-" * 50 + "\n", delay=0.01)
                    
                except Exception as e:
                    error_sections = [
                        ("❌ AI Agent Error:", str(e)),
                        ("💡 Suggestion:", "Try using a shortcut command (type /help) or rephrase your request.")
                    ]
                    progressive_reveal(error_sections, section_delay=0.3)
                    if "--verbose" in cmd:
                        traceback.print_exc()
            else:
                warning_sections = [
                    ("⚠️ Notice:", "Natural language processing not available in standalone mode."),
                    ("💡 Available:", "Use shortcut commands (type /help) or install dependencies with:"),
                    ("📦 Install:", "pip install -r requirements.txt")
                ]
                progressive_reveal(warning_sections, section_delay=0.3)

        except KeyboardInterrupt:
            stream_text("\n⚠️ Interrupted. Type 'exit' to quit or continue with another command.", delay=0.02)
            continue
        except EOFError:
            stream_text("\n👋 Goodbye! Happy coding!", delay=0.05)
            break
        except Exception as e:
            error_sections = [
                ("❌ Unexpected Error:", str(e)),
                ("💡 Suggestion:", "Please try again or use /help for available commands.")
            ]
            progressive_reveal(error_sections, section_delay=0.3)
            if "--verbose" in str(e):
                traceback.print_exc()
