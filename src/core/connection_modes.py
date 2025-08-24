"""Connection modes for different deployment scenarios."""

import os
import sys
import time
import threading
import subprocess
import requests
from typing import Optional, Dict, Any
from pathlib import Path


class ConnectionMode:
    """Base class for connection modes."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.server_url = config.get('server_url', 'http://localhost:8000')
        self.ollama_url = config.get('ollama_url', 'http://localhost:11434')
    
    def connect(self) -> bool:
        """Establish connection. Returns True if successful."""
        raise NotImplementedError
    
    def disconnect(self):
        """Clean up connection."""
        pass
    
    def is_available(self) -> bool:
        """Check if the connection is available."""
        raise NotImplementedError


class DirectMode(ConnectionMode):
    """Direct mode - starts server temporarily for single queries."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.server_process = None
        self.server_port = config.get('temp_server_port', 8001)  # Use different port to avoid conflicts
        self.server_host = config.get('temp_server_host', '127.0.0.1')
        self.startup_timeout = config.get('startup_timeout', 10)
        self.server_url = f"http://{self.server_host}:{self.server_port}"
    
    def connect(self) -> bool:
        """Start temporary server in background."""
        try:
            print(f"🚀 Starting temporary server on {self.server_url}...")
            
            # Get the main.py path
            script_dir = Path(__file__).parent.parent.parent  # Go up to project root
            main_py = script_dir / "main.py"
            
            if not main_py.exists():
                print(f"❌ Error: main.py not found at {main_py}")
                return False
            
            # Start server process
            cmd = [
                sys.executable, str(main_py),
                "--server",
                "--host", self.server_host,
                "--port", str(self.server_port)
            ]
            
            print(f"🔍 Debug: Starting command: {' '.join(cmd)}")
            
            # Start in background with minimal output
            # Set encoding environment variables to handle Unicode characters on Windows
            env = dict(os.environ)
            env.update({
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8"  # Force UTF-8 encoding
            })
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(script_dir),
                env=env,
                text=True,  # Ensure text mode for better error handling
                encoding='utf-8',  # Explicitly set encoding
                errors='replace'  # Replace problematic characters instead of failing
            )
            
            # Wait for server to start
            print(f"⏳ Waiting for server to start (timeout: {self.startup_timeout}s)...")
            
            for i in range(self.startup_timeout * 2):  # Check every 0.5 seconds
                if self.server_process.poll() is not None:
                    # Process died - get the error output
                    try:
                        stdout, stderr = self.server_process.communicate(timeout=1)
                        error_msg = stderr.strip() if stderr else stdout.strip() if stdout else "Unknown error"
                        print(f"❌ Server process died with exit code {self.server_process.returncode}")
                        print(f"❌ Error output: {error_msg}")
                    except subprocess.TimeoutExpired:
                        print(f"❌ Server process died (exit code: {self.server_process.returncode})")
                    return False
                
                if self._check_server_health():
                    print(f"✅ Temporary server ready at {self.server_url}")
                    return True
                
                time.sleep(0.5)
                if i % 4 == 0:  # Print dots every 2 seconds
                    print(".", end="", flush=True)
            
            print(f"\n❌ Server failed to start within {self.startup_timeout}s")
            self.disconnect()
            return False
            
        except Exception as e:
            print(f"❌ Error starting temporary server: {e}")
            self.disconnect()
            return False
    
    def disconnect(self):
        """Stop the temporary server."""
        if self.server_process:
            print(f"🛑 Stopping temporary server...")
            try:
                self.server_process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("⚠️  Server didn't stop gracefully, forcing termination...")
                    self.server_process.kill()
                    self.server_process.wait()
                
                print("✅ Temporary server stopped")
            except Exception as e:
                print(f"⚠️  Error stopping server: {e}")
            finally:
                self.server_process = None
    
    def is_available(self) -> bool:
        """Check if temporary server is running."""
        return self.server_process is not None and self._check_server_health()
    
    def _check_server_health(self) -> bool:
        """Check if server is responding."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=2)
            return response.status_code == 200
        except:
            try:
                # Try models endpoint as fallback
                response = requests.get(f"{self.server_url}/v1/models", timeout=2)
                return response.status_code == 200
            except:
                return False


class RequestMode(ConnectionMode):
    """Request mode - uses existing running server."""
    
    def connect(self) -> bool:
        """Check if server is available."""
        if self.is_available():
            print(f"✅ Connected to existing server at {self.server_url}")
            return True
        else:
            print(f"❌ Server not available at {self.server_url}")
            print(f"💡 Hint: Make sure the server is running with: python main.py --server")
            print(f"💡 Or use docker-compose: docker-compose up")
            return False
    
    def disconnect(self):
        """No cleanup needed for request mode."""
        pass
    
    def is_available(self) -> bool:
        """Check if external server is responding."""
        try:
            # First try health endpoint
            response = requests.get(f"{self.server_url}/health", timeout=3)
            if response.status_code == 200:
                return True
        except:
            pass
        
        try:
            # Fallback to models endpoint
            response = requests.get(f"{self.server_url}/v1/models", timeout=3)
            return response.status_code == 200
        except:
            pass
        
        return False


class HybridMode(ConnectionMode):
    """Hybrid mode - tries request mode first, falls back to direct mode."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.request_mode = RequestMode(config)
        self.direct_mode = DirectMode(config)
        self.active_mode = None
    
    def connect(self) -> bool:
        """Try request mode first, then direct mode."""
        print("🔄 Trying hybrid connection (request mode first, then direct mode)...")
        
        # Try request mode first
        if self.request_mode.connect():
            self.active_mode = self.request_mode
            self.server_url = self.request_mode.server_url
            return True
        
        print("🔄 Request mode failed, trying direct mode...")
        
        # Fall back to direct mode
        if self.direct_mode.connect():
            self.active_mode = self.direct_mode
            self.server_url = self.direct_mode.server_url
            return True
        
        print("❌ Both request and direct modes failed")
        return False
    
    def disconnect(self):
        """Disconnect the active mode."""
        if self.active_mode:
            self.active_mode.disconnect()
            self.active_mode = None
    
    def is_available(self) -> bool:
        """Check if active mode is available."""
        return self.active_mode is not None and self.active_mode.is_available()


def get_connection_mode(mode_name: str, config: Optional[Dict[str, Any]] = None) -> ConnectionMode:
    """Factory function to create connection modes."""
    if config is None:
        config = {}
    
    mode_name = mode_name.lower()
    
    if mode_name == "direct":
        return DirectMode(config)
    elif mode_name == "request":
        return RequestMode(config)
    elif mode_name == "hybrid":
        return HybridMode(config)
    else:
        raise ValueError(f"Unknown connection mode: {mode_name}. Use 'direct', 'request', or 'hybrid'")


def detect_best_mode(config: Optional[Dict[str, Any]] = None) -> str:
    """Detect the best connection mode based on environment."""
    if config is None:
        config = {}
    
    # Check if server is already running
    request_mode = RequestMode(config)
    if request_mode.is_available():
        return "request"
    
    # Check if we can start a temporary server
    return "direct"  # Default to direct mode
