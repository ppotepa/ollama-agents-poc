"""Enhanced logging utilities for command interceptor and agents."""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class EnhancedLogger:
    """Enhanced logging system for agent operations and collaboration."""
    
    def __init__(self, log_dir: str = "logs", enable_console: bool = True):
        """Initialize enhanced logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.enable_console = enable_console
        
        # Create specialized loggers
        self.agent_logger = self._setup_logger("agent", "agent_operations.log")
        self.collaboration_logger = self._setup_logger("collaboration", "collaboration.log")
        self.command_logger = self._setup_logger("commands", "command_execution.log")
        self.debug_logger = self._setup_logger("debug", "debug.log")
        
    def _setup_logger(self, name: str, filename: str) -> logging.Logger:
        """Setup individual logger with file and console handlers."""
        logger = logging.getLogger(f"ollama_agents.{name}")
        logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(self.log_dir / filename, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler (if enabled)
        if self.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Different formats for different loggers
            if name == "debug":
                console_handler.setLevel(logging.DEBUG)
        
        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        if self.enable_console:
            console_handler.setFormatter(simple_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def log_agent_operation(self, agent_name: str, operation: str, details: Dict[str, Any]):
        """Log agent operation with structured data."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "operation": operation,
            "details": details
        }
        
        self.agent_logger.info(f"Agent '{agent_name}' - {operation}")
        self.agent_logger.debug(f"Details: {json.dumps(details, indent=2)}")
        
        # Save structured log entry
        self._save_structured_log("agent_operations", log_entry)
    
    def log_collaboration_step(self, step_type: str, main_agent: str, interceptor: str, 
                             command: str, result: Optional[str] = None):
        """Log collaboration step between agents."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step_type": step_type,
            "main_agent": main_agent,
            "interceptor": interceptor,
            "command": command,
            "result": result
        }
        
        self.collaboration_logger.info(
            f"Collaboration: {step_type} - {main_agent} + {interceptor}"
        )
        self.collaboration_logger.debug(f"Command: {command}")
        
        if result:
            self.collaboration_logger.debug(f"Result: {result[:200]}...")
        
        self._save_structured_log("collaboration", log_entry)
    
    def log_command_execution(self, command: str, agent: str, success: bool, 
                            output: str, duration: float):
        """Log command execution details."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "agent": agent,
            "success": success,
            "output": output,
            "duration_seconds": duration
        }
        
        status = "SUCCESS" if success else "FAILED"
        self.command_logger.info(f"Command {status}: {command} (Agent: {agent})")
        self.command_logger.debug(f"Duration: {duration:.2f}s")
        
        if not success:
            self.command_logger.error(f"Error output: {output}")
        
        self._save_structured_log("command_execution", log_entry)
    
    def log_interceptor_analysis(self, prompt: str, analysis: Dict[str, Any], 
                               recommendations: List[str]):
        """Log interceptor analysis results."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "analysis": analysis,
            "recommendations": recommendations
        }
        
        self.collaboration_logger.info("Interceptor analysis completed")
        self.collaboration_logger.debug(f"Recommendations: {recommendations}")
        
        self._save_structured_log("interceptor_analysis", log_entry)
    
    def log_model_compatibility(self, model: str, supports_tools: bool, 
                              capabilities: List[str], error_details: Optional[str] = None):
        """Log model compatibility information."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "supports_tools": supports_tools,
            "capabilities": capabilities,
            "error_details": error_details
        }
        
        status = "TOOLS SUPPORTED" if supports_tools else "LLM-ONLY MODE"
        self.agent_logger.info(f"Model '{model}' - {status}")
        self.agent_logger.debug(f"Capabilities: {capabilities}")
        
        if error_details:
            self.agent_logger.debug(f"Tool detection error: {error_details}")
        
        self._save_structured_log("model_compatibility", log_entry)
    
    def debug(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log debug message with optional context."""
        self.debug_logger.debug(message)
        if context:
            self.debug_logger.debug(f"Context: {json.dumps(context, indent=2)}")
    
    def info(self, message: str):
        """Log info message."""
        self.agent_logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.agent_logger.warning(message)
    
    def error(self, message: str, exception: Optional[Exception] = None):
        """Log error message with optional exception."""
        self.agent_logger.error(message)
        if exception:
            self.agent_logger.exception(f"Exception details: {exception}")
    
    def _save_structured_log(self, log_type: str, entry: Dict[str, Any]):
        """Save structured log entry to JSON file."""
        log_file = self.log_dir / f"{log_type}.jsonl"
        
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            self.debug_logger.error(f"Failed to save structured log: {e}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session logs."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "agent_operations": self._count_log_entries("agent_operations"),
            "collaboration_steps": self._count_log_entries("collaboration"),
            "commands_executed": self._count_log_entries("command_execution"),
            "errors": self._count_log_entries("debug", level="ERROR")
        }
        
        return summary
    
    def _count_log_entries(self, log_type: str, level: Optional[str] = None) -> int:
        """Count entries in structured log file."""
        log_file = self.log_dir / f"{log_type}.jsonl"
        
        if not log_file.exists():
            return 0
        
        try:
            count = 0
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if level:
                        entry = json.loads(line.strip())
                        if entry.get('level') == level:
                            count += 1
                    else:
                        count += 1
            return count
        except Exception:
            return 0
    
    def clear_logs(self):
        """Clear all log files for new session."""
        for log_file in self.log_dir.glob("*.log"):
            log_file.unlink(missing_ok=True)
        
        for log_file in self.log_dir.glob("*.jsonl"):
            log_file.unlink(missing_ok=True)
        
        self.info("Log files cleared for new session")


# Global logger instance
enhanced_logger = None


def get_logger(enable_console: bool = True) -> EnhancedLogger:
    """Get or create global enhanced logger instance."""
    global enhanced_logger
    
    if enhanced_logger is None:
        enhanced_logger = EnhancedLogger(enable_console=enable_console)
    
    return enhanced_logger


def log_agent_start(agent_name: str, model: str, tools_enabled: bool):
    """Convenience function to log agent startup."""
    logger = get_logger()
    logger.log_agent_operation(
        agent_name=agent_name,
        operation="startup",
        details={
            "model": model,
            "tools_enabled": tools_enabled,
            "timestamp": datetime.now().isoformat()
        }
    )


def log_collaborative_session(main_agent: str, interceptor: str, 
                            commands_executed: int, success_rate: float):
    """Convenience function to log collaborative session summary."""
    logger = get_logger()
    logger.log_collaboration_step(
        step_type="session_summary",
        main_agent=main_agent,
        interceptor=interceptor,
        command=f"Executed {commands_executed} commands",
        result=f"Success rate: {success_rate:.1%}"
    )
