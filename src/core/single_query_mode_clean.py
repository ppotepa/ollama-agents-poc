"""Single query mode for command-line usage - uses proper agent implementations."""

import os
import sys
from typing import Optional, Dict, Any
from pathlib import Path


def get_agent_instance(agent_name: str):
    """Get the appropriate agent instance based on the agent name."""
    try:
        # Add parent directory to path for imports
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        
        from integrations.model_config_reader import ModelConfigReader
        
        # Load the model configuration to get agent details
        config_reader = ModelConfigReader('src/config/models.yaml')
        model_config = config_reader.get_model(agent_name)
        
        if not model_config:
            # Try to find by model ID or partial match
            for model in config_reader.get_all_models():
                if (agent_name in model.model_id or 
                    model.model_id.startswith(agent_name) or
                    agent_name in model.short_name):
                    model_config = model
                    break
        
        if not model_config:
            raise ValueError(f"No configuration found for agent '{agent_name}'")
        
        # Convert model config to agent config format
        agent_config = {
            "name": model_config.name,
            "model_id": model_config.model_id,
            "backend_image": model_config.model_id,
            "parameters": model_config.parameters,
            "tools": model_config.tools,
            "system_message": model_config.system_message,
            "supports_coding": model_config.supports_coding
        }
        
        # Determine which agent implementation to use based on agent type
        if agent_name.lower() in ['deepcoder', 'coder'] or model_config.supports_coding:
            # Use the sophisticated DeepCoderAgent for coding agents
            try:
                from src.agents.deepcoder.agent import create_agent
                return create_agent(agent_name, agent_config)
            except ImportError:
                print(f"⚠️  Warning: DeepCoderAgent not available, falling back to simple agent")
                return SimpleQueryAgent(agent_name, agent_config)
        else:
            # Use simple agent for non-coding agents
            return SimpleQueryAgent(agent_name, agent_config)
            
    except Exception as e:
        print(f"⚠️  Warning: Could not load proper agent, using fallback: {e}")
        # Create a basic fallback config
        fallback_config = {
            'name': agent_name.title(),
            'model_id': agent_name,
            'backend_image': agent_name,
            'parameters': {'temperature': 0.7, 'num_ctx': 8192},
            'tools': [],
            'system_message': "You are an AI assistant.",
            'supports_coding': True
        }
        return SimpleQueryAgent(agent_name, fallback_config)


class SimpleQueryAgent:
    """Simple agent for handling single queries when advanced agents aren't available."""
    
    def __init__(self, agent_name: str, config: Dict[str, Any]):
        """Initialize the simple query agent."""
        self.agent_name = agent_name
        self.config = config
        self.agent_id = agent_name
    
    def _choose_optimal_model(self, query: str, original_model: str) -> str:
        """Choose the best model based on query complexity."""
        # Simple queries that can use fast models
        simple_patterns = [
            'what is', 'calculate', 'add', 'subtract', 'multiply', 'divide',
            '+', '-', '*', '/', '=', 'math', 'number', 'sum', 'total'
        ]
        
        query_lower = query.lower()
        is_simple = any(pattern in query_lower for pattern in simple_patterns)
        is_short = len(query) < 100
        
        if is_simple and is_short:
            # Use fast model for simple queries
            return "tinyllama:latest"
        else:
            # Use original model for complex queries
            return original_model
    
    def run_query(self, query: str) -> str:
        """Run a single query using optimized approach."""
        try:
            print(f"🔍 Debug: Starting run_query with query: {query[:50]}...")
            
            # Get model configuration
            model_config = self.config
            original_model_id = model_config.get('model_id', self.agent_name)
            
            # Choose optimal model for performance
            model_id = self._choose_optimal_model(query, original_model_id)
            
            print(f"🔍 Debug: Using model {model_id} (original: {original_model_id}) with optimized params")
            
            # Try direct Ollama API first for better performance
            try:
                print(f"🔍 Debug: Attempting direct Ollama API call...")
                import requests
                import json
                
                # Create a simple prompt without complex system messages for speed
                prompt = f"User: {query}\nAssistant:"
                
                print(f"🔍 Debug: Making POST request to Ollama...")
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model_id,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,  # Lower temperature for faster, more consistent responses
                            "num_ctx": 1024,     # Much smaller context for speed
                            "num_predict": 128,  # Much smaller response for speed
                        }
                    },
                    timeout=60  # Reasonable timeout
                )
                
                print(f"🔍 Debug: Received response with status {response.status_code}")
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get('response', '').strip()
                    print(f"🔍 Debug: Direct API success, response length: {len(answer)}")
                    return answer
                else:
                    # Fallback to LangChain if direct API fails
                    raise Exception(f"Direct API failed with status {response.status_code}")
                    
            except Exception as direct_api_error:
                # Fallback to LangChain approach
                print(f"🔍 Debug: Direct API failed ({direct_api_error}), falling back to LangChain...")
                
                try:
                    from langchain_ollama import ChatOllama
                    from langchain.schema import SystemMessage, HumanMessage
                    
                    print(f"🔍 Debug: Invoking LangChain LLM...")
                    llm = ChatOllama(
                        model=model_id,
                        temperature=0.3,
                        num_ctx=1024
                    )
                    
                    # Create messages
                    messages = []
                    system_msg = model_config.get('system_message')
                    if system_msg:
                        messages.append(SystemMessage(content=system_msg))
                    messages.append(HumanMessage(content=query))
                    
                    # Get response
                    response = llm.invoke(messages)
                    return response.content.strip()
                    
                except Exception as langchain_error:
                    print(f"🔍 Debug: Error in run_query: {langchain_error}")
                    return f"❌ Error processing query: {langchain_error}"
                    
        except Exception as e:
            print(f"🔍 Debug: Error in run_query: {e}")
            return f"❌ Error processing query: {e}"


# Public functions for main.py
def run_single_query(query: str, agent_name: str) -> str:
    """Run a single query using the appropriate agent implementation."""
    try:
        print(f"🔍 Debug: Starting run_single_query for agent '{agent_name}'")
        
        # Get the appropriate agent instance
        agent = get_agent_instance(agent_name)
        
        print(f"🔍 Debug: Agent initialized, running query...")
        
        # Use proper agent method based on type
        if hasattr(agent, 'stream') and hasattr(agent, 'load'):
            # This is a proper agent (like DeepCoderAgent)
            agent.load()  # Load the agent
            
            # For single query mode, we'll collect the streamed response
            result_parts = []
            def collect_token(token):
                result_parts.append(token)
            
            agent.stream(query, collect_token)
            result = ''.join(result_parts)
        else:
            # This is a simple agent
            result = agent.run_query(query)
        
        print(f"🔍 Debug: Query completed, returning result")
        return result
    except Exception as e:
        print(f"🔍 Debug: Error in run_single_query: {e}")
        return f"❌ Error initializing agent '{agent_name}': {e}"


# Direct execution support
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        agent_name = sys.argv[1]
        query = sys.argv[2]
        result = run_single_query(query, agent_name)
        print(result)
    else:
        print("Usage: python single_query_mode.py <agent_name> <query>")
