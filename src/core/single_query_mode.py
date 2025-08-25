"""Single query mode for command-line usage - uses proper agent implementations."""

import os
import sys
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path


def run_intelligent_investigation(query: str, streaming: bool = True) -> str:
    """Run a query using the intelligent orchestrator for complex investigations."""
    try:
        from src.core.intelligent_orchestrator import get_orchestrator, ExecutionMode
        from src.core.intelligence_enhancer import apply_intelligence_improvements, is_enhanced_investigation_enabled
        
        print(f"🧠 Starting intelligent investigation")
        print(f"🔍 Query: {query[:100]}...")
        print(f"🎬 Streaming: {streaming}")
        print("=" * 60)
        
        # Get the orchestrator instance
        orchestrator = get_orchestrator(enable_streaming=streaming)
        
        # Apply intelligence improvements if enabled
        if is_enhanced_investigation_enabled():
            print("🚀 Using enhanced investigation capabilities")
            apply_intelligence_improvements(orchestrator)
        
        # Run the investigation asynchronously
        async def run_investigation():
            session_id = await orchestrator.start_investigation(
                query=query,
                execution_mode=ExecutionMode.ADAPTIVE,
                context={"working_directory": os.getcwd()}
            )
            
            # Wait for completion and get results
            max_wait_time = 600  # 10 minutes
            wait_interval = 5  # 5 seconds
            elapsed_time = 0
            
            while elapsed_time < max_wait_time:
                status = orchestrator.get_session_status(session_id)
                if status:
                    state = status["session"]["current_state"]
                    if state == "completed":
                        print(f"✅ Investigation completed successfully")
                        
                        # Get final results from context
                        context = orchestrator.context_manager.get_context(session_id)
                        if context:
                            # Compile final result from execution history
                            final_results = []
                            for record in context.execution_history:
                                if record.get("result") and "ERROR" not in record.get("result", ""):
                                    final_results.append(record["result"])
                            
                            if final_results:
                                return "\n\n".join(final_results[-3:])  # Last 3 successful results
                            else:
                                return "Investigation completed but no results found."
                        else:
                            return "Investigation completed successfully."
                    
                    elif state == "error":
                        print(f"❌ Investigation failed")
                        return "Investigation failed due to an error."
                    
                    else:
                        # Print progress
                        progress = status.get("progress", {})
                        completed = progress.get("completed_steps", 0)
                        total = progress.get("total_steps", 0)
                        if total > 0:
                            percentage = progress.get("completion_percentage", 0)
                            print(f"🔄 Progress: {completed}/{total} steps ({percentage:.1f}%)")
                
                await asyncio.sleep(wait_interval)
                elapsed_time += wait_interval
            
            # Timeout reached
            print(f"⏱️  Investigation timed out after {max_wait_time} seconds")
            await orchestrator.pause_session(session_id)
            return "Investigation timed out. Please try a more specific query."
        
        # Run the async investigation
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        return asyncio.run(run_investigation())
        
    except Exception as e:
        print(f"❌ Error in intelligent investigation: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to collaborative mode
        print(f"🔄 Falling back to collaborative mode")
        return run_collaborative_query(query, "universal", 5, streaming)


def run_collaborative_query(query: str, agent_name: str, max_iterations: int = 5, streaming: bool = True) -> str:
    """Run a query using collaborative back-and-forth between main agent and interceptor."""
    try:
        from src.core.collaborative_system import create_collaborative_system
        from src.core.helpers import get_agent_instance
        
        print(f"🤝 Starting collaborative query execution")
        print(f"🤖 Main agent: {agent_name}")
        print(f"🔄 Max iterations: {max_iterations}")
        print(f"🎬 Streaming: {streaming}")
        print("=" * 60)
        
        # Get the main agent instance with streaming configuration
        main_agent = get_agent_instance(agent_name, streaming=streaming)
        
        # Create collaborative system
        collaborative_system = create_collaborative_system(main_agent, max_iterations)
        
        # Execute collaborative query
        results = collaborative_system.collaborative_execution(
            query=query,
            working_directory=os.getcwd(),
            max_steps=max_iterations
        )
        
        if results["success"]:
            print(f"✅ Collaborative execution completed successfully")
            print(f"📊 Total steps: {results['total_steps']}")
            print(f"🔧 Commands executed: {', '.join(results['commands_executed'])}")
            print(f"📁 Files discovered: {len(results['files_discovered'])}")
            return results["final_answer"]
        else:
            return f"❌ Collaborative execution failed"
            
    except Exception as e:
        print(f"❌ Error in collaborative mode: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to regular single query
        print(f"🔄 Falling back to regular single query mode")
        return run_single_query(query, agent_name, collaborative=False, force_streaming=streaming)


def run_single_query(query: str, agent_name: str, connection_mode: str = "hybrid", repository_url: str = None, interception_mode: str = "smart", force_streaming: bool = False, collaborative: bool = False, max_iterations: int = 5, intelligent_investigation: bool = False, no_tools: bool = False) -> str:
    """Run a single query with cognitive command interpretation and connection mode support.
    
    Args:
        collaborative: If True, use iterative collaboration between main agent and interceptor
        max_iterations: Maximum number of collaborative iterations (only used if collaborative=True)
        intelligent_investigation: If True, use intelligent orchestrator for complex investigation
    """
    connection = None
    try:
        print(f"🔍 Debug: Starting run_single_query for agent '{agent_name}' with {connection_mode} mode")
        
        # **INTELLIGENT INVESTIGATION MODE**: Use the intelligent orchestrator for complex investigations
        if intelligent_investigation:
            print(f"🧠 Enabling intelligent investigation mode")
            return run_intelligent_investigation(query, force_streaming)
        
        # **COLLABORATIVE MODE**: Use iterative back-and-forth between agents
        if collaborative:
            print(f"🤝 Enabling collaborative mode with {max_iterations} max iterations")
            return run_collaborative_query(query, agent_name, max_iterations, force_streaming)
        
        # **PROMPT INTERCEPTION**: Intercept and enhance the prompt with contextual information
        print(f"🧠 Intercepting and analyzing prompt...")
        from src.core.prompt_interceptor import intercept_and_enhance_prompt, InterceptionMode
        
        # Convert string mode to enum
        mode_map = {
            "full": InterceptionMode.FULL,
            "lightweight": InterceptionMode.LIGHTWEIGHT,
            "smart": InterceptionMode.SMART
        }
        mode = mode_map.get(interception_mode.lower(), InterceptionMode.SMART)
        
        try:
            supplemented = intercept_and_enhance_prompt(query, os.getcwd(), repository_url, mode)
            
            # Show what was detected and enhanced
            if supplemented.context_used:
                print(f"🎯 Detected intent: {supplemented.metadata.get('detected_intent', 'Unknown')}")
                print(f"📊 Confidence: {supplemented.metadata.get('confidence', 0):.2f}")
                print(f"🔧 Context added: {', '.join(supplemented.context_used)}")
                
                # Use the supplemented prompt instead of the original
                enhanced_query = supplemented.supplemented_prompt
                print(f"✨ Prompt enhanced with contextual information")
                
                # **CONFIRM WHAT'S BEING SENT TO AGENT**
                print(f"\n🎯 SENDING TO AGENT:")
                print(f"📏 Enhanced query length: {len(enhanced_query)} characters")
                print(f"🔤 First 200 chars: {enhanced_query}...")                
                print()
                
                # Store interceptor data for agent context
                interceptor_data = {
                    'detected_intent': supplemented.metadata.get('detected_intent', 'Unknown'),
                    'confidence': supplemented.metadata.get('confidence', 0),
                    'context_types': supplemented.context_used,
                    'commands_executed': [
                        {
                            'command': cmd.command_name,
                            'duration': cmd.duration,
                            'success': cmd.success,
                            'result_length': cmd.result_length,
                            'error': cmd.error_message
                        }
                        for cmd in supplemented.commands_executed
                    ],
                    'execution_stats': {
                        'total_commands': len(supplemented.commands_executed),
                        'successful_commands': len([cmd for cmd in supplemented.commands_executed if cmd.success]),
                        'total_execution_time': sum(cmd.duration for cmd in supplemented.commands_executed),
                        'total_data_gathered': sum(cmd.result_length for cmd in supplemented.commands_executed if cmd.success)
                    }
                }
            else:
                enhanced_query = query
                print(f"📝 No contextual enhancement applied")
                interceptor_data = None
                
        except Exception as e:
            print(f"⚠️  Prompt interception failed: {e}")
            enhanced_query = query
            interceptor_data = None
        
        # Initialize connection mode
        from src.core.connection_modes import get_connection_mode
        
        connection_config = {
            'server_url': 'http://localhost:8000',
            'ollama_url': 'http://localhost:11434',
            'temp_server_port': 8001,
            'startup_timeout': 15
        }
        
        connection = get_connection_mode(connection_mode, connection_config)
        
        # Try to establish connection
        if not connection.connect():
            print(f"❌ Failed to establish {connection_mode} connection")
            # Fall back to direct Ollama connection
            return run_single_query_direct_ollama(query, agent_name, streaming=force_streaming)
        
        print(f"✅ Connection established via {connection_mode} mode")
        
        # First, try to resolve the query using CommandResolver (skip if forcing streaming)
        if not force_streaming:
            from src.core.command_resolver import resolve_user_input
            
            print(f"🧠 Attempting cognitive command resolution...")
            resolved_command = resolve_user_input(enhanced_query)
            
            if resolved_command:
                print(f"🧠 Command resolved: {resolved_command.command_name} (confidence: {resolved_command.confidence:.2f})")
                
                if resolved_command.confidence >= 0.6:  # Lowered threshold for better coverage
                    print(f"📋 Description: {resolved_command.description}")
                    
                    # Try to execute the resolved command
                    try:
                        result = execute_resolved_command(resolved_command, enhanced_query)
                        if result:
                            return result
                    except Exception as e:
                        print(f"⚠️  Command execution failed, falling back to agent: {e}")
                else:
                    print(f"⚠️  Confidence too low ({resolved_command.confidence:.2f}), falling back to agent")
            else:
                print(f"🧠 No command resolved, falling back to agent")
        else:
            print(f"🎬 Skipping command resolution - forcing agent streaming")
        
        # Fall back to agent-based processing
        if force_streaming:
            print(f"🎬 Forcing direct streaming connection...")
            return run_single_query_direct_ollama(enhanced_query, agent_name, interceptor_data, streaming=force_streaming, no_tools=no_tools)
        
        # Use server-based processing (default)
        print(f"🤖 Using server-based agent processing for enhanced query")
        
        try:
            # Use the server API for processing with enhanced query
            result = query_via_server(enhanced_query, agent_name, connection.server_url)
            if result:
                return result
        except Exception as e:
            print(f"⚠️  Server-based processing failed: {e}")
        
        # Final fallback to direct Ollama with enhanced query
        print(f"🔄 Falling back to direct Ollama connection...")
        return run_single_query_direct_ollama(enhanced_query, agent_name, streaming=force_streaming, no_tools=no_tools)
        
    except Exception as e:
        print(f"❌ Error in run_single_query: {e}")
        # Use enhanced query if available, otherwise use original
        fallback_query = enhanced_query if 'enhanced_query' in locals() else query
        fallback_interceptor_data = interceptor_data if 'interceptor_data' in locals() else None
        return run_single_query_direct_ollama(fallback_query, agent_name, fallback_interceptor_data, streaming=force_streaming, no_tools=no_tools)
    finally:
        # Clean up connection
        if connection:
            connection.disconnect()


def query_via_server(query: str, agent_name: str, server_url: str) -> str:
    """Query the agent via our native server API."""
    try:
        import requests
        
        # Use our native agent query endpoint instead of chat completions
        payload = {
            "agent": agent_name,
            "query": query
        }
        
        endpoint = f"{server_url}/v1/agents/query"
        print(f"🔍 Debug: Sending request to {endpoint}")
        
        response = requests.post(
            endpoint,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                content = result.get("response", "")
                print(f"✅ Server response received ({len(content)} chars)")
                return content
            else:
                error_msg = result.get("error", "Unknown error")
                raise Exception(f"Server returned error: {error_msg}")
        else:
            raise Exception(f"Server returned status {response.status_code}: {response.text}")
    
    except Exception as e:
        print(f"❌ Server query failed: {e}")
        raise


def run_single_query_direct_ollama(query: str, agent_name: str, interceptor_data: dict = None, streaming: bool = True, no_tools: bool = False) -> str:
    """Original direct Ollama connection as fallback."""
    try:
        print(f"🔄 Using direct Ollama connection for fallback")
        
        # Use the helpers version which supports streaming
        from src.core.helpers import get_agent_instance as helpers_get_agent_instance
        agent = helpers_get_agent_instance(agent_name, streaming=streaming, with_tools=(not no_tools))
        
        # Prepare agent context with interceptor data
        agent_context = {}
        if interceptor_data:
            agent_context['interceptor_analysis'] = interceptor_data
            print(f"🧠 Providing agent with interceptor data:")
            print(f"   📊 Intent: {interceptor_data['detected_intent']} (confidence: {interceptor_data['confidence']:.2%})")
            print(f"   🔧 Commands executed: {interceptor_data['execution_stats']['total_commands']}")
            print(f"   ✅ Successful: {interceptor_data['execution_stats']['successful_commands']}")
            print(f"   📈 Data gathered: {interceptor_data['execution_stats']['total_data_gathered']} chars")
        
        # Check if this is a sophisticated agent with streaming
        if hasattr(agent, 'stream') and callable(getattr(agent, 'stream')):
            # For single query mode, we'll collect the response but avoid double printing
            result_parts = []
            printed_parts = set()
            
            def collect_token_safely(token):
                """Collect token and print only if not already printed."""
                if token not in printed_parts:
                    print(token, end='', flush=True)  # Show streaming tokens in real-time
                    printed_parts.add(token)
                result_parts.append(token)
            
            print(f"🎬 Streaming response:")
            # Pass interceptor context to streaming agent if supported
            if interceptor_data and hasattr(agent, 'stream_with_context'):
                agent.stream_with_context(query, collect_token_safely, agent_context)
            else:
                agent.stream(query, collect_token_safely)
            result = ''.join(result_parts)
            print(f"\n")  # Add newline after streaming
        else:
            # This is a simple agent
            if interceptor_data and hasattr(agent, 'run_query_with_context'):
                result = agent.run_query_with_context(query, agent_context)
            else:
                result = agent.run_query(query)
        
        print(f"🔍 Debug: Direct query completed, returning result")
        return result
    except Exception as e:
        print(f"🔍 Debug: Error in direct query: {e}")
        return f"❌ Error processing query '{query}': {e}"


def execute_resolved_command(resolved_command, original_query: str) -> Optional[str]:
    """Execute a resolved command from the commands folder."""
    try:
        # Import the command module dynamically
        command_module_path = f"src.commands.{resolved_command.command_name}"
        
        print(f"🔧 Attempting to load command module: {command_module_path}")
        
        # Check if the command directory exists
        command_dir = Path(f"src/commands/{resolved_command.command_name}")
        if not command_dir.exists():
            print(f"📁 Command directory not found: {command_dir}")
            return None
        
        # Try to import the command module
        try:
            import importlib
            command_module = importlib.import_module(command_module_path)
            
            # Get the command class
            command_class = getattr(command_module, resolved_command.command_class)
            
            # Instantiate and execute the command
            command_instance = command_class()
            result = command_instance.execute(original_query, resolved_command.parameters)
            
            return result
            
        except (ImportError, AttributeError) as e:
            print(f"⚠️  Could not load command {resolved_command.command_name}: {e}")
            return None
            
    except Exception as e:
        print(f"❌ Error executing command: {e}")
        return None

import os
import sys
from typing import Optional, Dict, Any
from pathlib import Path


def get_agent_instance(agent_name: str, streaming: bool = True):
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
