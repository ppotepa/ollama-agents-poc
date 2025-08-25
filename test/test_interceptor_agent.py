#!/usr/bin/env python3
"""
Test script for the InterceptorAgent to verify phi3:mini model usage.

This script allows you to test the interceptor agent independently and see
how it uses the phi3:mini model for prompt analysis and command recommendations.
"""

import sys
import os
import json
from typing import List, Dict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.agents.interceptor.agent import InterceptorAgent, CommandRecommendation
    from src.core.prompt_interceptor import InterceptionMode
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the ollama directory")
    sys.exit(1)


class InterceptorAgentTester:
    """Test class for experimenting with the InterceptorAgent."""
    
    def __init__(self):
        """Initialize the tester with different agent configurations."""
        self.configs = {
            "basic": {
                "name": "Basic Interceptor Agent",
                "backend_image": "phi3:mini",
                "parameters": {
                    "temperature": 0.1,
                    "num_ctx": 2048,
                    "num_predict": 512
                }
            },
            "creative": {
                "name": "Creative Interceptor Agent",
                "backend_image": "phi3:mini",
                "parameters": {
                    "temperature": 0.7,
                    "num_ctx": 4096,
                    "num_predict": 1024
                }
            },
            "precise": {
                "name": "Precise Interceptor Agent",
                "backend_image": "phi3:mini",
                "parameters": {
                    "temperature": 0.0,
                    "num_ctx": 1024,
                    "num_predict": 256
                }
            }
        }
        self.current_agent = None
        self.current_config_name = None
    
    def create_agent(self, config_name: str = "basic") -> InterceptorAgent:
        """Create an interceptor agent with the specified configuration."""
        if config_name not in self.configs:
            print(f"❌ Unknown config: {config_name}")
            print(f"Available configs: {list(self.configs.keys())}")
            return None
        
        config = self.configs[config_name]
        print(f"🤖 Creating InterceptorAgent with config: {config_name}")
        print(f"   Model: {config['backend_image']}")
        print(f"   Temperature: {config['parameters']['temperature']}")
        print(f"   Context: {config['parameters']['num_ctx']}")
        
        try:
            agent = InterceptorAgent("interceptor", config)
            self.current_agent = agent
            self.current_config_name = config_name
            print(f"✅ Agent created successfully!")
            return agent
        except Exception as e:
            print(f"❌ Failed to create agent: {e}")
            return None
    
    def test_pattern_matching(self, prompts: List[str]) -> None:
        """Test pattern matching capabilities."""
        if not self.current_agent:
            print("❌ No agent available. Create one first with create_agent()")
            return
        
        print(f"\n🧪 Testing Pattern Matching ({self.current_config_name} config)")
        print("=" * 60)
        
        for prompt in prompts:
            print(f"\n📝 Prompt: '{prompt}'")
            try:
                recommendations = self.current_agent.analyze_prompt(prompt, InterceptionMode.LIGHTWEIGHT)
                
                if recommendations:
                    print(f"   ✅ Found {len(recommendations)} recommendations:")
                    for i, rec in enumerate(recommendations[:3], 1):
                        print(f"     {i}. {rec.command} (confidence: {rec.confidence:.3f})")
                        print(f"        📝 {rec.description}")
                        print(f"        🏷️  Category: {rec.category}")
                        if rec.required_context:
                            print(f"        📋 Context: {', '.join(rec.required_context)}")
                else:
                    print("   ⚠️  No recommendations found")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    def test_llm_analysis(self, prompts: List[str]) -> None:
        """Test LLM-based analysis capabilities."""
        if not self.current_agent:
            print("❌ No agent available. Create one first with create_agent()")
            return
        
        print(f"\n🧠 Testing LLM Analysis ({self.current_config_name} config)")
        print("=" * 60)
        
        for prompt in prompts:
            print(f"\n📝 Prompt: '{prompt}'")
            try:
                # Test if LLM analysis is available
                if hasattr(self.current_agent, 'analyze_prompt_with_llm'):
                    print("   🔍 Using LLM analysis...")
                    result = self.current_agent.analyze_prompt_with_llm(prompt)
                    print(f"   📊 LLM Result: {result}")
                else:
                    print("   ⚠️  LLM analysis not available, using pattern matching")
                    recommendations = self.current_agent.analyze_prompt(prompt, InterceptionMode.LIGHTWEIGHT)
                    print(f"   📊 Pattern Result: {len(recommendations)} recommendations")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    def test_lightweight_response(self, prompts: List[str]) -> None:
        """Test lightweight response generation."""
        if not self.current_agent:
            print("❌ No agent available. Create one first with create_agent()")
            return
        
        print(f"\n⚡ Testing Lightweight Response Generation ({self.current_config_name} config)")
        print("=" * 80)
        
        for prompt in prompts:
            print(f"\n📝 Prompt: '{prompt}'")
            print("-" * 50)
            try:
                response = self.current_agent.generate_lightweight_response(prompt)
                print(response)
            except Exception as e:
                print(f"❌ Error generating response: {e}")
                import traceback
                traceback.print_exc()
    
    def interactive_mode(self):
        """Interactive mode for testing prompts."""
        if not self.current_agent:
            self.create_agent()
        
        print(f"\n🎮 Interactive Mode ({self.current_config_name} config)")
        print("Commands:")
        print("  /config <name>  - Switch agent configuration (basic/creative/precise)")
        print("  /pattern        - Test pattern matching only")
        print("  /llm           - Test LLM analysis only")
        print("  /response      - Test full response generation")
        print("  /quit          - Exit interactive mode")
        print("  <prompt>       - Analyze prompt with current agent")
        print("=" * 60)
        
        while True:
            try:
                user_input = input(f"\n[{self.current_config_name}] Enter prompt: ").strip()
                
                if user_input.lower() in ['/quit', 'quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.startswith('/config '):
                    config_name = user_input[8:].strip()
                    self.create_agent(config_name)
                    continue
                
                if user_input == '/pattern':
                    test_prompt = input("Enter test prompt: ").strip()
                    if test_prompt:
                        self.test_pattern_matching([test_prompt])
                    continue
                
                if user_input == '/llm':
                    test_prompt = input("Enter test prompt: ").strip()
                    if test_prompt:
                        self.test_llm_analysis([test_prompt])
                    continue
                
                if user_input == '/response':
                    test_prompt = input("Enter test prompt: ").strip()
                    if test_prompt:
                        self.test_lightweight_response([test_prompt])
                    continue
                
                if not user_input:
                    continue
                
                # Default: full analysis
                print("\n🔍 Full Analysis:")
                self.test_pattern_matching([user_input])
                self.test_lightweight_response([user_input])
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function to run the interceptor agent tests."""
    print("🚀 InterceptorAgent Test Suite")
    print("=" * 50)
    
    # Sample test prompts
    test_prompts = [
        "analyze repository structure",
        "what files handle configuration in this project",
        "find all Python files",
        "how is this project organized",
        "what technologies are used",
        "debug this error message",
        "create a new feature",
        "read the README file",
        "list all directories",
        "search for authentication code"
    ]
    
    tester = InterceptorAgentTester()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--interactive":
            tester.interactive_mode()
            return
        elif sys.argv[1] == "--config" and len(sys.argv) > 2:
            config_name = sys.argv[2]
            tester.create_agent(config_name)
        else:
            print("Usage:")
            print("  python test_interceptor_agent.py --interactive")
            print("  python test_interceptor_agent.py --config <basic|creative|precise>")
            print("  python test_interceptor_agent.py  (run all tests)")
            return
    else:
        tester.create_agent("basic")
    
    if not tester.current_agent:
        print("❌ Failed to create agent, exiting")
        return
    
    # Run tests
    print(f"\n1️⃣  Testing Pattern Matching")
    tester.test_pattern_matching(test_prompts[:5])
    
    print(f"\n2️⃣  Testing LLM Analysis")
    tester.test_llm_analysis(test_prompts[:3])
    
    print(f"\n3️⃣  Testing Lightweight Response")
    tester.test_lightweight_response(test_prompts[:3])
    
    print(f"\n✅ All tests completed!")
    print(f"\nTo run interactive mode: python test_interceptor_agent.py --interactive")


if __name__ == "__main__":
    main()
