#!/usr/bin/env python3
"""
Simple test to verify phi3:mini model usage in InterceptorAgent.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.agents.interceptor.agent import InterceptorAgent
    from src.core.prompt_interceptor import InterceptionMode
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_phi_model_usage():
    """Test if the InterceptorAgent is actually using phi3:mini."""
    
    print("🧪 Testing phi3:mini Model Usage")
    print("=" * 40)
    
    # Create agent with phi3:mini
    config = {
        "name": "Test Interceptor Agent",
        "backend_image": "phi3:mini",
        "parameters": {
            "temperature": 0.1,
            "num_ctx": 1024,
            "num_predict": 256
        }
    }
    
    print(f"🤖 Creating agent with model: {config['backend_image']}")
    agent = InterceptorAgent("interceptor", config)
    
    # Check if LLM is built
    print(f"🔍 Checking LLM availability...")
    if hasattr(agent, 'llm') and agent.llm is not None:
        print(f"✅ LLM is available: {type(agent.llm)}")
        print(f"   Model: {getattr(agent.llm, 'model', 'Unknown')}")
    else:
        print(f"⚠️  LLM is not available or None")
    
    # Test pattern-based analysis
    test_prompt = "analyze repository structure"
    print(f"\n📝 Testing prompt: '{test_prompt}'")
    
    print(f"🔍 Pattern-based analysis:")
    recommendations = agent.analyze_prompt(test_prompt, InterceptionMode.LIGHTWEIGHT)
    print(f"   Found {len(recommendations)} recommendations")
    
    # Test if LLM method exists and works
    if hasattr(agent, 'analyze_prompt_with_llm'):
        print(f"🔍 LLM-based analysis:")
        try:
            llm_result = agent.analyze_prompt_with_llm(test_prompt)
            print(f"   ✅ LLM responded: {llm_result[:100]}...")
        except Exception as e:
            print(f"   ❌ LLM analysis failed: {e}")
    else:
        print(f"⚠️  LLM analysis method not available")
    
    # Test response generation
    print(f"🔍 Testing response generation:")
    try:
        response = agent.generate_lightweight_response(test_prompt)
        print(f"   ✅ Response generated ({len(response)} chars)")
        print(f"   Preview: {response[:150]}...")
    except Exception as e:
        print(f"   ❌ Response generation failed: {e}")


def test_direct_llm_call():
    """Test direct LLM call to verify phi3:mini is working."""
    print(f"\n🔗 Testing Direct LLM Call")
    print("=" * 30)
    
    try:
        from langchain_ollama import ChatOllama
        
        print(f"🤖 Creating direct ChatOllama connection to phi3:mini...")
        llm = ChatOllama(
            model="phi3:mini",
            temperature=0.1,
            num_ctx=1024
        )
        
        test_message = "Analyze this prompt and suggest commands: 'find configuration files'"
        print(f"📝 Sending message: {test_message}")
        
        response = llm.invoke(test_message)
        print(f"✅ Response received:")
        print(f"   Content: {response.content}")
        print(f"   Type: {type(response)}")
        
    except Exception as e:
        print(f"❌ Direct LLM test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_phi_model_usage()
    test_direct_llm_call()
