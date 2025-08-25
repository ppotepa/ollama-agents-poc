#!/usr/bin/env python3
"""Direct streaming test to isolate character duplication."""

import sys
sys.path.append('.')

from src.agents.universal.agent import UniversalAgent
from src.core.helpers import get_agent_instance

def test_direct_streaming():
    """Test direct agent streaming to isolate duplication."""
    print("🧪 Direct Streaming Test")
    print("=" * 40)
    
    # Create agent instance
    config = {
        "model_id": "qwen2.5:7b-instruct-q4_K_M",
        "type": "instruct"
    }
    
    agent = UniversalAgent("test-agent", config, streaming=True)
    
    # Test simple prompt
    prompt = "Say hello and count 1,2,3"
    
    print(f"🎯 Prompt: {prompt}")
    print("🎬 Streaming output:")
    print("-" * 40)
    
    # Collect tokens with custom callback
    collected_tokens = []
    token_count = 0
    
    def token_callback(token):
        nonlocal token_count
        token_count += 1
        collected_tokens.append(token)
        # Don't print here to avoid double printing
        # print(f"[{token_count}] '{token}'", end="", flush=True)
    
    try:
        result = agent.stream(prompt, token_callback)
        
        print("\n" + "-" * 40)
        print(f"✅ Streaming completed!")
        print(f"📊 Tokens received: {token_count}")
        print(f"📊 Result length: {len(result)}")
        print(f"📊 Collected tokens: {len(collected_tokens)}")
        
        # Check for duplicates
        token_set = set(collected_tokens)
        if len(token_set) < len(collected_tokens):
            print(f"⚠️  Duplicate tokens detected: {len(collected_tokens) - len(token_set)} duplicates")
        else:
            print("✅ No duplicate tokens detected")
            
        print(f"\n📝 Final result:\n{result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_streaming()
