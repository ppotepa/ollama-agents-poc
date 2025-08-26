#!/usr/bin/env python3
"""Debug streaming chunks to understand the data structure."""

import sys
sys.path.append('../')

from src.agents.universal.agent import UniversalAgent

def debug_streaming_chunks():
    """Debug what types of chunks we get during streaming."""
    print("🔍 Debugging Streaming Chunks")
    print("=" * 40)
    
    # Create agent instance
    config = {
        "model_id": "qwen2.5:7b-instruct-q4_K_M",
        "type": "instruct"
    }
    
    agent = UniversalAgent("debug-agent", config, streaming=True)
    agent.load()  # Load the agent
    
    prompt = "Say hello"
    
    print(f"🎯 Prompt: {prompt}")
    print("🔍 Analyzing chunks:")
    print("-" * 40)
    
    try:
        chunk_count = 0
        for chunk in agent._llm.stream(prompt):
            chunk_count += 1
            print(f"\n--- Chunk {chunk_count} ---")
            print(f"Type: {type(chunk)}")
            print(f"Has content: {hasattr(chunk, 'content')}")
            print(f"Has text: {hasattr(chunk, 'text')}")
            
            if hasattr(chunk, 'content'):
                print(f"Content: '{chunk.content}' (type: {type(chunk.content)})")
            if hasattr(chunk, 'text'):
                print(f"Text: '{chunk.text}' (type: {type(chunk.text)})")
            
            # Show all attributes
            attrs = [attr for attr in dir(chunk) if not attr.startswith('_')]
            print(f"Attributes: {attrs[:10]}")  # Show first 10 to avoid clutter
            
            if chunk_count > 5:  # Limit to first 5 chunks
                print("... (limiting output)")
                break
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_streaming_chunks()
