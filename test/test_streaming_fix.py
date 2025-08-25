#!/usr/bin/env python3
"""Test script to check if character duplication in streaming is fixed."""

from src.core.single_query_mode import run_single_query

def test_streaming():
    """Test streaming functionality to verify duplication fix."""
    print("🧪 Testing streaming character duplication fix...")
    print("=" * 50)
    
    # Simple test query
    test_query = "Count from 1 to 5"
    agent_type = "qwen2.5:7b-instruct-q4_K_M"  # Use a valid model from our discovery
    
    print(f"Query: {test_query}")
    print(f"Agent: {agent_type}")
    print("-" * 50)
    
    try:
        result = run_single_query(
            query=test_query,
            agent_name=agent_type,
            force_streaming=True,
            interception_mode="off"  # Disable interceptor for cleaner test
        )
        print(f"\n" + "=" * 50)
        print("✅ Streaming test completed!")
        print(f"Final result length: {len(result) if result else 0} characters")
        
    except Exception as e:
        print(f"❌ Error during streaming test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_streaming()
