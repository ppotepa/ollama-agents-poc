#!/usr/bin/env python3
"""Test the query logging system manually."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.query_logger import get_query_logger

def test_query_logging():
    """Test the query logging system manually."""
    print("🧪 Testing Query Logging System")
    print("=" * 40)
    
    # Get logger
    query_logger = get_query_logger()
    
    # Start a test session
    query_id = query_logger.start_query_session("test query - what is 2+2?", "manual_test")
    print(f"📊 Started query session: {query_id}")
    
    # Simulate execution steps
    step1_id = query_logger.start_execution_step(
        "analysis", 
        "qwen2.5:7b-instruct-q4_K_M", 
        "Analyzing simple math question"
    )
    print(f"🔄 Started step {step1_id}")
    
    # Log some context usage
    query_logger.log_context_usage(
        source="user_input",
        content="test query - what is 2+2?",
        metadata={"type": "math_question", "complexity": "simple"}
    )
    
    query_logger.log_context_usage(
        source="system_knowledge",
        content="Basic arithmetic: 2 + 2 = 4",
        metadata={"domain": "mathematics"}
    )
    
    # Log prompt decoration
    query_logger.log_prompt_decoration(
        original="what is 2+2?",
        decorated="You are a helpful assistant. Please answer: what is 2+2?",
        decorations=["system_message", "instruction_prefix"],
        system_message="You are a helpful assistant."
    )
    
    # Simulate tool execution
    query_logger.log_tool_execution(
        tool_name="calculator",
        args={"expression": "2+2"},
        execution_time=0.001,
        success=True,
        output="4"
    )
    
    # End the step
    query_logger.end_execution_step("The answer is 4", execution_time=0.5)
    
    # Simulate another step with model switch
    step2_id = query_logger.start_execution_step(
        "verification",
        "gemma:7b-instruct-q4_K_M",
        "Verifying the calculation"
    )
    
    # Log agent switch
    query_logger.log_agent_switch(
        from_agent="qwen2.5:7b-instruct-q4_K_M",
        to_agent="gemma:7b-instruct-q4_K_M", 
        reason="Better for verification tasks",
        context_preserved="Previous calculation: 2+2=4",
        success=True
    )
    
    # End verification step
    query_logger.end_execution_step("Verified: 2+2=4 is correct", execution_time=0.3)
    
    # End the session
    final_answer = "The answer to 2+2 is 4. This is basic arithmetic."
    query_logger.end_query_session(final_answer, success=True)
    
    print(f"✅ Ended query session: {query_id}")
    
    # Test tree visualization
    print("\n🌳 Execution Tree:")
    print("-" * 40)
    tree = query_logger.create_execution_tree_visualization()
    print(tree)
    
    # Test session summary
    print("\n📊 Session Summary:")
    print("-" * 40)
    summary = query_logger.get_session_summary()
    print(f"Total queries: {summary['session_summary']['total_queries']}")
    print(f"Success rate: {summary['session_summary']['success_rate']:.1%}")
    print(f"Average time: {summary['session_summary']['average_query_time']:.2f}s")
    
    return query_id

if __name__ == "__main__":
    test_query_logging()
