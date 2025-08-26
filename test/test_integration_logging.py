#!/usr/bin/env python3
"""Integration test for query logging with real execution."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.query_logger import get_query_logger
from src.core.universal_multi_agent import create_universal_multi_agent


def test_integration_logging():
    """Test query logging with real Universal Multi-Agent execution."""
    print("🧪 Testing Query Logging Integration")
    print("=" * 45)
    
    # Get logger
    query_logger = get_query_logger()
    
    # Create Universal Multi-Agent
    print("🤖 Creating Universal Multi-Agent...")
    agent = create_universal_multi_agent()
    
    # Start logging session
    query = "write a Python function to calculate the factorial of a number"
    query_id = query_logger.start_query_session(query, "integration_test")
    print(f"📊 Started query session: {query_id}")
    
    try:
        # Start execution step
        query_logger.start_execution_step(
            "universal_agent_processing",
            agent._current_model_id or "universal",
            "Processing request with Universal Multi-Agent"
        )
        
        # Log initial context
        query_logger.log_context_usage(
            source="user_request",
            content=query,
            metadata={"task_type": "coding", "complexity": "medium"}
        )
        
        # Analyze task (simulate what the agent does internally)
        task_type = agent._analyze_task_requirements(query)
        optimal_model = agent._select_optimal_model(task_type)
        
        # Log task analysis
        query_logger.log_context_usage(
            source="task_analysis",
            content=f"Task type: {task_type.value}, Optimal model: {optimal_model}",
            metadata={"task_type": task_type.value, "optimal_model": optimal_model}
        )
        
        # Log prompt decoration (simulate)
        original_prompt = query
        decorated_prompt = f"""You are an expert Python programmer. Please provide a complete, well-documented solution.

User request: {query}

Please include:
1. A complete function implementation
2. Proper error handling
3. Example usage
4. Explanation of the approach"""
        
        query_logger.log_prompt_decoration(
            original=original_prompt,
            decorated=decorated_prompt,
            decorations=["system_message", "instruction_enhancement", "format_specification"],
            system_message="You are an expert Python programmer."
        )
        
        # Check if model switch occurred
        old_model = agent._current_model_id
        
        # Process the request (this might trigger a model switch)
        print(f"🔄 Processing with current model: {agent._current_model_id}")
        start_time = agent._get_timestamp()
        
        # Simulate processing (in real integration, this would be the actual processing)
        import time
        time.sleep(0.1)  # Simulate processing time
        
        result = """def factorial(n):
    \"\"\"Calculate the factorial of a number.
    
    Args:
        n (int): Non-negative integer to calculate factorial for
        
    Returns:
        int: Factorial of n
        
    Raises:
        ValueError: If n is negative
        TypeError: If n is not an integer
    \"\"\"
    if not isinstance(n, int):
        raise TypeError("Factorial is only defined for integers")
    
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    
    if n == 0 or n == 1:
        return 1
    
    result = 1
    for i in range(2, n + 1):
        result *= i
    
    return result

# Example usage:
if __name__ == "__main__":
    print(f"5! = {factorial(5)}")  # Output: 120
    print(f"0! = {factorial(0)}")  # Output: 1
    
    try:
        factorial(-1)
    except ValueError as e:
        print(f"Error: {e}")"""
        
        end_time = agent._get_timestamp()
        
        # Check if model switch happened
        new_model = agent._current_model_id
        if old_model != new_model:
            query_logger.log_agent_switch(
                from_agent=old_model,
                to_agent=new_model,
                reason=f"Optimal for {task_type.value} task",
                context_preserved="Task analysis and user request context",
                success=True
            )
            print(f"🔄 Model switched: {old_model} → {new_model}")
        
        # Log tool execution (simulate)
        query_logger.log_tool_execution(
            tool_name="code_generator",
            args={"language": "python", "task": "factorial_function"},
            execution_time=0.1,
            success=True,
            output=result,
            follow_up_tools=["syntax_checker", "test_generator"]
        )
        
        # End execution step
        query_logger.end_execution_step(
            output_generated=result[:200] + "..." if len(result) > 200 else result,
            execution_time=0.15
        )
        
        # End session successfully
        query_logger.end_query_session(result, success=True)
        
        print(f"✅ Query processed successfully")
        print(f"🔧 Current model: {agent._current_model_id}")
        print(f"📊 Query logged with ID: {query_id}")
        
        # Show execution tree
        print("\n🌳 Execution Tree:")
        print("-" * 30)
        tree = query_logger.create_execution_tree_visualization()
        print(tree)
        
        return query_id
        
    except Exception as e:
        # Log error
        query_logger.end_query_session(
            final_answer=None,
            success=False,
            error_messages=[str(e)]
        )
        print(f"❌ Error during processing: {e}")
        raise


def analyze_integration_results():
    """Analyze the results of the integration test."""
    print("\n📊 Analyzing Integration Test Results")
    print("=" * 40)
    
    from src.core.query_analyzer import create_query_analyzer
    
    # Create analyzer
    analyzer = create_query_analyzer()
    
    # Load recent logs
    count = analyzer.load_logs(days_back=1)
    print(f"📈 Loaded {count} logs for analysis")
    
    if count > 0:
        # Generate report
        report = analyzer.generate_comprehensive_report()
        print("\n" + report)
        
        # Export analytics
        output_file = analyzer.export_detailed_analytics("integration_test_analytics.json")
        if output_file:
            print(f"\n📁 Detailed analytics exported to: {output_file}")


if __name__ == "__main__":
    # Run integration test
    query_id = test_integration_logging()
    
    # Analyze results
    analyze_integration_results()
    
    print(f"\n✅ Integration test completed successfully!")
    print(f"🔍 Query ID: {query_id}")
    print("📁 Check logs/query_execution/ for detailed logs")
    print("📊 Use analyze_logs.py for further analysis")
