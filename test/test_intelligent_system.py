#!/usr/bin/env python3
"""Test script for the intelligent investigation system."""

import sys
import os

# Add the src directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def test_task_decomposition():
    """Test the task decomposer."""
    print("🧪 Testing Task Decomposition...")
    
    from src.core.task_decomposer import TaskDecomposer
    
    decomposer = TaskDecomposer()
    query = "Analyze the codebase structure and find any potential bugs"
    
    result = decomposer.decompose(query)
    
    print(f"📝 Original query: {query}")
    print(f"🔍 Execution strategy: {result.execution_strategy}")
    print(f"📊 Number of subtasks: {len(result.subtasks)}")
    
    for i, subtask in enumerate(result.subtasks[:3], 1):  # Show first 3
        print(f"  {i}. {subtask.description}")
        print(f"     Type: {subtask.task_type}")
        print(f"     Model: {subtask.preferred_models[0] if subtask.preferred_models else 'default'}")
        print(f"     Complexity: {subtask.estimated_complexity:.2f}")
    
    return result

def test_execution_planning():
    """Test the execution planner."""
    print("\n🧪 Testing Execution Planning...")
    
    from src.core.execution_planner import ExecutionPlanner
    from src.core.task_decomposer import TaskDecomposer
    
    # First decompose a task
    decomposer = TaskDecomposer()
    decomposition = decomposer.decompose("Implement a new feature to sort files by date")
    
    # Then create execution plan
    planner = ExecutionPlanner()
    plan = planner.create_execution_plan(decomposition)  # Pass the full decomposition object
    
    print(f"📋 Execution plan created with {len(plan.execution_steps)} steps")
    print(f"⏱️  Total estimated duration: {plan.total_estimated_time:.0f} seconds")
    
    for i, step in enumerate(plan.execution_steps[:3], 1):  # Show first 3
        print(f"  {i}. {step.subtask.description}")
        print(f"     Model: {step.assigned_model}")
        print(f"     Confidence: {step.model_confidence:.2f}")
    
    return plan

def test_investigation_strategies():
    """Test the investigation strategy manager."""
    print("\n🧪 Testing Investigation Strategies...")
    
    from src.core.investigation_strategies import InvestigationStrategyManager
    
    manager = InvestigationStrategyManager()
    
    queries = [
        "Find all Python files in the project",
        "Give me a complete overview of the codebase", 
        "Debug this specific error in main.py",
        "Implement a comprehensive logging system"
    ]
    
    for query in queries:
        strategy = manager.select_optimal_strategy(query, {})
        plan = manager.create_investigation_plan(query, strategy)
        
        print(f"🔍 Query: {query[:50]}...")
        print(f"📋 Strategy: {strategy.value}")
        print(f"📊 Steps: {len(plan.steps)}")
        print()

def test_reflection_system():
    """Test the reflection system."""
    print("🧪 Testing Reflection System...")
    
    from src.core.reflection_system import ReflectionSystem, ReflectionTrigger
    
    system = ReflectionSystem()
    
    # Simulate a step execution
    session_id = "test_session_123"
    step_id = "test_step_1"
    model = "qwen2.5:7b-instruct-q4_K_M"
    result = "Successfully analyzed the file structure and found 15 Python files"
    execution_time = 45.5
    
    reflection = system.evaluate_step(
        session_id, step_id, model, result, execution_time
    )
    
    print(f"🤔 Reflection for step: {step_id}")
    print(f"🎯 Confidence: {reflection.confidence_level.value}")
    print(f"🔄 Should swap model: {reflection.should_swap}")
    print(f"💡 Reasoning: {reflection.reasoning}")
    if reflection.improvement_suggestions:
        print(f"📈 Suggestions: {reflection.improvement_suggestions[0]}")

def test_context_manager():
    """Test the context manager."""
    print("\n🧪 Testing Context Manager...")
    
    from src.core.context_manager import ContextManager
    
    manager = ContextManager()
    session_id = "test_context_session"
    
    # Create a context
    context = manager.create_context(
        session_id=session_id,
        original_query="Analyze codebase for potential optimizations"
    )
    
    # Add some context data
    manager.add_context_data(
        session_id, "discovered_files", ["main.py", "utils.py", "config.py"],
        "file_scanner"
    )
    
    manager.add_context_data(
        session_id, "optimization_opportunities", 
        ["Loop optimization in main.py", "Memory usage in utils.py"],
        "code_analyzer"
    )
    
    # Record an execution
    manager.record_execution(
        session_id, "scan_files", "qwen2.5:7b-instruct-q4_K_M",
        "Found 3 Python files", 12.5
    )
    
    # Get statistics
    stats = manager.get_session_statistics(session_id)
    
    print(f"📊 Context statistics:")
    print(f"   Session: {stats['session_id']}")
    print(f"   Steps: {stats['total_steps']}")
    print(f"   Duration: {stats['total_duration']:.1f}s")
    print(f"   Context entries: {stats['context_entries']}")

def main():
    """Run all tests."""
    print("🚀 Testing Intelligent Investigation System")
    print("=" * 60)
    
    try:
        # Test individual components
        test_task_decomposition()
        test_execution_planning()
        test_investigation_strategies()
        test_reflection_system()
        test_context_manager()
        
        print("\n✅ All tests completed successfully!")
        print("\n🎯 System is ready for intelligent investigation!")
        print("\nTo use the system:")
        print("  python main.py --agent universal --intelligent -q 'your complex query here'")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
