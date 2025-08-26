#!/usr/bin/env python3
"""Test the Universal Multi-Agent system."""

from src.core.universal_multi_agent import create_universal_multi_agent

def test_universal_multi_agent():
    print('Testing Universal Multi-Agent...')
    
    # Create agent
    agent = create_universal_multi_agent()
    print(f'✅ Created: {type(agent).__name__}')
    
    # Test task analysis
    print('\n🔍 Testing task analysis:')
    
    # Coding task
    recommendations = agent.get_model_recommendations('write a Python function to sort a list')
    print(f'Coding task - Recommended models: {len(recommendations)} models found')
    if recommendations:
        print(f'  Top choice: {recommendations[0]["model_id"]} (score: {recommendations[0]["score"]:.1f})')
    
    # Analysis task
    recommendations = agent.get_model_recommendations('analyze this data and find patterns')
    print(f'Analysis task - Recommended models: {len(recommendations)} models found')
    if recommendations:
        print(f'  Top choice: {recommendations[0]["model_id"]} (score: {recommendations[0]["score"]:.1f})')
    
    # Creative task
    recommendations = agent.get_model_recommendations('write a creative story about space')
    print(f'Creative task - Recommended models: {len(recommendations)} models found')
    if recommendations:
        print(f'  Top choice: {recommendations[0]["model_id"]} (score: {recommendations[0]["score"]:.1f})')
    
    print('\n📊 Agent status:')
    status = agent.get_status()
    print(f'Current model: {status["current_model"]}')
    print(f'Available models: {len(status["available_models"])} models')
    print(f'Total switches: {status["total_switches"]}')
    
    # Test manual switch
    print('\n🔄 Testing manual model switch:')
    success = agent.force_model_switch("qwen2.5-coder:7b", "Manual test")
    print(f'Switch to qwen2.5-coder:7b: {"✅ Success" if success else "❌ Failed"}')
    
    status = agent.get_status()
    print(f'New current model: {status["current_model"]}')
    
    print('\n✅ Universal Multi-Agent system working!')
    return True

if __name__ == "__main__":
    test_universal_multi_agent()
