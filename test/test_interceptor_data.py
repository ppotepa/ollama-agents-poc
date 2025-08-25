#!/usr/bin/env python3
"""Test script to demonstrate interceptor data enhancement."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.prompt_interceptor import intercept_and_enhance_prompt, InterceptionMode

def main():
    """Test the interceptor data collection and enhancement."""
    print("🧪 Testing Interceptor Data Enhancement")
    print("=" * 50)
    
    # Test query
    query = "what files are in the repository"
    working_dir = os.getcwd()
    
    # Test with lightweight mode
    print(f"📝 Original Query: '{query}'")
    print(f"📁 Working Directory: {working_dir}")
    print()
    
    try:
        supplemented = intercept_and_enhance_prompt(query, working_dir, None, InterceptionMode.LIGHTWEIGHT)
        
        print("🔍 INTERCEPTOR ANALYSIS RESULTS:")
        print(f"   🎯 Detected Intent: {supplemented.metadata.get('detected_intent', 'Unknown')}")
        print(f"   📊 Confidence: {supplemented.metadata.get('confidence', 0):.2%}")
        print(f"   🔧 Context Types: {', '.join(supplemented.context_used)}")
        print()
        
        print("🛠️ COMMAND EXECUTION DETAILS:")
        for i, cmd in enumerate(supplemented.commands_executed, 1):
            status = "✅" if cmd.success else "❌"
            print(f"   [{i}] {cmd.command_name}: {status} ({cmd.duration:.3f}s, {cmd.result_length} chars)")
            if cmd.error_message:
                print(f"       Error: {cmd.error_message}")
        print()
        
        print("📈 EXECUTION STATISTICS:")
        total_commands = len(supplemented.commands_executed)
        successful_commands = len([cmd for cmd in supplemented.commands_executed if cmd.success])
        total_execution_time = sum(cmd.duration for cmd in supplemented.commands_executed)
        total_data_gathered = sum(cmd.result_length for cmd in supplemented.commands_executed if cmd.success)
        
        print(f"   📊 Total Commands: {total_commands}")
        print(f"   ✅ Successful Commands: {successful_commands}")
        print(f"   ⏱️  Total Execution Time: {total_execution_time:.3f}s")
        print(f"   📈 Total Data Gathered: {total_data_gathered} characters")
        print()
        
        print("💬 AGENT-READY DATA STRUCTURE:")
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
                'total_commands': total_commands,
                'successful_commands': successful_commands,
                'total_execution_time': total_execution_time,
                'total_data_gathered': total_data_gathered
            }
        }
        
        print("   📦 Data ready for agent:")
        for key, value in interceptor_data.items():
            if key == 'commands_executed':
                print(f"      {key}: {len(value)} command results")
            elif key == 'execution_stats':
                print(f"      {key}: {value}")
            else:
                print(f"      {key}: {value}")
        print()
        
        print("🚀 ENHANCED PROMPT PREVIEW:")
        print("-" * 50)
        print(supplemented.supplemented_prompt[:500] + "..." if len(supplemented.supplemented_prompt) > 500 else supplemented.supplemented_prompt)
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
