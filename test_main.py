#!/usr/bin/env python3
"""Test main.py"""

import sys
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

print("🔧 Testing main.py functionality...")

try:
    from src.config.settings import config_manager
    print("✅ Config manager imported successfully")
    
    agents = config_manager.get_available_agents()
    print(f"✅ Found {len(agents)} agents: {list(agents.keys())}")
    
    print("=" * 60)
    print("🤖 Generic Ollama Agent - Modular AI Assistant Platform")
    print("🚀 Choose your AI model and start coding!")
    print("=" * 60)
    
    print("\n📋 Available AI Agents:")
    print("-" * 40)
    for idx, (key, cfg) in enumerate(agents.items(), 1):
        name = cfg.get("name", key)
        desc = cfg.get("description", "No description available")
        backend = cfg.get("backend_image") or cfg.get("model_id", "Unknown")
        print(f"\n{idx}. {name}")
        print(f"   ID: {backend}")
        print(f"   📝 {desc}")
        
    print(f"\n📊 Found {len(agents)} available agents")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
