#!/usr/bin/env python3
"""
Test script for the integrations system.
Verifies that the integration manager and Ollama integration work correctly.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.integrations import IntegrationManager, OllamaIntegration


def test_ollama_integration():
    """Test the Ollama integration directly."""
    print("🔧 Testing Ollama Integration...")
    
    # Test both local and container configurations
    configurations = [
        ("localhost", "http://localhost:11434"),
        ("container", "http://ollama:11434")
    ]
    
    for config_name, host in configurations:
        print(f"   Testing {config_name} configuration ({host})...")
        ollama = OllamaIntegration(host=host)
        
        # Test availability
        available = ollama.is_available()
        print(f"   ✅ Ollama Available ({config_name}): {available}")
        
        if available:
            # Test version
            version = ollama.get_version()
            print(f"   📋 Ollama Version ({config_name}): {version}")
            
            # Test model discovery
            models = ollama.get_models()
            print(f"   🎯 Models Found ({config_name}): {len(models)}")
            
            if models:
                first_model = models[0]
                print(f"   🎯 First Model ({config_name}): {first_model['id']} ({first_model['details']['size']})")
            
            return True  # Return success if any configuration works
    
    return False  # Return failure if no configuration works


def test_integration_manager():
    """Test the integration manager."""
    print("\n🔧 Testing Integration Manager...")
    
    manager = IntegrationManager()
    
    # Replace the default Ollama integration with a localhost one for testing
    manager.remove_integration("ollama")
    manager.add_integration("ollama", OllamaIntegration(host="http://localhost:11434"))
    
    # List integrations
    integrations = manager.list_integrations()
    print(f"   📋 Registered Integrations: {integrations}")
    
    # Check availability
    available = manager.get_available_integrations()
    print(f"   ✅ Available Integrations: {available}")
    
    # Get all models
    all_models = manager.get_all_models()
    print(f"   🎯 Total Models: {len(all_models)}")
    
    # Health check
    health = manager.health_check()
    print(f"   💚 Health Status:")
    for name, status in health.items():
        print(f"      {name}: {status['status']} (v{status['version']})")
    
    return len(available) > 0


def main():
    """Run all tests."""
    print("🚀 Testing Integrations System")
    print("=" * 50)
    
    # Test individual integration
    ollama_ok = test_ollama_integration()
    
    # Test integration manager
    manager_ok = test_integration_manager()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Ollama Integration: {'✅ PASS' if ollama_ok else '❌ FAIL'}")
    print(f"   Integration Manager: {'✅ PASS' if manager_ok else '❌ FAIL'}")
    
    if ollama_ok and manager_ok:
        print("🎉 All tests passed! Integration system is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check your setup.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
