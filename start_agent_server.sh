#!/bin/bash
# Generic Ollama Agent Server Launcher for Linux/WSL
echo "🚀 Starting Generic Ollama Agent Server..."

# Function to activate virtual environment
activate_venv() {
    if [ -f ".venv/bin/activate" ]; then
        echo "🔧 Activating virtual environment (.venv)..."
        source .venv/bin/activate
    elif [ -f "venv/bin/activate" ]; then
        echo "🔧 Activating virtual environment (venv)..."
        source venv/bin/activate
    elif [ -f "ollama-agents/bin/activate" ]; then
        echo "🔧 Activating virtual environment (ollama-agents)..."
        source ollama-agents/bin/activate
    else
        echo "⚠️ No virtual environment found, using system Python"
    fi
}

activate_venv

echo "🖥️ Launching Agent Server..."
python src/core/server.py
