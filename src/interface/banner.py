"""Banner and agent listing display utilities."""
from __future__ import annotations

from typing import List, Tuple

from src.utils.animations import stream_text, progressive_reveal

FAST_MENU = False  # will be set by main via configure

def configure(fast_menu: bool):
    global FAST_MENU
    FAST_MENU = fast_menu

def emit(text: str, delay: float = 0.02, newline: bool = True):
    if FAST_MENU:
        print(text, end='' if not newline else '\n')
    else:
        stream_text(text, delay=delay, newline=newline)

def display_banner():
    lines = [
        '=' * 60,
        '🤖 Generic Ollama Agent - Modular AI Assistant Platform',
        '🚀 Choose your AI model and start coding!',
        '=' * 60,
    ]
    for line in lines:
        emit(line, delay=0.02)

def format_capabilities(cap: dict) -> str:
    icons = []
    if cap.get('coding'): icons.append('💻 Coding')
    if cap.get('file_operations'): icons.append('📁 Files')
    if cap.get('streaming'): icons.append('⚡ Streaming')
    if cap.get('general_qa'): icons.append('💬 Q&A')
    return ' | '.join(icons) if icons else ''

def list_agents(config_manager) -> List[str]:
    agents = config_manager.get_available_agents()
    if not agents:
        stream_text('❌ No agents configured. Please check config/agents.yaml')
        return []
    emit('\n📋 Available AI Agents:')
    emit('-' * 40, delay=0.01)
    ordered: List[str] = []
    for idx, (key, cfg) in enumerate(agents.items(), 1):
        name = cfg.get('name', key)
        desc = cfg.get('description', 'No description available')
        backend = cfg.get('backend_image') or cfg.get('model_id', 'Unknown')
        emit(f"\n{idx}. {name}")
        emit(f"   ID: {backend}", delay=0.005)
        emit(f"   📝 {desc}", delay=0.005)
        caps = format_capabilities(cfg.get('capabilities', {}))
        if caps:
            emit(f"   🎯 {caps}", delay=0.005)
        ordered.append(key)
    return ordered

def display_agent_info(cfg: dict):
    name = cfg.get('name', 'Unknown')
    backend = cfg.get('backend_image') or cfg.get('model_id', 'Unknown')
    provider = cfg.get('provider', 'local').title()
    sections: List[Tuple[str, str]] = [
        ('✅ Agent Selected:', name),
        ('🔧 Backend Image:', backend),
        ('🌐 Provider:', provider),
        ('🎯 Status:', 'Initializing...'),
    ]
    if FAST_MENU:
        for t, c in sections:
            print(f"{t} {c}")
    else:
        progressive_reveal(sections, section_delay=0.2)
