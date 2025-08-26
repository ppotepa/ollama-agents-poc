"""UI banner and agent listing functionality."""

from src.config.settings import config_manager
from src.utils.animations import stream_text


def _emit(text: str, delay: float = 0.02, newline: bool = True, fast_menu: bool = False):
    """Unified emitter respecting fast menu flags."""
    if fast_menu:
        print(text, end='' if not newline else '\n')
    else:
        stream_text(text, delay=delay, newline=newline)


def display_banner(fast_menu: bool = False):
    """Display the Generic Ollama Agent banner."""
    banner_lines = [
        "=" * 60,
        "🤖 Generic Ollama Agent - Modular AI Assistant Platform",
        "🚀 Choose your AI model and start coding!",
        "=" * 60
    ]

    for line in banner_lines:
        _emit(line, delay=0.02, fast_menu=fast_menu)


def display_agent_list(fast_menu: bool = False):
    """Display available agents and return ordered list of keys."""
    agents = config_manager.get_available_agents()
    if not agents:
        stream_text("❌ No agents configured. Please check config/agents.yaml")
        return []

    _emit("\n📋 Available AI Agents:", delay=0.02, fast_menu=fast_menu)
    _emit("-" * 40, delay=0.01, fast_menu=fast_menu)
    ordered = []
    for idx, (key, cfg) in enumerate(agents.items(), 1):
        name = cfg.get("name", key)
        desc = cfg.get("description", "No description available")
        backend = cfg.get("backend_image") or cfg.get("model_id", "Unknown")
        _emit(f"\n{idx}. {name}", delay=0.01, fast_menu=fast_menu)
        _emit(f"   ID: {backend}", delay=0.005, fast_menu=fast_menu)
        _emit(f"   📝 {desc}", delay=0.005, fast_menu=fast_menu)
        cap = cfg.get("capabilities", {})
        icons = []
        if cap.get("coding"):
            icons.append("💻 Coding")
        if cap.get("file_operations"):
            icons.append("📁 Files")
        if cap.get("streaming"):
            icons.append("⚡ Streaming")
        if cap.get("general_qa"):
            icons.append("💬 Q&A")
        if icons:
            _emit(f"   🎯 {' | '.join(icons)}", delay=0.005, fast_menu=fast_menu)
        ordered.append(key)
    return ordered


def get_user_choice(agent_keys, fast_menu: bool = False):
    """Interactive selection from list of agent keys."""
    while True:
        try:
            _emit(f"\n🔘 Choose an agent (1-{len(agent_keys)}) or 'q' to quit: ",
                  delay=0.01, newline=False, fast_menu=fast_menu)
            choice = input().strip().lower()

            if choice == 'q' or choice == 'quit':
                return None

            choice_num = int(choice)
            if 1 <= choice_num <= len(agent_keys):
                return agent_keys[choice_num - 1]
            else:
                _emit(f"❌ Enter a number between 1 and {len(agent_keys)}", fast_menu=fast_menu)

        except ValueError:
            _emit("❌ Please enter a valid number or 'q' to quit", fast_menu=fast_menu)
        except KeyboardInterrupt:
            _emit("\n👋 Goodbye!", fast_menu=fast_menu)
            return None
