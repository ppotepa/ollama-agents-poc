"""Agent display and initialization utilities."""

from src.config.settings import config_manager
from src.core import registry as agent_registry
from src.utils.animations import progressive_reveal, show_thinking_animation, stream_text


def display_agent_info(agent_key, fast_menu: bool = False):
    """Display selected agent configuration details."""
    cfg = config_manager.get_agent_config(agent_key)
    if not cfg:
        _emit(f"❌ Agent configuration not found for: {agent_key}", fast_menu=fast_menu)
        return False
    name = cfg.get("name", agent_key)
    backend = cfg.get("backend_image") or cfg.get("model_id", "Unknown")
    provider = cfg.get("provider", "local")
    sections = [
        ("✅ Agent Selected:", name),
        ("🔧 Backend Image:", backend),
        ("🌐 Provider:", provider.title()),
        ("🎯 Status:", "Initializing...")
    ]
    if fast_menu:
        # Immediate display without progressive animation
        for title, content in sections:
            print(f"{title} {content}")
    else:
        progressive_reveal(sections, section_delay=0.2)
    return True


def check_agent_availability(agent_key, fast_menu: bool = False):
    """Check if agent backend is available."""
    if not fast_menu:
        show_thinking_animation(1.0, "🔍 Checking agent backend")
    _emit("✅ Backend assumed available (stub)", fast_menu=fast_menu)
    return True


def initialize_agent(agent_key, fast_menu: bool = False):
    """Create and load the agent; return agent instance or None."""
    cfg = config_manager.get_agent_config(agent_key)
    name = cfg.get("name", agent_key)
    if not fast_menu:
        show_thinking_animation(1.0, f"🚀 Initializing {name}")
    try:
        agent = agent_registry.create(agent_key, cfg)
        agent.load()
        tools = agent.tool_names()
        _emit(f"🎉 {name} initialized! Tools: {', '.join(tools) if tools else 'none'}", fast_menu=fast_menu)
        return agent
    except Exception as e:  # pragma: no cover
        _emit(f"❌ Initialization failed: {e}", fast_menu=fast_menu)
        return None


def _emit(text: str, fast_menu: bool = False):
    """Simple text emitter."""
    if fast_menu:
        print(text)
    else:
        stream_text(text)
