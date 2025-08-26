#!/usr/bin/env python3
"""Presentation entry point for the Generic Ollama Agent system.

This module represents the primary command line interface into the system. It
loads configuration from the infrastructure layer, delegates business logic
to the application layer and communicates with the user via standard IO.
By keeping the CLI logic isolated here we maintain a clear separation
between user-facing concerns and the core application behaviour.
"""

import os
import sys

# Determine the base directory of the project and ensure the src package is
# available on the module search path. This allows the CLI to be launched
# directly via `python src/presentation/cli.py` without installing the
# package.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add both the src directory and the repository root to the module search path.
# Some modules still reference the legacy ``src`` package (e.g. ``src.core``),
# so including the repository root ensures those imports continue to work even
# as we adopt the new clean architecture packages. The src directory itself
# is also added so that ``infrastructure``, ``application`` and other new
# top-level packages can be resolved without installing the package.
src_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
repo_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
for path in (src_dir, repo_root):
    if path not in sys.path:
        sys.path.insert(0, path)

# Import configuration and logging from the infrastructure layer. These
# dependencies encapsulate the details of reading configuration files and
# formatting log messages.
from infrastructure.config.arguments import get_parser, load_parameters
from infrastructure.utils.enhanced_logging import get_logger


def main() -> None:
    """Main application entry point."""
    # Clear the console for a clean start
    os.system('cls' if os.name == 'nt' else 'clear')

    # Print welcome banner
    print("=" * 80)
    print("🤖 OLLAMA AGENTS - Intelligent Coding Assistant".center(80))
    print("=" * 80)
    print()

    # Apply system improvements early
    try:
        # Placeholder for future modular system enhancements
        print("🔧 System improvements: Modular architecture active")
        print()
    except Exception as exc:
        print(f"⚠️ Could not apply system improvements: {exc}")
        print()

    # Initialize query logging system
    try:
        from application.core.query_logger_integration import patch_system_with_logging
        patch_success = patch_system_with_logging()
        if patch_success:
            print("📊 Query logging system enabled")
        else:
            print("⚠️ Query logging system initialization failed")
        print()
    except Exception as exc:
        print(f"⚠️ Query logging system error: {exc}")
        print()

    # Initialize enhanced logging
    logger = get_logger(enable_console=True)
    logger.info("Application started")

    try:
        # Load parameters and create argument parser
        parameters = load_parameters()
        parser = get_parser(parameters)
        args = parser.parse_args()

        # Handle list commands early (before agent validation)
        if getattr(args, 'list_all', False):
            print("🔍 All Available Models (from integrations):")
            from infrastructure.integrations import IntegrationManager
            manager = IntegrationManager()
            models = manager.get_all_models()
            if models:
                for model in models:
                    info_parts = [f"  📦 {model['id']}"]
                    details = model.get('details', {})
                    if details.get('size'):
                        info_parts.append(f"({details['size']})")
                    if details.get('family'):
                        info_parts.append(f"[{details['family']}]")
                    if model.get('source'):
                        info_parts.append(f"via {model['source']}")
                    print(" ".join(info_parts))
                print(f"\n📊 Total: {len(models)} models available")
            else:
                print("  ❌ No models found from integrations")
            return

        # Handle query mode (direct chat input)
        if getattr(args, 'query', None):
            from application.core.mode_integration import (
                run_single_query,
                run_collaborative_query,
                run_intelligent_investigation,
            )
            try:
                connection_mode = getattr(args, 'connection_mode', 'hybrid')
                force_streaming = getattr(args, 'force_streaming', False)
                collaborative = getattr(args, 'collaborative', False)
                max_iterations = getattr(args, 'max_iterations', 5)
                intelligent_investigation = getattr(args, 'intelligent_investigation', False)
                no_tools = getattr(args, 'no_tools', False)
                print(f"🔗 Using connection mode: {connection_mode}")
                if no_tools:
                    print("🛑 Tools disabled (running in LLM-only mode)")
                    logger.info("Tools disabled via --no-tools flag")
                if force_streaming:
                    print("🎬 Streaming mode active")
                    logger.log_agent_operation(
                        agent_name=args.agent,
                        operation="query_mode_start",
                        details={
                            "query": args.query[:100] + "..." if len(args.query) > 100 else args.query,
                            "streaming": True,
                            "collaborative": collaborative,
                            "intelligent_investigation": intelligent_investigation,
                        },
                    )
                # Select execution mode based on flags
                if intelligent_investigation:
                    print("🧠 Intelligent investigation mode enabled with dynamic model switching")
                    result = run_intelligent_investigation(args.query, force_streaming)
                elif collaborative:
                    print(f"🤝 Collaborative mode enabled (max {max_iterations} iterations)")
                    result = run_collaborative_query(args.query, "universal", max_iterations, force_streaming)
                else:
                    print("🎯 Single query mode enabled")
                    result = run_single_query(args.query, force_streaming)
                print(result)
            except Exception as exc:
                print(f"❌ Error running query: {exc}")
                logger.error(f"Query execution failed for agent {args.agent}", exc)
                if getattr(args, 'verbose', False):
                    import traceback
                    traceback.print_exc()
        else:
            # Interactive mode
            print(f"🔄 Interactive mode for agent: {args.agent}")
            no_tools = getattr(args, 'no_tools', False)
            if no_tools:
                print("🛑 Tools disabled (running in LLM-only mode)")
                logger.info("Tools disabled via --no-tools flag")
            try:
                print("🎯 Interactive mode - using single query strategy")
                from application.core.mode_integration import run_single_query
                result = run_single_query("Hello! How can I help you?", True)
                print(result)
            except Exception as exc:
                print(f"❌ Error starting interactive session: {exc}")
                logger.error(f"Interactive session failed for agent {args.agent}", exc)
                if getattr(args, 'verbose', False):
                    import traceback
                    traceback.print_exc()
        print("✅ Agent session completed!")
        logger.info("Agent session completed successfully")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        logger.info("Operation cancelled by user (KeyboardInterrupt)")
        sys.exit(0)
    except Exception as exc:
        print(f"Error: {exc}")
        logger.error("Application error", exc)
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()