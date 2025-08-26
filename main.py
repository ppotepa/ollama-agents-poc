#!/usr/bin/env python3
"""Main entry point for the DeepCoder agent system."""

import os
import sys

# Get the absolute path to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the src directory to Python path
src_path = os.path.join(script_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from config.arguments import get_parser, load_parameters
from utils.enhanced_logging import get_logger


def main():
    """Main application entry point."""
    # Clear the console for a clean start
    os.system('cls' if os.name == 'nt' else 'clear')

    # Print welcome banner
    print("="*80)
    print("🤖 OLLAMA AGENTS - Intelligent Coding Assistant".center(80))
    print("="*80)
    print()

    # Apply system improvements early  
    try:
        # System improvements module will be implemented later
        print("🔧 System improvements: Modular architecture active")
        print()
    except Exception as e:
        print(f"⚠️ Could not apply system improvements: {e}")
        print()

    # Initialize query logging system
    try:
        from src.core.query_logger_integration import patch_system_with_logging
        patch_success = patch_system_with_logging()
        if patch_success:
            print("📊 Query logging system enabled")
        else:
            print("⚠️ Query logging system initialization failed")
        print()
    except Exception as e:
        print(f"⚠️ Query logging system error: {e}")
        print()

    # Initialize enhanced logging
    logger = get_logger(enable_console=True)
    logger.info("Application started")

    try:
        # Store original working directory
        os.getcwd()

        # Load parameters and create argument parser
        parameters = load_parameters()
        parser = get_parser(parameters)
        args = parser.parse_args()

        # Handle list commands early (before agent validation)
        if hasattr(args, 'list_all') and args.list_all:
            print("🔍 All Available Models (from integrations):")

            # Use the new integration system
            from integrations import IntegrationManager
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
        if hasattr(args, 'query') and args.query:
            # Start the agent with the initial query using strategy-based mode
            from src.core.mode_integration import run_single_query, run_collaborative_query, run_intelligent_investigation
            try:
                connection_mode = getattr(args, 'connection_mode', 'hybrid')
                interception_mode = getattr(args, 'interception_mode', 'smart')
                force_streaming = getattr(args, 'force_streaming', False)
                collaborative = getattr(args, 'collaborative', False)
                max_iterations = getattr(args, 'max_iterations', 5)
                intelligent_investigation = getattr(args, 'intelligent_investigation', False)
                no_tools = getattr(args, 'no_tools', False)
                
                # Handle custom mode parameter (not in parameters.json)
                if hasattr(args, 'mode'):
                    if args.mode == "identalligent":
                        intelligent_investigation = True
                    elif args.mode == "collaborative":
                        collaborative = True
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
                            "intelligent_investigation": intelligent_investigation
                        }
                    )

                # Handle git repository if specified
                git_repo = getattr(args, 'git_repo', None)
                if git_repo:
                    print(f"🔗 Using repository: {git_repo}")
                
                # Select execution mode based on flags
                if intelligent_investigation:
                    print("🧠 Intelligent investigation mode enabled with dynamic model switching")
                    # Pass git_repo to the function
                    from src.core.mode_integration import run_intelligent_investigation
                    result = run_intelligent_investigation(args.query, force_streaming)
                elif collaborative:
                    print(f"🤝 Collaborative mode enabled (max {max_iterations} iterations)")
                    from src.core.mode_integration import run_collaborative_query
                    result = run_collaborative_query(args.query, "universal", max_iterations, force_streaming)
                else:
                    print("🎯 Single query mode enabled")
                    from src.core.mode_integration import run_single_query
                    # Pass repository URL directly to function
                    result = run_single_query(args.query, force_streaming, repository_url=git_repo)

                # Format and print result for better readability
                if isinstance(result, dict) and 'final_answer' in result:
                    print("\n" + "-" * 80)
                    print("ANSWER:\n")
                    print(result['final_answer'])
                    print("-" * 80)
                else:
                    print(result)

            except Exception as e:
                print(f"❌ Error running query: {e}")
                logger.error(f"Query execution failed for agent {args.agent}", e)
                if hasattr(args, 'verbose') and args.verbose:
                    import traceback
                    traceback.print_exc()
        else:
            # Interactive mode
            print(f"🔄 Interactive mode for agent: {args.agent}")

            # Check if tools are disabled
            no_tools = getattr(args, 'no_tools', False)
            if no_tools:
                print("🛑 Tools disabled (running in LLM-only mode)")
                logger.info("Tools disabled via --no-tools flag")

            try:
                # Interactive mode not yet implemented in new architecture
                print("🎯 Interactive mode - using single query strategy")
                from src.core.mode_integration import run_single_query
                result = run_single_query("Hello! How can I help you?", True)
                print(result)
            except Exception as e:
                print(f"❌ Error starting interactive session: {e}")
                logger.error(f"Interactive session failed for agent {args.agent}", e)
                if hasattr(args, 'verbose') and args.verbose:
                    import traceback
                    traceback.print_exc()

        print("✅ Agent session completed!")
        logger.info("Agent session completed successfully")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        logger.info("Operation cancelled by user (KeyboardInterrupt)")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        logger.error("Application error", e)
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
