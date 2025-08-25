#!/usr/bin/env python3
"""Main entry point for the DeepCoder agent system."""

import sys
import os

# Get the absolute path to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the src directory to Python path
src_path = os.path.join(script_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from config.arguments import get_parser, load_parameters
from core.helpers import validate_repository_requirement
from core.enums import AgentCapability
from utils.enhanced_logging import get_logger, log_agent_start


def main():
    """Main application entry point."""
    # Clear the console for a clean start
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Print welcome banner
    print("="*80)
    print("🤖 OLLAMA AGENTS - Intelligent Coding Assistant".center(80))
    print("="*80)
    print()
    
    # Initialize enhanced logging
    logger = get_logger(enable_console=True)
    logger.info("Application started")
    
    try:
        # Store original working directory
        original_cwd = os.getcwd()
        
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
            
        if hasattr(args, 'list_models') and args.list_models:
            print("🤖 Models with Agent Implementation Status:")
            
            # Get models from integrations
            from integrations import IntegrationManager, AgentRegistry
            manager = IntegrationManager()
            registry = AgentRegistry()
            
            models = manager.get_all_models()
            matches = registry.match_models_with_agents(models)
            
            if matches:
                # Group by agent status
                with_agents = [m for m in matches if m.has_agent]
                without_agents = [m for m in matches if not m.has_agent]
                
                if with_agents:
                    print(f"\n✅ Models WITH Agent Implementation ({len(with_agents)}):")
                    for match in sorted(with_agents, key=lambda x: x.match_confidence, reverse=True):
                        confidence_icon = "🎯" if match.match_confidence > 0.8 else "✓"
                        agent_name = match.agent_info.name if match.agent_info else "unknown"
                        size = match.model_info.get('details', {}).get('size', 'Unknown')
                        print(f"  {confidence_icon} {match.model_id} → {agent_name} ({size}) - {match.match_reason}")
                
                if without_agents:
                    print(f"\n❌ Models WITHOUT Agent Implementation ({len(without_agents)}):")
                    for match in without_agents[:10]:  # Limit to first 10
                        size = match.model_info.get('details', {}).get('size', 'Unknown')
                        family = match.model_info.get('details', {}).get('family', 'unknown')
                        print(f"  ⚠️  {match.model_id} ({size}) [{family}] - No compatible agent found")
                    
                    if len(without_agents) > 10:
                        print(f"  ... and {len(without_agents) - 10} more")
                
                print(f"\n📊 Summary: {len(with_agents)} ready, {len(without_agents)} need agents")
            else:
                print("  ❌ No models found")
            return
            
        if hasattr(args, 'list_agents') and args.list_agents:
            print("Available agents:")
            print("  - deepcoder (coding assistant)")
            print("  - phi3_mini (compact efficient assistant)")
            print("  - interceptor (prompt analysis)")
            print("  - coder (general coding)")
            print("  - assistant (general purpose)")
            return
        
        if hasattr(args, 'create_models_config') and args.create_models_config:
            print("Creating sample models.txt configuration file...")
            from core.model_source import ModelSource
            source = ModelSource()
            source.create_sample_models_txt()
            return
        
        # Handle server command early (before agent validation)
        if hasattr(args, 'server') and args.server:
            print("🚀 Starting API Server...")
            host = getattr(args, 'host', '0.0.0.0')
            port = getattr(args, 'port', 8000)
            print(f"📡 Server will run on http://{host}:{port}")
            
            # Import and run the server
            from core.server import app
            import uvicorn
            uvicorn.run(app, host=host, port=port)
            return
        
        # Agent resolution logic
        resolved_agent = args.agent
        if not args.agent:
            # If no agent specified, try to resolve the best one for the query
            if hasattr(args, 'query') and args.query:
                print("🔍 No agent specified, analyzing query to find best suitable model...")
                
                from core.agent_resolver import create_agent_resolver
                resolver = create_agent_resolver(max_size_b=14.0)
                resolved_agent = resolver.resolve_best_agent(args.query)
                
                if resolved_agent:
                    print(f"✅ Auto-selected agent: {resolved_agent}")
                    
                    # Show recommendations for transparency
                    recommendations = resolver.get_model_recommendations(args.query, top_n=3)
                    if len(recommendations) > 1:
                        print("🎯 Top model recommendations:")
                        for i, rec in enumerate(recommendations[:3], 1):
                            size = rec.size_in_billions
                            size_str = f" ({size}B)" if size > 0 else ""
                            score_str = f"score: {rec.score:.1f}"
                            selected = "👑 " if i == 1 else f"   {i}. "
                            reasoning = ", ".join(rec.reasoning[:2])  # Show first 2 reasons
                            print(f"{selected}{rec.model_id}{size_str} - {score_str} ({reasoning})")
                        print()
                else:
                    print("❌ Could not auto-resolve suitable agent for your query.")
                    print("Please specify an agent manually using --agent <name>")
                    print("Available options: --list-agents")
                    sys.exit(1)
            else:
                print("Error: No agent specified. Use --agent <name> or see available options with --list-agents")
                print("For automatic agent selection, provide a query: --query 'your question'")
                sys.exit(1)
        
        print(f"Starting agent: {resolved_agent}")
        
        # Update args.agent to use resolved agent for the rest of the flow
        args.agent = resolved_agent
        
        # Get git URL if provided (check multiple possible argument names)
        git_url = None
        if hasattr(args, 'git_repo') and args.git_repo:
            git_url = args.git_repo
        elif hasattr(args, 'g') and args.g:
            git_url = args.g
        
        # Check if agent supports coding and enforce repository requirement
        from core.helpers import check_agent_supports_coding
        agent_supports_coding = check_agent_supports_coding(args.agent)
        
        # For command-line mode with query, make repository optional for coding agents
        if hasattr(args, 'query') and args.query and agent_supports_coding and not git_url:
            print(f"💡 Info: Agent '{args.agent}' is a coding agent and works best with a git repository.")
            print("   You can provide a git repository URL using the -g flag for enhanced functionality:")
            print("   Example: python main.py --agent deepcoder -g https://github.com/user/repo.git --query 'your question'")
            print("   Continuing without repository...\n")
        
        # For interactive mode with coding agents, we'll prompt for repository later
        
        # Validate repository requirement for coding agents
        try:
            # Use original working directory for data folder
            data_path = os.path.join(original_cwd, "data")
            # Make repository optional for command-line queries with coding agents
            is_command_line_query = hasattr(args, 'query') and args.query
            optional_repo = is_command_line_query and agent_supports_coding and not git_url
            # Get interception mode for repository handling
            interception_mode = getattr(args, 'interception_mode', 'smart')
            validation_passed, working_dir = validate_repository_requirement(
                args.agent, ".", git_url, data_path, optional=optional_repo, interception_mode=interception_mode
            )
            if working_dir != ".":
                print(f"✓ Repository cloned and validated: {working_dir}")
                # Change to the cloned directory for the agent to work in
                os.chdir(working_dir)
                print(f"✓ Changed working directory to: {working_dir}")
            else:
                if optional_repo and agent_supports_coding:
                    print(f"✓ Running agent '{args.agent}' without repository (optional mode)")
                else:
                    print(f"✓ Repository validation passed for agent '{args.agent}'")
        except ValueError as e:
            print(f"✗ Repository validation failed: {e}")
            sys.exit(1)
        
        # If git flag is provided, show repository info
        if hasattr(args, 'git') and args.git:
            from core.helpers import get_repository_info
            repo_info = get_repository_info()
            if repo_info:
                print(f"Repository info: {repo_info}")
            else:
                print("No git repository found")
        
        # Display parsed arguments if verbose
        if hasattr(args, 'verbose') and args.verbose:
            print(f"Parsed arguments: {vars(args)}")
        
        # Initialize and run the selected agent
        print(f"🤖 Initializing agent: {args.agent}")
        
        # Handle fast mode flags
        fast_mode = hasattr(args, 'fast') and args.fast
        fast_all_mode = hasattr(args, 'fast_all') and args.fast_all
        
        # Default streaming mode (enabled unless -fa flag is used)
        stream_mode = not fast_all_mode
        
        if fast_all_mode:
            print("⚡ Fast all mode: Disabling tokenization & streaming everywhere")
            logger.info("Fast all mode enabled - streaming disabled")
        elif fast_mode:
            print("🚀 Fast menu mode: Skipping tokenized output for menus") 
            logger.info("Fast menu mode enabled - streaming enabled for agents")
        else:
            print("🎬 Streaming mode enabled (use -fa to disable)")
            logger.info("Default streaming mode enabled")
        
        # Handle query mode (direct chat input)
        if hasattr(args, 'query') and args.query:
            # Start the agent with the initial query using single query mode
            from core.single_query_mode import run_single_query
            try:
                connection_mode = getattr(args, 'connection_mode', 'hybrid')
                interception_mode = getattr(args, 'interception_mode', 'smart')
                force_streaming = getattr(args, 'force_streaming', False) or stream_mode
                collaborative = getattr(args, 'collaborative', False)
                max_iterations = getattr(args, 'max_iterations', 5)
                
                print(f"🔗 Using connection mode: {connection_mode}")
                if force_streaming:
                    print(f"🎬 Streaming mode active")
                    logger.log_agent_operation(
                        agent_name=args.agent,
                        operation="query_mode_start",
                        details={
                            "query": args.query[:100] + "..." if len(args.query) > 100 else args.query,
                            "streaming": True,
                            "collaborative": collaborative
                        }
                    )
                if collaborative:
                    print(f"🤝 Collaborative mode enabled (max {max_iterations} iterations)")
                
                result = run_single_query(
                    args.query, 
                    args.agent, 
                    connection_mode, 
                    git_url, 
                    interception_mode, 
                    force_streaming,
                    collaborative=collaborative,
                    max_iterations=max_iterations
                )
                print(result)
                logger.log_agent_operation(
                    agent_name=args.agent,
                    operation="query_completed",
                    details={"result_length": len(result) if result else 0}
                )
            except Exception as e:
                print(f"❌ Error running query: {e}")
                logger.error(f"Query execution failed for agent {args.agent}", e)
                if hasattr(args, 'verbose') and args.verbose:
                    import traceback
                    traceback.print_exc()
        else:
            # Interactive mode
            print(f"🔄 Interactive mode for agent: {args.agent}")
            logger.log_agent_operation(
                agent_name=args.agent,
                operation="interactive_mode_start",
                details={
                    "streaming": stream_mode,
                    "fast_mode": fast_mode,
                    "fast_all_mode": fast_all_mode
                }
            )
            
            # Start the generic interactive session (it will handle repository prompting internally)
            from core.generic_interactive_mode import run_interactive_session
            try:
                run_interactive_session(args.agent, stream_mode=stream_mode)
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
