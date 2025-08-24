#!/usr/bin/env python3
"""Main entry point for the DeepCoder agent system."""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.arguments import get_parser, load_parameters
from core.repository_validation import validate_repository_requirement
from core.enums import AgentCapability


def main():
    """Main application entry point."""
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
        
        print(f"Starting agent: {args.agent}")
        
        # Ensure agent is specified
        if not args.agent:
            print("Error: No agent specified. Use --agent <name> or see available options with --list-agents")
            sys.exit(1)
        
        # Get git URL if provided (check multiple possible argument names)
        git_url = None
        if hasattr(args, 'git_repo') and args.git_repo:
            git_url = args.git_repo
        elif hasattr(args, 'g') and args.g:
            git_url = args.g
        
        # Check if agent supports coding and enforce repository requirement
        from core.repository_validation import check_agent_supports_coding
        agent_supports_coding = check_agent_supports_coding(args.agent)
        
        # For command-line mode with query, require -g flag for coding agents
        if hasattr(args, 'query') and args.query and agent_supports_coding and not git_url:
            print(f"❌ Error: Agent '{args.agent}' is a coding agent and requires a git repository.")
            print("   Please provide a git repository URL using the -g flag:")
            print("   Example: python main.py --agent deepcoder -g https://github.com/user/repo.git --query 'your question'")
            sys.exit(1)
        
        # For interactive mode with coding agents, we'll prompt for repository later
        
        # Validate repository requirement for coding agents
        try:
            # Use original working directory for data folder
            data_path = os.path.join(original_cwd, "data")
            validation_passed, working_dir = validate_repository_requirement(args.agent, ".", git_url, data_path)
            if working_dir != ".":
                print(f"✓ Repository cloned and validated: {working_dir}")
                # Change to the cloned directory for the agent to work in
                os.chdir(working_dir)
                print(f"✓ Changed working directory to: {working_dir}")
            else:
                print(f"✓ Repository validation passed for agent '{args.agent}'")
        except ValueError as e:
            print(f"✗ Repository validation failed: {e}")
            sys.exit(1)
        
        # If git flag is provided, show repository info
        if hasattr(args, 'git') and args.git:
            from core.repository_validation import get_repository_info
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
        
        if fast_all_mode:
            print("⚡ Fast all mode: Disabling tokenization & streaming everywhere")
        elif fast_mode:
            print("🚀 Fast menu mode: Skipping tokenized output for menus")
        
        # Handle query mode (direct chat input)
        if hasattr(args, 'query') and args.query:
            print(f"💬 Query mode: {args.query}")
            # Start the agent with the initial query
            from core.generic_interactive import run_single_query
            try:
                result = run_single_query(args.query, args.agent)
                print(f"🤖 Agent response: {result}")
            except Exception as e:
                print(f"❌ Error running query: {e}")
                if hasattr(args, 'verbose') and args.verbose:
                    import traceback
                    traceback.print_exc()
        else:
            # Interactive mode
            print(f"🔄 Interactive mode for agent: {args.agent}")
            
            # Start the generic interactive session (it will handle repository prompting internally)
            from core.generic_interactive import run_interactive_session
            try:
                run_interactive_session(args.agent)
            except Exception as e:
                print(f"❌ Error starting interactive session: {e}")
                if hasattr(args, 'verbose') and args.verbose:
                    import traceback
                    traceback.print_exc()
            
        print("✅ Agent session completed!")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
