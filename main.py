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
        if hasattr(args, 'list_models') and args.list_models:
            print("Available models:")
            
            # Use the new model source system
            from core.model_source import get_available_models
            models = get_available_models()
            
            if models:
                for model in models:
                    info_parts = [f"  - {model.name}"]
                    if model.family:
                        info_parts.append(f"({model.family})")
                    if model.description:
                        info_parts.append(f"- {model.description}")
                    if model.size:
                        info_parts.append(f"[{model.size}]")
                    print(" ".join(info_parts))
            else:
                print("  No models found")
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
            # This would normally start the agent with the initial query
            print(f"🎯 Agent '{args.agent}' would process query: '{args.query}'")
        else:
            # Interactive mode
            print(f"🔄 Interactive mode for agent: {args.agent}")
            
            # Show available models
            print("\nAvailable model families:")
            from core.enums import ModelFamily
            for family in ModelFamily:
                print(f"  - {family.name}")
            
            # This is where the actual agent would start interactive mode
            print(f"\n🎯 Agent '{args.agent}' would start interactive session here...")
            print("💡 Use Ctrl+C to exit")
            
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
