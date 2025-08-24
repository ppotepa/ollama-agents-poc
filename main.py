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
        
        print(f"Starting agent: {args.agent}")
        
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
        
        # Here you would typically initialize and run the selected agent
        print(f"Agent '{args.agent}' would be initialized here...")
        
        # Example of enum-based processing
        print("Available model families:")
        from core.enums import ModelFamily
        for family in ModelFamily:
            print(f"  - {family.name}")
        
        print("\nDemo complete!")
        
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
