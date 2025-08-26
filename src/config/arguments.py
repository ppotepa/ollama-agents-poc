"""Argument parsing with JSON configuration support."""

import argparse
import json
from pathlib import Path
from typing import Any, Optional


def load_parameters(config_path: Optional[str] = None) -> dict[str, Any]:
    """
    Load parameters from JSON configuration file.

    Args:
        config_path: Path to parameters.json file. If None, looks in default locations.

    Returns:
        Dictionary containing parameter definitions
    """
    if config_path is None:
        # Look for parameters.json in config directory
        current_dir = Path(__file__).parent
        config_path = current_dir / "parameters.json"

        if not config_path.exists():
            # Fallback to empty config
            return {"arguments": {}}

    try:
        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)
            return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load parameters from {config_path}: {e}")
        return {"arguments": {}}


def _add_arguments_from_dict(parser: argparse.ArgumentParser, args_dict: dict[str, Any]) -> None:
    """Add arguments to parser from dictionary configuration."""
    for arg_name, arg_config in args_dict.items():
        # Prepare argument name (add -- prefix for long options)
        if not arg_name.startswith('-'):
            arg_name = f"--{arg_name}"

        # Extract argument properties
        arg_type = arg_config.get("type", "str")
        help_text = arg_config.get("help", "")
        default = arg_config.get("default")
        required = arg_config.get("required", False)
        choices = arg_config.get("choices")
        action = arg_config.get("action")
        dest = arg_config.get("dest")
        metavar = arg_config.get("metavar")

        # Convert type string to actual type
        type_map = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool
        }
        actual_type = type_map.get(arg_type, str)

        # Prepare keyword arguments for add_argument
        kwargs = {
            "help": help_text
        }

        if action:
            kwargs["action"] = action
        elif arg_type != "bool":
            kwargs["type"] = actual_type

        if default is not None:
            kwargs["default"] = default

        if required:
            kwargs["required"] = required

        if choices:
            kwargs["choices"] = choices

        if dest:
            kwargs["dest"] = dest

        if metavar:
            kwargs["metavar"] = metavar

        # Add the argument
        parser.add_argument(arg_name, **kwargs)


def get_parser(parameters: Optional[dict[str, Any]] = None) -> argparse.ArgumentParser:
    """
    Create and configure argument parser with JSON-based configuration.

    Args:
        parameters: Parameter configuration dictionary. If None, loads from default file.

    Returns:
        Configured ArgumentParser instance
    """
    if parameters is None:
        parameters = load_parameters()

    # Create base parser
    parser = argparse.ArgumentParser(
        description="DeepCoder Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Add arguments from configuration
    arguments_config = parameters.get("arguments", {})
    if arguments_config:
        _add_arguments_from_dict(parser, arguments_config)
    else:
        # Fallback: add basic arguments manually
        parser.add_argument("--agent", type=str, required=True,
                          help="Agent to run (e.g., deepcoder, assistant)")
        parser.add_argument("--model", type=str, default="llama3:latest",
                          help="Model to use for the agent")
        parser.add_argument("--verbose", action="store_true",
                          help="Enable verbose output")
        parser.add_argument("--interception-mode", type=str, choices=["full", "lightweight", "smart"],
                          default="smart", help="Prompt interception mode: full (complete analysis), lightweight (fast), or smart (adaptive)")
        parser.add_argument("--git", action="store_true",
                          help="Show git repository information")
        parser.add_argument("--output", type=str,
                          help="Output file path")
        parser.add_argument("--config", type=str,
                          help="Configuration file path")

    return parser


def main():
    """Test the argument parser."""
    parameters = load_parameters()
    parser = get_parser(parameters)
    args = parser.parse_args()
    print(f"Parsed arguments: {vars(args)}")


if __name__ == "__main__":
    main()
