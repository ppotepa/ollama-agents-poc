"""Main entry point for repository analysis following SOLID principles.

Single Responsibility: Provide simple interface for repository analysis operations.
"""
from __future__ import annotations

from pathlib import Path

# Try relative imports for module use, fallback to absolute for standalone
try:
    from .analyzers.repository_analyzer import RepositoryAnalyzer
    from .formatters.exporters import JSONExporter, MarkdownExporter
    from .formatters.text_formatter import TextFormatter
    from .models.repository_context import AnalysisConfig, RepositoryContext
except ImportError:
    # Standalone execution - add path and use absolute imports
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.repository.analyzers.repository_analyzer import RepositoryAnalyzer
    from src.repository.formatters.exporters import JSONExporter, MarkdownExporter
    from src.repository.formatters.text_formatter import TextFormatter
    from src.repository.models.repository_context import AnalysisConfig, RepositoryContext


def analyze_repository_context(repository_path: str = ".", config: AnalysisConfig | None = None) -> str:
    """Analyze repository context and return human-readable summary.

    Args:
        repository_path: Path to repository root (default: current directory)
        config: Analysis configuration (default: AnalysisConfig.default())

    Returns:
        Human-readable repository context summary
    """
    try:
        analyzer = RepositoryAnalyzer(repository_path, config)
        context = analyzer.analyze_repository()
        formatter = TextFormatter()
        return formatter.format_repository_summary(context)
    except Exception as e:
        return f"❌ Error analyzing repository: {e}"


def analyze_repository_detailed(repository_path: str = ".", config: AnalysisConfig | None = None) -> RepositoryContext:
    """Analyze repository and return detailed context object.

    Args:
        repository_path: Path to repository root (default: current directory)
        config: Analysis configuration (default: AnalysisConfig.default())

    Returns:
        Complete RepositoryContext object with all analysis results
    """
    analyzer = RepositoryAnalyzer(repository_path, config)
    return analyzer.analyze_repository()


def save_analysis_to_json(repository_path: str, output_path: str, config: AnalysisConfig | None = None) -> None:
    """Analyze repository and save results to JSON file.

    Args:
        repository_path: Path to repository root
        output_path: Path where to save JSON results
        config: Analysis configuration (default: AnalysisConfig.default())
    """
    context = analyze_repository_detailed(repository_path, config)
    JSONExporter.save_to_file(context, output_path)


def save_analysis_to_markdown(repository_path: str, output_path: str, config: AnalysisConfig | None = None) -> None:
    """Analyze repository and save results to Markdown file.

    Args:
        repository_path: Path to repository root
        output_path: Path where to save Markdown results
        config: Analysis configuration (default: AnalysisConfig.default())
    """
    context = analyze_repository_detailed(repository_path, config)
    MarkdownExporter.save_to_file(context, output_path)


def get_files_by_language(repository_path: str, language: str, config: AnalysisConfig | None = None) -> list:
    """Get all files of a specific programming language.

    Args:
        repository_path: Path to repository root
        language: Programming language name
        config: Analysis configuration (default: AnalysisConfig.default())

    Returns:
        List of FileInfo objects for the specified language
    """
    analyzer = RepositoryAnalyzer(repository_path, config)
    return analyzer.get_files_by_language(language)


def get_language_breakdown(repository_path: str, config: AnalysisConfig | None = None) -> str:
    """Get detailed language breakdown for repository.

    Args:
        repository_path: Path to repository root
        config: Analysis configuration (default: AnalysisConfig.default())

    Returns:
        Formatted language breakdown string
    """
    try:
        context = analyze_repository_detailed(repository_path, config)
        formatter = TextFormatter()
        return formatter.format_language_breakdown(context)
    except Exception as e:
        return f"❌ Error analyzing languages: {e}"


def get_directory_structure(repository_path: str, max_depth: int = 3, config: AnalysisConfig | None = None) -> str:
    """Get formatted directory structure overview.

    Args:
        repository_path: Path to repository root
        max_depth: Maximum depth to show in structure
        config: Analysis configuration (default: AnalysisConfig.default())

    Returns:
        Formatted directory structure string
    """
    try:
        context = analyze_repository_detailed(repository_path, config)
        formatter = TextFormatter()
        return formatter.format_directory_structure(context, max_depth)
    except Exception as e:
        return f"❌ Error analyzing directory structure: {e}"


# CLI interface for standalone usage
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Analyze repository structure and content")
    parser.add_argument("path", nargs="?", default=".", help="Repository path to analyze")
    parser.add_argument("--output", "-o", help="Output file path (JSON or Markdown)")
    parser.add_argument("--format", "-f", choices=["json", "markdown"], default="text",
                       help="Output format")
    parser.add_argument("--languages", "-l", action="store_true",
                       help="Show detailed language breakdown")
    parser.add_argument("--structure", "-s", action="store_true",
                       help="Show directory structure")
    parser.add_argument("--depth", "-d", type=int, default=3,
                       help="Maximum depth for directory structure")
    parser.add_argument("--no-git", action="store_true",
                       help="Skip Git information")
    parser.add_argument("--no-binary", action="store_true",
                       help="Skip binary files")

    args = parser.parse_args()

    # Create configuration
    config = AnalysisConfig.default()
    if args.no_git:
        config.include_git_info = False
    if args.no_binary:
        config.include_binary_files = False

    try:
        if args.languages:
            result = get_language_breakdown(args.path, config)
        elif args.structure:
            result = get_directory_structure(args.path, args.depth, config)
        elif args.output:
            if args.format == "json":
                save_analysis_to_json(args.path, args.output, config)
                print(f"✅ Analysis saved to {args.output}")
                sys.exit(0)
            elif args.format == "markdown":
                save_analysis_to_markdown(args.path, args.output, config)
                print(f"✅ Analysis saved to {args.output}")
                sys.exit(0)
        else:
            result = analyze_repository_context(args.path, config)

        print(result)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
