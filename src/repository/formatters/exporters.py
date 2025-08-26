"""Data export services for repository analysis results.

Single Responsibility: Export repository analysis results to various formats.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..models.repository_context import RepositoryContext


class JSONExporter:
    """Exports repository analysis results to JSON format.

    Responsibility: Convert analysis data to JSON format for storage/API consumption.
    """

    @staticmethod
    def export_to_dict(context: RepositoryContext) -> dict[str, Any]:
        """Convert RepositoryContext to dictionary.

        Args:
            context: Repository analysis results

        Returns:
            Dictionary representation of the context
        """
        return asdict(context)

    @staticmethod
    def export_to_json_string(context: RepositoryContext, indent: int = 2) -> str:
        """Convert RepositoryContext to JSON string.

        Args:
            context: Repository analysis results
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        data = JSONExporter.export_to_dict(context)
        return json.dumps(data, indent=indent, default=str)

    @staticmethod
    def save_to_file(context: RepositoryContext, output_path: str, indent: int = 2) -> None:
        """Save RepositoryContext to JSON file.

        Args:
            context: Repository analysis results
            output_path: Path where to save the JSON file
            indent: JSON indentation level
        """
        json_str = JSONExporter.export_to_json_string(context, indent)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)

    @staticmethod
    def load_from_file(input_path: str) -> RepositoryContext:
        """Load RepositoryContext from JSON file.

        Args:
            input_path: Path to the JSON file

        Returns:
            RepositoryContext object
        """
        with open(input_path, encoding='utf-8') as f:
            data = json.load(f)
            return RepositoryContext(**data)


class CSVExporter:
    """Exports file information to CSV format.

    Responsibility: Export file data as CSV for spreadsheet analysis.
    """

    @staticmethod
    def export_files_to_csv(context: RepositoryContext, output_path: str) -> None:
        """Export file information to CSV format.

        Args:
            context: Repository analysis results
            output_path: Path where to save the CSV file
        """
        import csv

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['path', 'size', 'lines', 'language', 'mime_type', 'is_binary', 'last_modified']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for file_info in context.files:
                writer.writerow(asdict(file_info))

    @staticmethod
    def export_languages_to_csv(context: RepositoryContext, output_path: str) -> None:
        """Export language statistics to CSV format.

        Args:
            context: Repository analysis results
            output_path: Path where to save the CSV file
        """
        import csv

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['language', 'file_count', 'percentage']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            total_files = sum(context.languages.values()) if context.languages else 0

            for language, count in sorted(context.languages.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_files * 100) if total_files > 0 else 0
                writer.writerow({
                    'language': language,
                    'file_count': count,
                    'percentage': f"{percentage:.2f}%"
                })


class MarkdownExporter:
    """Exports repository analysis results to Markdown format.

    Responsibility: Generate Markdown documentation from analysis results.
    """

    @staticmethod
    def export_to_markdown(context: RepositoryContext) -> str:
        """Convert RepositoryContext to Markdown format.

        Args:
            context: Repository analysis results

        Returns:
            Markdown string representation
        """
        lines = []
        repo_name = Path(context.root_path).name

        # Header
        lines.append(f"# Repository Analysis: {repo_name}")
        lines.append("")

        # Overview
        lines.append("## Overview")
        lines.append("")
        lines.append(f"- **Files**: {context.total_files:,}")
        lines.append(f"- **Directories**: {context.total_directories:,}")
        lines.append(f"- **Total Size**: {JSONExporter._format_size(context.total_size)}")
        lines.append(f"- **Lines of Code**: {context.total_lines:,}")
        lines.append("")

        # Git Information
        if context.git_info:
            lines.append("## Git Information")
            lines.append("")
            for key, value in context.git_info.items():
                formatted_key = key.replace('_', ' ').title()
                lines.append(f"- **{formatted_key}**: {value}")
            lines.append("")

        # Programming Languages
        if context.languages:
            lines.append("## Programming Languages")
            lines.append("")
            lines.append("| Language | Files | Percentage |")
            lines.append("|----------|--------|------------|")

            total_files = sum(context.languages.values())
            for lang, count in sorted(context.languages.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_files * 100) if total_files > 0 else 0
                lines.append(f"| {lang} | {count} | {percentage:.1f}% |")
            lines.append("")

        # Largest Files
        if context.largest_files:
            lines.append("## Largest Files")
            lines.append("")
            lines.append("| File | Size |")
            lines.append("|------|------|")

            for file_info in context.largest_files[:10]:
                size = JSONExporter._format_size(file_info.size)
                lines.append(f"| `{file_info.path}` | {size} |")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def save_to_file(context: RepositoryContext, output_path: str) -> None:
        """Save repository analysis as Markdown file.

        Args:
            context: Repository analysis results
            output_path: Path where to save the Markdown file
        """
        markdown_content = MarkdownExporter.export_to_markdown(context)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)


# Helper function for size formatting (shared across exporters)
def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


# Add the helper to JSONExporter for use in Markdown
JSONExporter._format_size = _format_size
