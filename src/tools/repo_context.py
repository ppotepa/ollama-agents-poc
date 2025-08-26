"""Repository Context Analyzer - Provides comprehensive repository analysis for coding agents.

This tool analyzes repository structure and provides detailed context including:
- File analysis (counts, sizes, line counts, language detection)
- Directory structure (hierarchical analysis)
- Git integration (branch info, commit counts, remote URLs)
- Language detection and breakdown
- Size analysis and largest files identification
"""

import json
import mimetypes
import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class FileInfo:
    """Individual file metadata."""
    path: str
    size: int
    line_count: int = 0
    language: str = "Unknown"
    mime_type: str = ""
    is_binary: bool = False
    modified_time: str = ""


@dataclass
class DirectoryInfo:
    """Directory metadata."""
    path: str
    file_count: int = 0
    subdirectory_count: int = 0
    total_size: int = 0


@dataclass
class RepositoryContext:
    """Complete repository analysis results."""
    name: str
    path: str
    total_files: int = 0
    total_directories: int = 0
    total_size: int = 0
    total_lines: int = 0

    # Git information
    git_branch: str = ""
    git_remote_url: str = ""
    git_commits: int = 0

    # Language breakdown
    languages: dict[str, int] = field(default_factory=dict)
    language_percentages: dict[str, float] = field(default_factory=dict)

    # File and directory lists
    files: list[FileInfo] = field(default_factory=list)
    directories: list[DirectoryInfo] = field(default_factory=list)

    # Largest files for overview
    largest_files: list[tuple[str, int]] = field(default_factory=list)

    # Analysis metadata
    analysis_time: str = ""


class RepositoryAnalyzer:
    """Analyzes repository structure and content."""

    # Directories to ignore during analysis
    IGNORE_PATTERNS = {
        '__pycache__', '.git', '.svn', '.hg', 'node_modules', 'venv', '.venv',
        'env', '.env', 'build', 'dist', '.dist', 'target', '.target',
        '.idea', '.vscode', '.DS_Store', 'Thumbs.db', '.pytest_cache',
        '.mypy_cache', '.coverage', 'htmlcov', '.tox', '.eggs', '*.egg-info',
        'models', 'data'  # Skip model files and data directories
    }

    # Language detection by file extension
    LANGUAGE_MAP = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.ts': 'TypeScript',
        '.jsx': 'JavaScript',
        '.tsx': 'TypeScript',
        '.java': 'Java',
        '.c': 'C',
        '.cpp': 'C++',
        '.cc': 'C++',
        '.cxx': 'C++',
        '.h': 'C/C++',
        '.hpp': 'C++',
        '.cs': 'C#',
        '.php': 'PHP',
        '.rb': 'Ruby',
        '.go': 'Go',
        '.rs': 'Rust',
        '.swift': 'Swift',
        '.kt': 'Kotlin',
        '.scala': 'Scala',
        '.sh': 'Shell',
        '.bash': 'Bash',
        '.zsh': 'Zsh',
        '.fish': 'Fish',
        '.ps1': 'PowerShell',
        '.bat': 'Batch',
        '.cmd': 'Batch',
        '.html': 'HTML',
        '.htm': 'HTML',
        '.css': 'CSS',
        '.scss': 'SCSS',
        '.sass': 'Sass',
        '.less': 'Less',
        '.xml': 'XML',
        '.json': 'JSON',
        '.yaml': 'YAML',
        '.yml': 'YAML',
        '.toml': 'TOML',
        '.ini': 'INI',
        '.cfg': 'Config',
        '.conf': 'Config',
        '.md': 'Markdown',
        '.rst': 'reStructuredText',
        '.txt': 'Text',
        '.sql': 'SQL',
        '.r': 'R',
        '.R': 'R',
        '.m': 'MATLAB',
        '.pl': 'Perl',
        '.lua': 'Lua',
        '.vim': 'Vim',
        '.dockerfile': 'Dockerfile',
        '.makefile': 'Makefile',
        '.cmake': 'CMake'
    }

    def __init__(self, repo_path: str = "."):
        """Initialize analyzer with repository path."""
        self.repo_path = Path(repo_path).resolve()
        self.repo_name = self.repo_path.name

    def analyze_repository(self) -> RepositoryContext:
        """Perform complete repository analysis."""
        context = RepositoryContext(
            name=self.repo_name,
            path=str(self.repo_path),
            analysis_time=datetime.now().isoformat()
        )

        # Analyze files and directories
        self._analyze_structure(context)

        # Get git information
        self._analyze_git_info(context)

        # Calculate language percentages
        self._calculate_language_percentages(context)

        # Find largest files
        self._find_largest_files(context)

        return context

    def _analyze_structure(self, context: RepositoryContext):
        """Analyze file and directory structure."""
        language_counts = defaultdict(int)

        for root, dirs, files in os.walk(self.repo_path):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not self._should_ignore(d)]

            root_path = Path(root)
            rel_root = root_path.relative_to(self.repo_path)

            # Count directory
            if str(rel_root) != ".":
                context.total_directories += 1
                dir_info = DirectoryInfo(
                    path=str(rel_root),
                    file_count=len(files),
                    subdirectory_count=len(dirs)
                )
                context.directories.append(dir_info)

            # Analyze files
            for file_name in files:
                if self._should_ignore(file_name):
                    continue

                file_path = root_path / file_name
                rel_file_path = file_path.relative_to(self.repo_path)

                try:
                    file_info = self._analyze_file(file_path, str(rel_file_path))
                    context.files.append(file_info)
                    context.total_files += 1
                    context.total_size += file_info.size
                    context.total_lines += file_info.line_count

                    # Count language
                    language_counts[file_info.language] += 1

                except (OSError, PermissionError):
                    # Skip files we can't read
                    continue

        context.languages = dict(language_counts)

    def _analyze_file(self, file_path: Path, rel_path: str) -> FileInfo:
        """Analyze individual file."""
        stat = file_path.stat()

        # Detect language from extension
        language = self._detect_language(file_path)

        # Get MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        mime_type = mime_type or "unknown"

        # Check if binary
        is_binary = self._is_binary_file(file_path)

        # Count lines for text files
        line_count = 0
        if not is_binary:
            line_count = self._count_lines(file_path)

        return FileInfo(
            path=rel_path,
            size=stat.st_size,
            line_count=line_count,
            language=language,
            mime_type=mime_type,
            is_binary=is_binary,
            modified_time=datetime.fromtimestamp(stat.st_mtime).isoformat()
        )

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        suffix = file_path.suffix.lower()

        # Special cases
        if file_path.name.lower() in ['dockerfile', 'makefile', 'cmake']:
            return file_path.name.title()

        return self.LANGUAGE_MAP.get(suffix, "Unknown")

    def _is_binary_file(self, file_path: Path) -> bool:
        """Check if file is binary."""
        try:
            # Skip very large files (>50MB) to avoid hanging
            if file_path.stat().st_size > 50 * 1024 * 1024:
                return True

            with open(file_path, 'rb') as f:
                chunk = f.read(8192)
                return b'\0' in chunk
        except (OSError, PermissionError):
            return True

    def _count_lines(self, file_path: Path) -> int:
        """Count lines in text file."""
        try:
            with open(file_path, encoding='utf-8', errors='ignore') as f:
                return sum(1 for _ in f)
        except (OSError, PermissionError, UnicodeDecodeError):
            return 0

    def _should_ignore(self, name: str) -> bool:
        """Check if file/directory should be ignored."""
        if name.startswith('.') and name not in {'.gitignore', '.gitattributes'}:
            return True
        return name in self.IGNORE_PATTERNS

    def _analyze_git_info(self, context: RepositoryContext):
        """Extract git repository information."""
        try:
            # Get current branch
            result = subprocess.run(
                ['git', 'branch', '--show-current'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                context.git_branch = result.stdout.strip()

            # Get remote URL
            result = subprocess.run(
                ['git', 'config', '--get', 'remote.origin.url'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                context.git_remote_url = result.stdout.strip()

            # Get commit count
            result = subprocess.run(
                ['git', 'rev-list', '--count', 'HEAD'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                context.git_commits = int(result.stdout.strip())

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError):
            # Not a git repository or git not available
            pass

    def _calculate_language_percentages(self, context: RepositoryContext):
        """Calculate language distribution percentages."""
        if not context.languages:
            return

        total_files = sum(context.languages.values())
        context.language_percentages = {
            lang: (count / total_files) * 100
            for lang, count in context.languages.items()
        }

    def _find_largest_files(self, context: RepositoryContext):
        """Find the largest files in the repository."""
        file_sizes = [(f.path, f.size) for f in context.files]
        file_sizes.sort(key=lambda x: x[1], reverse=True)
        context.largest_files = file_sizes[:10]  # Top 10 largest files

    @staticmethod
    def format_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"


class RepositoryContextFormatter:
    """Formats repository context for display."""

    @staticmethod
    def format_summary(context: RepositoryContext) -> str:
        """Format repository context as human-readable summary."""
        lines = []

        # Header
        lines.append(f"📁 Repository: {context.name}")
        lines.append("")

        # Overview
        lines.append("📊 Overview:")
        lines.append(f"   • {context.total_files:,} files")
        lines.append(f"   • {context.total_directories:,} directories")
        lines.append(f"   • {RepositoryAnalyzer.format_size(context.total_size)} total size")
        lines.append(f"   • {context.total_lines:,} lines of code")
        lines.append("")

        # Git info
        if context.git_branch or context.git_remote_url:
            lines.append("🔧 Git Info:")
            if context.git_branch:
                lines.append(f"   • Branch: {context.git_branch}")
            if context.git_commits > 0:
                lines.append(f"   • Commits: {context.git_commits}")
            if context.git_remote_url:
                lines.append(f"   • Remote: {context.git_remote_url}")
            lines.append("")

        # Languages
        if context.languages:
            lines.append("💻 Languages:")
            # Sort by count, descending
            sorted_langs = sorted(
                context.language_percentages.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for lang, percentage in sorted_langs[:10]:  # Top 10 languages
                count = context.languages[lang]
                lines.append(f"   • {lang}: {count} files ({percentage:.1f}%)")
            lines.append("")

        # Largest files
        if context.largest_files:
            lines.append("📋 Largest Files:")
            for file_path, size in context.largest_files[:5]:  # Top 5 files
                size_str = RepositoryAnalyzer.format_size(size)
                lines.append(f"   • {file_path}: {size_str}")

        return "\n".join(lines)

    @staticmethod
    def to_json(context: RepositoryContext) -> str:
        """Convert repository context to JSON."""
        # Convert to dict for JSON serialization
        data = {
            'name': context.name,
            'path': context.path,
            'total_files': context.total_files,
            'total_directories': context.total_directories,
            'total_size': context.total_size,
            'total_lines': context.total_lines,
            'git_branch': context.git_branch,
            'git_remote_url': context.git_remote_url,
            'git_commits': context.git_commits,
            'languages': context.languages,
            'language_percentages': context.language_percentages,
            'largest_files': context.largest_files,
            'analysis_time': context.analysis_time,
            'files': [
                {
                    'path': f.path,
                    'size': f.size,
                    'line_count': f.line_count,
                    'language': f.language,
                    'mime_type': f.mime_type,
                    'is_binary': f.is_binary,
                    'modified_time': f.modified_time
                }
                for f in context.files
            ],
            'directories': [
                {
                    'path': d.path,
                    'file_count': d.file_count,
                    'subdirectory_count': d.subdirectory_count,
                    'total_size': d.total_size
                }
                for d in context.directories
            ]
        }
        return json.dumps(data, indent=2)


# LangChain tool integration
def analyze_repo_structure(repo_path: str = ".") -> str:
    """
    Analyze repository structure and return detailed context.

    Args:
        repo_path: Path to repository root (default: current directory)

    Returns:
        Human-readable repository analysis summary
    """
    try:
        analyzer = RepositoryAnalyzer(repo_path)
        context = analyzer.analyze_repository()
        return RepositoryContextFormatter.format_summary(context)
    except Exception as e:
        return f"❌ Error analyzing repository: {str(e)}"


# CLI interface
def main():
    """Command-line interface for repository analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze repository structure and context")
    parser.add_argument("path", nargs="?", default=".", help="Repository path (default: current directory)")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--format", "-f", choices=["summary", "json"], default="summary", help="Output format")

    args = parser.parse_args()

    try:
        analyzer = RepositoryAnalyzer(args.path)
        context = analyzer.analyze_repository()

        if args.format == "json" or args.output:
            output = RepositoryContextFormatter.to_json(context)
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output)
                print(f"✅ Analysis saved to {args.output}")
            else:
                print(output)
        else:
            print(RepositoryContextFormatter.format_summary(context))

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
