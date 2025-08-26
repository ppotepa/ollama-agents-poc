"""Language detection service for file analysis.

Single Responsibility: Detect programming languages from file paths and content.
"""
from __future__ import annotations

from pathlib import Path


class LanguageDetector:
    """Detects programming languages from file extensions and names.

    Responsibility: Map file extensions and names to programming languages.
    """

    def __init__(self):
        self.extension_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.cxx': 'C++',
            '.cc': 'C++',
            '.c': 'C',
            '.cs': 'C#',
            '.go': 'Go',
            '.rs': 'Rust',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.kts': 'Kotlin',
            '.scala': 'Scala',
            '.r': 'R',
            '.R': 'R',
            '.m': 'Objective-C',
            '.mm': 'Objective-C++',
            '.sh': 'Shell',
            '.bash': 'Bash',
            '.zsh': 'Zsh',
            '.fish': 'Fish',
            '.ps1': 'PowerShell',
            '.psm1': 'PowerShell',
            '.psd1': 'PowerShell',
            '.sql': 'SQL',
            '.html': 'HTML',
            '.htm': 'HTML',
            '.css': 'CSS',
            '.scss': 'SCSS',
            '.sass': 'Sass',
            '.less': 'Less',
            '.vue': 'Vue',
            '.jsx': 'JSX',
            '.tsx': 'TSX',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.json': 'JSON',
            '.xml': 'XML',
            '.md': 'Markdown',
            '.markdown': 'Markdown',
            '.rst': 'reStructuredText',
            '.dockerfile': 'Dockerfile',
            '.makefile': 'Makefile',
            '.gradle': 'Gradle',
            '.bat': 'Batch',
            '.cmd': 'Batch',
            '.vim': 'Vim',
            '.lua': 'Lua',
            '.pl': 'Perl',
            '.pm': 'Perl',
            '.dart': 'Dart',
            '.elm': 'Elm',
            '.ex': 'Elixir',
            '.exs': 'Elixir',
            '.fs': 'F#',
            '.fsx': 'F#',
            '.hs': 'Haskell',
            '.lhs': 'Haskell',
            '.jl': 'Julia',
            '.nim': 'Nim',
            '.pas': 'Pascal',
            '.pp': 'Pascal',
            '.clj': 'Clojure',
            '.cljs': 'ClojureScript',
            '.edn': 'EDN',
            '.ml': 'OCaml',
            '.mli': 'OCaml',
            '.erl': 'Erlang',
            '.hrl': 'Erlang',
        }

        self.filename_map = {
            'dockerfile': 'Dockerfile',
            'makefile': 'Makefile',
            'rakefile': 'Ruby',
            'gemfile': 'Ruby',
            'podfile': 'Ruby',
            'vagrantfile': 'Ruby',
            'gruntfile.js': 'JavaScript',
            'gulpfile.js': 'JavaScript',
            'webpack.config.js': 'JavaScript',
            'package.json': 'JSON',
            'composer.json': 'JSON',
            'bower.json': 'JSON',
            'requirements.txt': 'Text',
            'setup.py': 'Python',
            'setup.cfg': 'INI',
            'pyproject.toml': 'TOML',
            'cargo.toml': 'TOML',
            '.gitignore': 'Text',
            '.gitattributes': 'Text',
            '.editorconfig': 'INI',
            'readme': 'Text',
            'readme.txt': 'Text',
            'license': 'Text',
            'changelog': 'Text',
            'changelog.md': 'Markdown',
        }

    def detect_language(self, file_path: Path) -> str | None:
        """Detect programming language from file path.

        Args:
            file_path: Path to the file

        Returns:
            Programming language name or None if not detected
        """
        # Check by filename first (for special files like Dockerfile)
        filename_lower = file_path.name.lower()
        if filename_lower in self.filename_map:
            return self.filename_map[filename_lower]

        # Check by file extension
        suffix = file_path.suffix.lower()
        if suffix in self.extension_map:
            return self.extension_map[suffix]

        return None

    def get_supported_languages(self) -> set[str]:
        """Get set of all supported programming languages."""
        languages = set(self.extension_map.values())
        languages.update(self.filename_map.values())
        return languages

    def get_extensions_for_language(self, language: str) -> list[str]:
        """Get file extensions associated with a programming language."""
        return [ext for ext, lang in self.extension_map.items() if lang == language]
