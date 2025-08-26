"""Repository context system for comprehensive file and folder analysis.

This system builds detailed information about every file and folder in the repository,
stores file contents in memory for fast access, and helps agents generate targeted
patches/diffs for individual files.

Key Features:
- Complete repository mapping with file paths and metadata
- File content caching for performance
- Intelligent file categorization and filtering
- Diff-friendly structure for patch generation
- Virtual directory support for ZIP-based repositories
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path

# Global repository context cache
_repository_context_cache = None
_current_virtual_directory = None


def get_virtual_repository_context(repo_url: str) -> str | None:
    """Get repository context from virtual repository if available."""
    try:
        from collections import defaultdict

        from src.core.helpers import get_virtual_repository

        # Get virtual repository directly by URL
        virtual_repo = get_virtual_repository(repo_url)

        if not virtual_repo:
            return None

        # Build context from virtual repository
        files = virtual_repo.get_all_files()
        directories = virtual_repo.get_all_directories()

        # Analyze languages
        language_counts = defaultdict(int)
        language_map = {
            '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
            '.java': 'Java', '.cpp': 'C++', '.c': 'C', '.cs': 'C#',
            '.go': 'Go', '.rs': 'Rust', '.php': 'PHP', '.rb': 'Ruby',
            '.json': 'JSON', '.yml': 'YAML', '.yaml': 'YAML', '.xml': 'XML',
            '.html': 'HTML', '.css': 'CSS', '.scss': 'SCSS', '.less': 'LESS'
        }

        for file_path in files:
            ext = Path(file_path).suffix.lower()
            if ext in language_map:
                language = language_map[ext]
                language_counts[language] += 1

        # Build context summary
        context_lines = [
            f"📁 Virtual Repository: {repo_url}",
            f"📊 Total Files: {len(files)}",
            f"📂 Total Directories: {len(directories)}",
            f"💾 Total Size: {virtual_repo.metadata.get('zip_size', 0) / 1024 / 1024:.1f} MB (compressed)",
            "",
            "🔤 Languages:"
        ]

        for lang, count in sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            percentage = (count / len(files)) * 100 if files else 0
            context_lines.append(f"   {lang}: {count} files ({percentage:.1f}%)")

        if not language_counts:
            context_lines.append("   No recognized programming languages")

        context_lines.extend([
            "",
            "📁 Top Directories:"
        ])

        # Show top directories by file count
        dir_file_counts = {}
        for file_path in files:
            dir_path = str(Path(file_path).parent)
            if dir_path != '.' and dir_path != '':
                # Get top-level directory, skipping GitHub ZIP prefix
                path_parts = dir_path.split('/')
                if len(path_parts) > 1:
                    # Skip the first part (repo-branch folder from GitHub ZIP)
                    top_dir = path_parts[1] if len(path_parts) > 1 else path_parts[0]
                else:
                    top_dir = path_parts[0]
                if top_dir and not top_dir.startswith('.'):  # Skip hidden directories in summary
                    dir_file_counts[top_dir] = dir_file_counts.get(top_dir, 0) + 1

        for dir_name, file_count in sorted(dir_file_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            context_lines.append(f"   {dir_name}: {file_count} files")

        if not dir_file_counts:
            context_lines.append("   All files in root directory")

        return "\n".join(context_lines)

    except Exception as e:
        print(f"⚠️  Error getting virtual repository context: {e}")
        return None

try:
    from langchain.tools import StructuredTool

    from src.tools.registry import register_tool
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False
    from src.tools.registry import register_tool


@dataclass
class FileContext:
    """Complete context information for a single file."""
    path: str
    absolute_path: str
    relative_path: str
    size: int
    modified_time: float
    content_hash: str
    mime_type: str
    language: str
    is_text: bool
    is_binary: bool
    line_count: int
    content: str | None = None  # Cached content for text files
    encoding: str = "utf-8"

    @property
    def extension(self) -> str:
        return Path(self.path).suffix.lower()

    @property
    def filename(self) -> str:
        return Path(self.path).name

    @property
    def directory(self) -> str:
        return str(Path(self.path).parent)


@dataclass
class DirectoryContext:
    """Complete context information for a directory."""
    path: str
    absolute_path: str
    relative_path: str
    file_count: int
    subdirectory_count: int
    total_size: int
    files: list[FileContext] = field(default_factory=list)
    subdirectories: list[str] = field(default_factory=list)
    languages: dict[str, int] = field(default_factory=dict)


@dataclass
class RepositoryContextMap:
    """Complete repository context with all files and directories mapped."""
    root_path: str
    total_files: int
    total_directories: int
    total_size: int
    files: dict[str, FileContext] = field(default_factory=dict)  # path -> FileContext
    directories: dict[str, DirectoryContext] = field(default_factory=dict)  # path -> DirectoryContext
    languages: dict[str, int] = field(default_factory=dict)
    file_extensions: dict[str, int] = field(default_factory=dict)
    content_cache: dict[str, str] = field(default_factory=dict)  # path -> content
    last_updated: float = 0.0

    def get_file_by_name(self, filename: str) -> list[FileContext]:
        """Find files by filename (supports partial matches)."""
        matches = []
        for file_ctx in self.files.values():
            if filename.lower() in file_ctx.filename.lower():
                matches.append(file_ctx)
        return matches

    def get_files_by_extension(self, extension: str) -> list[FileContext]:
        """Get all files with specific extension."""
        if not extension.startswith('.'):
            extension = '.' + extension
        return [f for f in self.files.values() if f.extension == extension.lower()]

    def get_files_by_language(self, language: str) -> list[FileContext]:
        """Get all files of specific programming language."""
        return [f for f in self.files.values() if f.language.lower() == language.lower()]

    def get_directory_contents(self, dir_path: str) -> DirectoryContext | None:
        """Get complete directory contents."""
        return self.directories.get(dir_path)

    def search_content(self, search_term: str) -> list[FileContext]:
        """Search for term in cached file contents."""
        matches = []
        search_lower = search_term.lower()

        for path, content in self.content_cache.items():
            if search_lower in content.lower() and path in self.files:
                matches.append(self.files[path])

        return matches


# Virtual Directory Integration Functions
def set_virtual_directory_for_context(git_url: str):
    """Set the current virtual directory for context operations."""
    global _current_virtual_directory

    try:
        from src.core.helpers import get_virtual_directory
        _current_virtual_directory = get_virtual_directory(git_url)
        if _current_virtual_directory:
            print(f"🧠 Virtual directory activated for context: {git_url}")
            return True
    except ImportError:
        pass

    _current_virtual_directory = None
    return False


def get_virtual_repository_summary() -> str:
    """Get a summary of the current virtual repository for context inclusion."""
    global _current_virtual_directory

    if _current_virtual_directory is None:
        return "No virtual repository loaded."

    try:
        summary = _current_virtual_directory.get_context_summary()

        lines = [
            "🧠 Virtual Repository Context:",
            f"📂 Repository: {summary['repo_url']}",
            f"📊 Files: {summary['total_files']} ({summary['total_size_mb']} MB)",
            f"📁 Directories: {summary['total_directories']}",
            "",
            "🔤 File Types:"
        ]

        # Add top file extensions
        for ext, count in list(summary['file_extensions'].items())[:5]:
            lines.append(f"   {ext}: {count} files")

        lines.append("")
        lines.append("📁 Main Directories:")

        # Add top directories
        for dir_name, count in list(summary['top_directories'].items())[:8]:
            lines.append(f"   {dir_name}/: {count} files")

        return "\n".join(lines)

    except Exception as e:
        return f"Error getting virtual repository summary: {e}"


def get_virtual_file_content(file_path: str) -> str | None:
    """Get file content directly from virtual directory (ZIP)."""
    global _current_virtual_directory

    if _current_virtual_directory is None:
        return None

    try:
        content_bytes = _current_virtual_directory.get_file_content(file_path)
        if content_bytes is None:
            return None

        # Try to decode as text
        try:
            return content_bytes.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return content_bytes.decode('latin-1')
            except UnicodeDecodeError:
                return None  # Binary file
    except Exception:
        return None


class RepositoryContextBuilder:
    """Builds comprehensive repository context with content caching."""

    # Files to ignore for content caching
    IGNORE_PATTERNS = {
        '.git', '__pycache__', 'node_modules', '.venv', 'venv', 'env',
        '.pytest_cache', '.mypy_cache', 'dist', '.tox',
        'coverage', '.coverage', 'htmlcov', '.DS_Store', 'Thumbs.db',
        'models', 'ollama-agents', '.idea', 'tmp', 'temp',
        'cache', '.cache', 'logs', '.logs'  # Removed 'build', 'data', '.vscode'
    }

    # Binary file extensions to skip content caching
    BINARY_EXTENSIONS = {
        '.exe', '.dll', '.so', '.dylib', '.bin', '.pkg', '.dmg', '.msi',
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.svg',
        '.mp3', '.mp4', '.avi', '.mov', '.wav', '.ogg', '.flv',
        '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar', '.xz',
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.pyc', '.pyo', '.class', '.jar'
    }

    # Text file size limit for content caching (1MB)
    MAX_CONTENT_SIZE = 1024 * 1024

    def __init__(self, repository_path: str = "."):
        self.repository_path = Path(repository_path).resolve()

    def build_context_map(self, cache_content: bool = True) -> RepositoryContextMap:
        """Build complete repository context map."""
        print("🔍 Starting repository scan...")

        context_map = RepositoryContextMap(
            root_path=str(self.repository_path),
            total_files=0,
            total_directories=0,
            total_size=0,
            last_updated=os.path.getmtime(self.repository_path)
        )

        file_count = 0
        dir_count = 0

        try:
            self._scan_directory(self.repository_path, context_map, cache_content, file_count, dir_count)
            self._calculate_summaries(context_map)
        except KeyboardInterrupt:
            print(f"⚠️  Scan interrupted. Processed {file_count} files, {dir_count} directories")
            raise
        except Exception as e:
            print(f"❌ Error during scan: {e}")
            raise

        return context_map

    def _scan_directory(self, dir_path: Path, context_map: RepositoryContextMap, cache_content: bool, file_count: int = 0, dir_count: int = 0):
        """Recursively scan directory and build context."""
        try:
            if self._should_ignore_path(dir_path):
                return file_count, dir_count

            # Progress indicator every 50 files
            if file_count % 50 == 0 and file_count > 0:
                print(f"   📊 Processed {file_count} files, {dir_count} directories...")

            dir_context = DirectoryContext(
                path=str(dir_path.relative_to(self.repository_path)),
                absolute_path=str(dir_path),
                relative_path=str(dir_path.relative_to(self.repository_path)),
                file_count=0,
                subdirectory_count=0,
                total_size=0
            )

            # Limit to prevent infinite loops or excessive scanning
            if file_count > 100000:  # Safety limit
                print("⚠️  Reached file limit (10,000), stopping scan")
                return file_count, dir_count

            # Process directory contents
            try:
                items = list(dir_path.iterdir())
            except OSError as e:
                print(f"⚠️  Cannot read directory {dir_path}: {e}")
                return file_count, dir_count

            for item in items:
                if self._should_ignore_path(item):
                    continue

                if item.is_file():
                    file_context = self._build_file_context(item, cache_content)
                    if file_context:
                        context_map.files[file_context.relative_path] = file_context
                        dir_context.files.append(file_context)
                        dir_context.file_count += 1
                        dir_context.total_size += file_context.size
                        file_count += 1

                        # Update language counts
                        if file_context.language:
                            dir_context.languages[file_context.language] = dir_context.languages.get(file_context.language, 0) + 1

                elif item.is_dir():
                    dir_context.subdirectories.append(str(item.relative_to(self.repository_path)))
                    dir_context.subdirectory_count += 1
                    dir_count += 1
                    # Recursively scan subdirectory
                    file_count, dir_count = self._scan_directory(item, context_map, cache_content, file_count, dir_count)

            context_map.directories[dir_context.relative_path] = dir_context
            context_map.total_directories += 1

            return file_count, dir_count

        except PermissionError:
            print(f"⚠️  Permission denied: {dir_path}")
            return file_count, dir_count
        except Exception as e:
            print(f"⚠️  Error scanning {dir_path}: {e}")
            return file_count, dir_count

    def _build_file_context(self, file_path: Path, cache_content: bool) -> FileContext | None:
        """Build complete context for a single file."""
        try:
            stat = file_path.stat()
            relative_path = str(file_path.relative_to(self.repository_path))

            # Determine file type and language
            mime_type, _ = mimetypes.guess_type(str(file_path))
            language = self._detect_language(file_path)
            extension = file_path.suffix.lower()

            is_binary = extension in self.BINARY_EXTENSIONS or self._is_binary_file(file_path)
            is_text = not is_binary

            # Calculate content hash
            content_hash = self._calculate_file_hash(file_path)

            file_context = FileContext(
                path=str(file_path),
                absolute_path=str(file_path.resolve()),
                relative_path=relative_path,
                size=stat.st_size,
                modified_time=stat.st_mtime,
                content_hash=content_hash,
                mime_type=mime_type or "application/octet-stream",
                language=language,
                is_text=is_text,
                is_binary=is_binary,
                line_count=0
            )

            # Cache content for text files if requested
            if cache_content and is_text and stat.st_size <= self.MAX_CONTENT_SIZE:
                content = self._read_file_content(file_path)
                if content is not None:
                    file_context.content = content
                    file_context.line_count = content.count('\n') + 1

            return file_context

        except Exception as e:
            print(f"⚠️  Error processing file {file_path}: {e}")
            return None

    def _should_ignore_path(self, path: Path) -> bool:
        """Check if path should be ignored."""
        name = path.name

        # Check ignore patterns
        for pattern in self.IGNORE_PATTERNS:
            if pattern in str(path) or name == pattern:
                return True

        # Skip specific hidden files/directories that are typically not needed
        # but allow important project configuration files
        if name.startswith('.'):
            # Allow these important dotfiles/directories
            allowed_dotfiles = {
                '.gitignore', '.env', '.editorconfig', '.github', '.vscode',
                '.config', '.devcontainer', '.eslint-plugin-local',
                '.gitattributes', '.npmrc', '.nvmrc', '.eslint-ignore',
                '.git-blame-ignore-revs', '.lsifrc.json', '.mailmap',
                '.mention-bot', '.vscode-test.js'
            }

            # Also allow dotfiles with important extensions
            allowed_extensions = {'.json', '.js', '.ts', '.yml', '.yaml', '.md', '.txt'}

            if name in allowed_dotfiles or path.suffix.lower() in allowed_extensions:
                return False  # Don't ignore these

            # Ignore other hidden files/directories
            return True

        return False

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        extension = file_path.suffix.lower()

        language_map = {
            '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
            '.java': 'Java', '.cpp': 'C++', '.c': 'C', '.h': 'C/C++',
            '.cs': 'C#', '.php': 'PHP', '.rb': 'Ruby', '.go': 'Go',
            '.rs': 'Rust', '.swift': 'Swift', '.kt': 'Kotlin',
            '.scala': 'Scala', '.clj': 'Clojure', '.hs': 'Haskell',
            '.ml': 'OCaml', '.fs': 'F#', '.vb': 'Visual Basic',
            '.html': 'HTML', '.css': 'CSS', '.scss': 'SCSS', '.sass': 'Sass',
            '.xml': 'XML', '.json': 'JSON', '.yaml': 'YAML', '.yml': 'YAML',
            '.md': 'Markdown', '.rst': 'reStructuredText', '.txt': 'Text',
            '.sh': 'Shell', '.bash': 'Bash', '.zsh': 'Zsh', '.fish': 'Fish',
            '.ps1': 'PowerShell', '.bat': 'Batch', '.cmd': 'Batch',
            '.sql': 'SQL', '.r': 'R', '.m': 'MATLAB', '.jl': 'Julia',
            '.dockerfile': 'Dockerfile', '.docker': 'Dockerfile'
        }

        # Special cases
        if file_path.name.lower() in {'dockerfile', 'makefile', 'rakefile'}:
            return file_path.name.title()

        return language_map.get(extension, 'Unknown')

    def _is_binary_file(self, file_path: Path) -> bool:
        """Check if file is binary by reading first chunk."""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                return b'\x00' in chunk
        except (OSError, PermissionError):
            return True

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file content."""
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()[:16]  # Short hash for memory efficiency
        except (OSError, PermissionError):
            return "unknown"

    def _read_file_content(self, file_path: Path) -> str | None:
        """Read file content with encoding detection."""
        encodings = ['utf-8', 'latin-1', 'ascii', 'cp1252']

        for encoding in encodings:
            try:
                with open(file_path, encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception:
                break

        return None

    def _calculate_summaries(self, context_map: RepositoryContextMap):
        """Calculate summary statistics."""
        context_map.total_files = len(context_map.files)
        context_map.total_size = sum(f.size for f in context_map.files.values())

        # Language summary
        for file_ctx in context_map.files.values():
            if file_ctx.language:
                context_map.languages[file_ctx.language] = context_map.languages.get(file_ctx.language, 0) + 1

        # Extension summary
        for file_ctx in context_map.files.values():
            ext = file_ctx.extension
            context_map.file_extensions[ext] = context_map.file_extensions.get(ext, 0) + 1

        # Cache content for quick access
        for file_ctx in context_map.files.values():
            if file_ctx.content:
                context_map.content_cache[file_ctx.relative_path] = file_ctx.content


# Global context cache
_repository_context_cache: RepositoryContextMap | None = None


def build_repository_context(repository_path: str = ".", force_rebuild: bool = False, cache_content: bool = False) -> str:
    """Build comprehensive repository context with file mapping and optional content caching."""
    global _repository_context_cache

    try:
        # Check if we need to rebuild
        if force_rebuild or _repository_context_cache is None:
            print(f"🔍 Building repository context for: {repository_path}")
            print(f"📁 Content caching: {'enabled' if cache_content else 'disabled (faster)'}")

            builder = RepositoryContextBuilder(repository_path)
            _repository_context_cache = builder.build_context_map(cache_content=cache_content)

            print(f"✅ Repository context built: {_repository_context_cache.total_files} files, {_repository_context_cache.total_directories} directories")

        # Return formatted summary
        ctx = _repository_context_cache
        lines = [
            f"📁 Repository Context: {ctx.root_path}",
            f"📊 Total Files: {ctx.total_files}",
            f"📂 Total Directories: {ctx.total_directories}",
            f"💾 Total Size: {ctx.total_size / (1024*1024):.1f} MB",
            "",
            "🔤 Top Languages:"
        ]

        # Show top 5 languages
        sorted_langs = sorted(ctx.languages.items(), key=lambda x: x[1], reverse=True)[:5]
        for lang, count in sorted_langs:
            percentage = (count / ctx.total_files) * 100 if ctx.total_files > 0 else 0
            lines.append(f"   • {lang}: {count} files ({percentage:.1f}%)")

        lines.extend([
            "",
            "📁 Top Directories:"
        ])

        # Show directories with most files
        sorted_dirs = sorted(ctx.directories.items(), key=lambda x: x[1].file_count, reverse=True)[:5]
        for dir_path, dir_ctx in sorted_dirs:
            lines.append(f"   • {dir_path or 'root'}: {dir_ctx.file_count} files")

        return "\n".join(lines)

    except Exception as e:
        return f"❌ Error building repository context: {e}"


def load_repository_context_after_clone(repository_path: str, cache_content: bool = True, quiet: bool = False) -> bool:
    """Load repository context into memory after a repository has been cloned.

    This function should be called immediately after a git repository is cloned
    to ensure the repository structure is loaded into memory for fast access.

    Args:
        repository_path: Path to the cloned repository
        cache_content: Whether to cache file contents for fast access
        quiet: If True, suppress output messages

    Returns:
        True if context was loaded successfully, False otherwise
    """
    global _repository_context_cache

    try:
        if not quiet:
            print("📂 Loading repository context into memory...")

        builder = RepositoryContextBuilder(repository_path)
        _repository_context_cache = builder.build_context_map(cache_content=cache_content)

        if not quiet:
            print(f"✅ Repository context loaded: {_repository_context_cache.total_files} files, "
                  f"{_repository_context_cache.total_directories} directories")

            # Show a quick summary of what was loaded
            ctx = _repository_context_cache
            if ctx.languages:
                top_lang = max(ctx.languages.items(), key=lambda x: x[1])
                print(f"📝 Primary language: {top_lang[0]} ({top_lang[1]} files)")

            if cache_content:
                cached_files = len([f for f in ctx.files.values() if f.content])
                print(f"💾 Content cached for {cached_files} text files")

        return True

    except Exception as e:
        if not quiet:
            print(f"❌ Error loading repository context: {e}")
        return False


def get_file_context(file_path: str) -> str:
    """Get detailed context for a specific file including content if cached."""
    global _repository_context_cache

    if _repository_context_cache is None:
        build_repository_context()

    file_ctx = _repository_context_cache.files.get(file_path)
    if not file_ctx:
        return f"❌ File not found: {file_path}"

    lines = [
        f"📄 File: {file_ctx.filename}",
        f"📍 Path: {file_ctx.relative_path}",
        f"💻 Language: {file_ctx.language}",
        f"📏 Size: {file_ctx.size} bytes",
        f"📝 Lines: {file_ctx.line_count}",
        f"🔗 Type: {'Text' if file_ctx.is_text else 'Binary'}",
        f"🔑 Hash: {file_ctx.content_hash}",
    ]

    if file_ctx.content:
        lines.extend([
            "",
            "📜 Content Preview (first 10 lines):",
            "```" + file_ctx.language.lower(),
            "\n".join(file_ctx.content.split('\n')[:10]),
            "```"
        ])

    return "\n".join(lines)


def search_files(query: str, search_type: str = "name") -> str:
    """Search files by name, extension, language, or content."""
    global _repository_context_cache

    if _repository_context_cache is None:
        build_repository_context()

    ctx = _repository_context_cache
    matches = []

    if search_type == "name":
        matches = ctx.get_file_by_name(query)
    elif search_type == "extension":
        matches = ctx.get_files_by_extension(query)
    elif search_type == "language":
        matches = ctx.get_files_by_language(query)
    elif search_type == "content":
        matches = ctx.search_content(query)
    else:
        return f"❌ Invalid search type: {search_type}. Use: name, extension, language, content"

    if not matches:
        return f"❌ No files found for query: '{query}' (type: {search_type})"

    lines = [f"🔍 Found {len(matches)} files for '{query}' ({search_type} search):"]

    for file_ctx in matches[:20]:  # Limit to first 20 results
        size_str = f"{file_ctx.size} bytes" if file_ctx.size < 1024 else f"{file_ctx.size/1024:.1f} KB"
        lines.append(f"   • {file_ctx.relative_path} ({file_ctx.language}, {size_str})")

    if len(matches) > 20:
        lines.append(f"   ... and {len(matches) - 20} more files")

    return "\n".join(lines)


def get_file_content(file_path: str) -> str:
    """Get full content of a file from cache or disk."""
    global _repository_context_cache

    if _repository_context_cache is None:
        build_repository_context()

    # Try cache first
    content = _repository_context_cache.content_cache.get(file_path)
    if content:
        return content

    # Try to read from disk
    file_ctx = _repository_context_cache.files.get(file_path)
    if not file_ctx:
        return f"❌ File not found: {file_path}"

    if file_ctx.is_binary:
        return f"❌ Cannot read binary file: {file_path}"

    try:
        builder = RepositoryContextBuilder()
        content = builder._read_file_content(Path(file_ctx.absolute_path))
        if content:
            # Cache it for future use
            _repository_context_cache.content_cache[file_path] = content
            return content
        else:
            return f"❌ Could not read file: {file_path}"
    except Exception as e:
        return f"❌ Error reading file {file_path}: {e}"


def analyze_repository_context(repository_path: str = ".") -> str:
    """Analyze repository structure and provide comprehensive context including file counts, languages, sizes, and structure."""
    return build_repository_context(repository_path)


def analyze_repo_languages(repository_path: str = ".") -> str:
    """Analyze programming languages used in repository with detailed breakdown and percentages."""
    global _repository_context_cache

    if _repository_context_cache is None:
        build_repository_context(repository_path)

    ctx = _repository_context_cache

    if not ctx.languages:
        return "No programming languages detected in repository."

    lines = ["💻 Programming Languages:"]

    # Sort by count, descending
    sorted_langs = sorted(ctx.languages.items(), key=lambda x: x[1], reverse=True)

    for lang, count in sorted_langs:
        percentage = (count / ctx.total_files) * 100
        lines.append(f"   • {lang}: {count} files ({percentage:.1f}%)")

    return "\n".join(lines)


def analyze_repo_directories(repository_path: str = ".", max_depth: int = 3) -> str:
    """Analyze repository directory structure with file counts and sizes."""
    global _repository_context_cache

    if _repository_context_cache is None:
        build_repository_context(repository_path)

    ctx = _repository_context_cache

    if not ctx.directories:
        return "No directories found in repository."

    lines = ["📁 Directory Structure:"]

    # Sort directories by depth and name
    sorted_dirs = sorted(ctx.directories.items(), key=lambda x: (x[0].count('/'), x[0]))

    for dir_path, dir_ctx in sorted_dirs:
        depth = dir_path.count('/')
        if depth <= max_depth:
            indent = "   " * (depth + 1)
            dir_name = dir_path.split('/')[-1] if dir_path else 'root'
            lines.append(f"{indent}📂 {dir_name}/")
            lines.append(f"{indent}   • {dir_ctx.file_count} files, {dir_ctx.subdirectory_count} subdirs")

    return "\n".join(lines)


def get_repository_state() -> str:
    """Get current repository state as JSON for agent context management."""
    global _repository_context_cache

    if _repository_context_cache is None:
        # Don't rebuild from current directory - return empty state
        return '{"error": "No repository context loaded. Use build_repository_context() or load_repository_context_after_clone() first."}'

    try:
        ctx = _repository_context_cache
        state = {
            "root_path": ctx.root_path,
            "total_files": ctx.total_files,
            "total_directories": ctx.total_directories,
            "total_size": ctx.total_size,
            "languages": ctx.languages,
            "file_extensions": ctx.file_extensions,
            "top_directories": [
                {"path": path, "file_count": dir_ctx.file_count}
                for path, dir_ctx in sorted(ctx.directories.items(), key=lambda x: x[1].file_count, reverse=True)[:10]
            ],
            "last_updated": ctx.last_updated
        }
        return json.dumps(state, indent=2)
    except Exception as e:
        return f"❌ Error getting repository state: {str(e)}"


# Register tools with the system
if LANGCHAIN_AVAILABLE:
    for fn in [build_repository_context, get_file_context, search_files, get_file_content,
               analyze_repository_context, analyze_repo_languages, analyze_repo_directories, get_repository_state]:
        register_tool(StructuredTool.from_function(fn, name=fn.__name__, description=fn.__doc__ or fn.__name__))
else:
    # Use the standard ToolWrapper from registry
    from src.tools.registry import ToolWrapper
    for _f in [build_repository_context, get_file_context, search_files, get_file_content,
               analyze_repository_context, analyze_repo_languages, analyze_repo_directories, get_repository_state]:
        register_tool(ToolWrapper(_f))


__all__ = [
    "build_repository_context",
    "get_file_context",
    "search_files",
    "get_file_content",
    "analyze_repository_context",
    "analyze_repo_languages",
    "analyze_repo_directories",
    "get_repository_state",
    "RepositoryContextMap",
    "FileContext",
    "DirectoryContext",
    "RepositoryContextBuilder"
]
