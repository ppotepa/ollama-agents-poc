"""Virtual repository classes for in-memory ZIP-based repository handling."""

import os
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Optional


class VirtualRepository:
    """In-memory representation of a repository extracted from ZIP."""

    def __init__(self, repo_url: str, zip_data: bytes):
        self.repo_url = repo_url
        self.zip_data = zip_data
        self.files: dict[str, bytes] = {}
        self.directories: dict[str, list[str]] = {}
        self.metadata: dict[str, Any] = {}
        self._extract_to_memory()

    def _extract_to_memory(self):
        """Extract ZIP contents to memory structures."""
        try:
            with zipfile.ZipFile(BytesIO(self.zip_data), 'r') as zip_ref:
                # Get all file and directory paths
                all_paths = zip_ref.namelist()

                # Process each path
                for path in all_paths:
                    if path.endswith('/'):
                        # It's a directory
                        dir_path = path.rstrip('/')
                        self.directories[dir_path] = []
                    else:
                        # It's a file
                        try:
                            file_content = zip_ref.read(path)
                            self.files[path] = file_content

                            # Add to parent directory listing
                            parent_dir = str(Path(path).parent)
                            if parent_dir == '.':
                                parent_dir = ''

                            if parent_dir not in self.directories:
                                self.directories[parent_dir] = []

                            filename = Path(path).name
                            if filename not in self.directories[parent_dir]:
                                self.directories[parent_dir].append(filename)

                        except Exception as e:
                            print(f"⚠️  Could not read file {path}: {e}")

                # Store metadata
                self.metadata = {
                    'total_files': len(self.files),
                    'total_directories': len(self.directories),
                    'repo_url': self.repo_url,
                    'zip_size': len(self.zip_data)
                }

        except Exception as e:
            print(f"❌ Error extracting ZIP to memory: {e}")
            raise

    def get_file_content(self, path: str) -> Optional[bytes]:
        """Get file content from memory."""
        return self.files.get(path)

    def get_file_content_text(self, path: str, encoding: str = 'utf-8') -> Optional[str]:
        """Get file content as text."""
        content = self.get_file_content(path)
        if content is None:
            return None
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            return None

    def list_directory(self, path: str = '') -> list[str]:
        """List contents of a directory."""
        return self.directories.get(path, [])

    def file_exists(self, path: str) -> bool:
        """Check if file exists in virtual repository."""
        return path in self.files

    def directory_exists(self, path: str) -> bool:
        """Check if directory exists in virtual repository."""
        return path in self.directories

    def get_all_files(self) -> list[str]:
        """Get list of all file paths."""
        return list(self.files.keys())

    def get_all_directories(self) -> list[str]:
        """Get list of all directory paths."""
        return list(self.directories.keys())


class VirtualDirectory:
    """In-memory representation of a ZIP-based repository for fast access."""

    def __init__(self, repo_url: str, zip_data: bytes):
        self.repo_url = repo_url
        self.zip_data = zip_data
        self.files = {}  # path -> file_info dict
        self.directories = set()
        self._build_virtual_structure()

    def _build_virtual_structure(self):
        """Build the virtual directory structure from ZIP data."""
        with zipfile.ZipFile(BytesIO(self.zip_data), 'r') as zip_ref:
            for info in zip_ref.infolist():
                # Skip directories and extract file info
                if not info.is_dir():
                    # Remove the top-level directory from path (e.g., "vscode-main/")
                    path_parts = info.filename.split('/', 1)
                    clean_path = path_parts[1] if len(path_parts) > 1 else info.filename

                    # Store file information
                    self.files[clean_path] = {
                        'size': info.file_size,
                        'compressed_size': info.compress_size,
                        'modified': info.date_time,
                        'zip_info': info
                    }

                    # Track directories
                    dir_path = os.path.dirname(clean_path)
                    while dir_path and dir_path != '.':
                        self.directories.add(dir_path)
                        dir_path = os.path.dirname(dir_path)

    def get_file_content(self, file_path: str) -> Optional[bytes]:
        """Get file content directly from ZIP."""
        if file_path not in self.files:
            return None

        try:
            with zipfile.ZipFile(BytesIO(self.zip_data), 'r') as zip_ref:
                zip_info = self.files[file_path]['zip_info']
                return zip_ref.read(zip_info)
        except Exception:
            return None

    def list_files(self, directory: str = "") -> list[str]:
        """List files in a directory."""
        if directory and not directory.endswith('/'):
            directory += '/'

        files = []
        for path in self.files:
            if directory == "" or path.startswith(directory):
                # Get relative path from directory
                relative_path = path[len(directory):] if directory else path
                # Only include direct children (no subdirectories)
                if '/' not in relative_path:
                    files.append(relative_path)
        return files

    def get_context_summary(self) -> dict[str, Any]:
        """Get a summary of the virtual directory for context."""
        total_files = len(self.files)
        total_dirs = len(self.directories)
        total_size = sum(info['size'] for info in self.files.values())

        # Get file extensions
        extensions = {}
        for path in self.files:
            ext = os.path.splitext(path)[1].lower()
            if ext:
                extensions[ext] = extensions.get(ext, 0) + 1

        # Get top directories by file count
        dir_counts = {}
        for path in self.files:
            dir_path = os.path.dirname(path)
            if dir_path:
                top_dir = dir_path.split('/')[0]
                dir_counts[top_dir] = dir_counts.get(top_dir, 0) + 1

        return {
            'repo_url': self.repo_url,
            'total_files': total_files,
            'total_directories': total_dirs,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'file_extensions': dict(sorted(extensions.items(), key=lambda x: x[1], reverse=True)[:10]),
            'top_directories': dict(sorted(dir_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        }
