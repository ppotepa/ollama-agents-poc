"""Virtual repository implementation for in-memory ZIP processing."""

import hashlib
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
                        'crc': info.CRC,
                        'zip_info': info
                    }

                    # Track directory structure
                    path_obj = Path(clean_path)
                    for parent in path_obj.parents:
                        if str(parent) != '.':
                            self.directories.add(str(parent))

    def get_file_info(self, path: str) -> Optional[dict]:
        """Get file information."""
        return self.files.get(path)

    def list_files(self, directory: str = '') -> list[str]:
        """List files in a directory."""
        files = []
        for file_path in self.files:
            file_dir = str(Path(file_path).parent)
            if file_dir == directory or (directory == '' and file_dir == '.'):
                files.append(Path(file_path).name)
        return files

    def read_file(self, path: str, encoding: str = 'utf-8') -> Optional[str]:
        """Read file content as text."""
        if path not in self.files:
            return None

        try:
            with zipfile.ZipFile(BytesIO(self.zip_data), 'r') as zip_ref:
                # Need to find the original path with top-level directory
                for zip_path in zip_ref.namelist():
                    if zip_path.endswith('/' + path) or zip_path.endswith(path):
                        content = zip_ref.read(zip_path)
                        return content.decode(encoding)
            return None
        except Exception as e:
            print(f"Error reading file {path}: {e}")
            return None

    def file_exists(self, path: str) -> bool:
        """Check if file exists."""
        return path in self.files

    def get_repository_info(self) -> dict:
        """Get repository information."""
        return {
            'url': self.repo_url,
            'total_files': len(self.files),
            'total_directories': len(self.directories),
            'zip_size': len(self.zip_data)
        }
