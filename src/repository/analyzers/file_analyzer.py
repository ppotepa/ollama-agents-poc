"""File analysis service for repository scanning.

Single Responsibility: Analyze individual files and extract metadata.
"""
from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Optional

from ..models.repository_context import FileInfo, AnalysisConfig
from .language_detector import LanguageDetector


class FileAnalyzer:
    """Analyzes individual files and extracts metadata.
    
    Responsibility: Extract file information including size, lines, language, etc.
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.language_detector = LanguageDetector()
    
    def should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored based on configuration.
        
        Args:
            path: File or directory path to check
            
        Returns:
            True if path should be ignored
        """
        # Check each part of the path
        for part in path.parts:
            if part in self.config.ignore_patterns:
                return True
        
        # Check glob patterns
        for pattern in self.config.ignore_patterns:
            if '*' in pattern and path.match(pattern):
                return True
                
        return False
    
    def is_binary_file(self, file_path: Path) -> bool:
        """Check if file is binary by looking for null bytes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file appears to be binary
        """
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(8192)  # Read first 8KB
                return b'\0' in chunk
        except Exception:
            return True  # Assume binary if can't read
    
    def count_lines(self, file_path: Path) -> int:
        """Count lines in a text file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Number of lines in the file
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return sum(1 for _ in f)
        except Exception:
            return 0
    
    def analyze_file(self, file_path: Path, root_path: Path) -> Optional[FileInfo]:
        """Analyze a single file and extract metadata.
        
        Args:
            file_path: Path to the file to analyze
            root_path: Repository root path for relative path calculation
            
        Returns:
            FileInfo object or None if file should be skipped
        """
        try:
            # Check if file should be ignored
            if self.should_ignore(file_path):
                return None
                
            # Get file stats
            stat = file_path.stat()
            
            # Check file size limit
            if self.config.max_file_size_mb:
                max_size_bytes = self.config.max_file_size_mb * 1024 * 1024
                if stat.st_size > max_size_bytes:
                    return None
            
            # Detect if binary
            is_binary = self.is_binary_file(file_path)
            
            # Skip binary files if configured
            if not self.config.include_binary_files and is_binary:
                return None
            
            # Count lines (only for text files)
            lines = 0 if is_binary else self.count_lines(file_path)
            
            # Get MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            # Detect programming language
            language = self.language_detector.detect_language(file_path)
            
            return FileInfo(
                path=str(file_path.relative_to(root_path)),
                size=stat.st_size,
                lines=lines,
                language=language,
                mime_type=mime_type,
                is_binary=is_binary,
                last_modified=stat.st_mtime
            )
            
        except Exception as e:
            # Return minimal info for problematic files
            try:
                return FileInfo(
                    path=str(file_path.relative_to(root_path)),
                    size=0,
                    lines=0,
                    language=None,
                    mime_type=None,
                    is_binary=True,
                    last_modified=0
                )
            except Exception:
                return None  # Can't even get relative path
    
    def get_file_statistics(self, files: list[FileInfo]) -> dict[str, any]:
        """Calculate statistics from a list of file info objects.
        
        Args:
            files: List of FileInfo objects
            
        Returns:
            Dictionary with file statistics
        """
        total_size = sum(f.size for f in files)
        total_lines = sum(f.lines for f in files)
        
        # Language distribution
        languages = {}
        for file_info in files:
            if file_info.language:
                languages[file_info.language] = languages.get(file_info.language, 0) + 1
        
        # File type distribution
        file_types = {}
        for file_info in files:
            ext = Path(file_info.path).suffix.lower()
            if ext:
                file_types[ext] = file_types.get(ext, 0) + 1
        
        # Largest files
        largest_files = sorted(files, key=lambda f: f.size, reverse=True)[:self.config.max_largest_files]
        
        return {
            'total_size': total_size,
            'total_lines': total_lines,
            'languages': languages,
            'file_types': file_types,
            'largest_files': largest_files,
        }
