"""Text formatting service for repository analysis output.

Single Responsibility: Format repository analysis results into human-readable text.
"""
from __future__ import annotations

from pathlib import Path
from ..models.repository_context import RepositoryContext, FileInfo


class TextFormatter:
    """Formats repository analysis results as human-readable text.
    
    Responsibility: Convert analysis data into formatted text output.
    """
    
    @staticmethod
    def format_size(size_bytes: int) -> str:
        """Format file size in human-readable format.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Human-readable size string
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    @staticmethod
    def format_number(number: int) -> str:
        """Format large numbers with thousand separators.
        
        Args:
            number: Number to format
            
        Returns:
            Formatted number string
        """
        return f"{number:,}"
    
    def format_repository_summary(self, context: RepositoryContext) -> str:
        """Generate a comprehensive repository summary.
        
        Args:
            context: Repository analysis results
            
        Returns:
            Formatted summary text
        """
        lines = []
        
        # Header
        repo_name = Path(context.root_path).name
        lines.append(f"📁 Repository: {repo_name}")
        lines.append("=" * 50)
        
        # Overview section
        lines.append(f"📊 Overview:")
        lines.append(f"   • {self.format_number(context.total_files)} files")
        lines.append(f"   • {self.format_number(context.total_directories)} directories")
        lines.append(f"   • {self.format_size(context.total_size)} total size")
        lines.append(f"   • {self.format_number(context.total_lines)} lines of code")
        lines.append("")
        
        # Git information
        if context.git_info:
            lines.append(f"🔧 Git Information:")
            if 'branch' in context.git_info:
                lines.append(f"   • Branch: {context.git_info['branch']}")
            if 'commit_count' in context.git_info:
                lines.append(f"   • Commits: {self.format_number(context.git_info['commit_count'])}")
            if 'last_commit_hash' in context.git_info:
                lines.append(f"   • Last Commit: {context.git_info['last_commit_hash']}")
            if 'last_commit_author' in context.git_info:
                lines.append(f"   • Last Author: {context.git_info['last_commit_author']}")
            if 'has_uncommitted_changes' in context.git_info:
                status = "dirty" if context.git_info['has_uncommitted_changes'] else "clean"
                lines.append(f"   • Status: {status}")
            lines.append("")
        
        # Programming languages
        if context.languages:
            lines.append(f"💻 Programming Languages:")
            total_files = context.total_files
            for lang, count in sorted(context.languages.items(), key=lambda x: x[1], reverse=True)[:10]:
                percentage = (count / total_files) * 100 if total_files > 0 else 0
                lines.append(f"   • {lang}: {count} files ({percentage:.1f}%)")
            lines.append("")
        
        # File types
        if context.file_types:
            lines.append(f"📄 File Types:")
            for ext, count in sorted(context.file_types.items(), key=lambda x: x[1], reverse=True)[:10]:
                percentage = (count / context.total_files) * 100 if context.total_files > 0 else 0
                lines.append(f"   • {ext}: {count} files ({percentage:.1f}%)")
            lines.append("")
        
        # Largest files
        if context.largest_files:
            lines.append(f"📋 Largest Files:")
            for file_info in context.largest_files[:10]:
                lines.append(f"   • {file_info.path}: {self.format_size(file_info.size)}")
            lines.append("")
        
        return "\n".join(lines)
    
    def format_file_info(self, file_info: FileInfo) -> str:
        """Format detailed information about a single file.
        
        Args:
            file_info: File information to format
            
        Returns:
            Formatted file information
        """
        lines = []
        
        lines.append(f"📄 File: {file_info.path}")
        lines.append(f"   • Size: {self.format_size(file_info.size)}")
        lines.append(f"   • Lines: {self.format_number(file_info.lines)}")
        lines.append(f"   • Language: {file_info.language or 'Unknown'}")
        lines.append(f"   • MIME Type: {file_info.mime_type or 'Unknown'}")
        lines.append(f"   • Binary: {'Yes' if file_info.is_binary else 'No'}")
        
        return "\n".join(lines)
    
    def format_language_breakdown(self, context: RepositoryContext) -> str:
        """Format detailed language breakdown.
        
        Args:
            context: Repository analysis results
            
        Returns:
            Formatted language breakdown
        """
        if not context.languages:
            return "No programming languages detected."
        
        lines = []
        lines.append("💻 Language Breakdown:")
        lines.append("-" * 30)
        
        total_files = sum(context.languages.values())
        
        for lang, count in sorted(context.languages.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_files) * 100
            bar_length = int(percentage / 2)  # Scale to 50 chars max
            bar = "█" * bar_length + "░" * (50 - bar_length)
            
            lines.append(f"{lang:15} │{bar}│ {count:4} files ({percentage:5.1f}%)")
        
        return "\n".join(lines)
    
    def format_directory_structure(self, context: RepositoryContext, max_depth: int = 3) -> str:
        """Format directory structure overview.
        
        Args:
            context: Repository analysis results
            max_depth: Maximum depth to show
            
        Returns:
            Formatted directory structure
        """
        lines = []
        lines.append("📁 Directory Structure:")
        lines.append("-" * 25)
        
        # Sort directories by path
        sorted_dirs = sorted(context.directory_structure, key=lambda d: d.path)
        
        for dir_info in sorted_dirs:
            depth = len(Path(dir_info.path).parts)
            if depth > max_depth:
                continue
                
            indent = "  " * (depth - 1)
            size_info = self.format_size(dir_info.total_size)
            file_info = f"{dir_info.file_count} files"
            
            lines.append(f"{indent}📁 {Path(dir_info.path).name} ({file_info}, {size_info})")
        
        return "\n".join(lines)
    
    def format_compact_summary(self, context: RepositoryContext) -> str:
        """Generate a compact one-line summary.
        
        Args:
            context: Repository analysis results
            
        Returns:
            Compact summary string
        """
        repo_name = Path(context.root_path).name
        file_count = self.format_number(context.total_files)
        size = self.format_size(context.total_size)
        
        main_lang = "Unknown"
        if context.languages:
            main_lang = max(context.languages.items(), key=lambda x: x[1])[0]
        
        return f"📁 {repo_name}: {file_count} files, {size}, primarily {main_lang}"
