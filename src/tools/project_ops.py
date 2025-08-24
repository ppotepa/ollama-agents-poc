"""Project tools."""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

try:
    from langchain.tools import StructuredTool
    from src.tools.registry import register_tool
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False
    from src.tools.registry import register_tool


def create_project(project_name: str, project_type: str = "python") -> str:
    if os.path.exists(project_name):
        return f"❌ Project directory {project_name} already exists"
    os.makedirs(project_name)
    if project_type.lower() == "python":
        with open(os.path.join(project_name, "main.py"), "w", encoding="utf-8") as f:
            f.write("print('Hello')\n")
    return f"🚀 Created {project_type} project: {project_name}"


def analyze_repository(path: str = ".", max_files: int = 100, include_content: bool = False) -> str:
    """Analyze repository structure, file sizes, and optionally content."""
    try:
        repo_path = Path(path).resolve()
        if not repo_path.exists():
            return f"❌ Path {path} does not exist"
        
        # Common directories to ignore
        ignore_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.env', 
                      'build', 'dist', '.pytest_cache', '.mypy_cache', 'models'}
        
        # Common file extensions to categorize
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', '.go', '.rs'}
        config_extensions = {'.yaml', '.yml', '.json', '.toml', '.ini', '.cfg', '.conf'}
        doc_extensions = {'.md', '.txt', '.rst', '.doc', '.docx', '.pdf'}
        
        analysis = {
            'path': str(repo_path),
            'total_files': 0,
            'total_size': 0,
            'file_types': defaultdict(int),
            'largest_files': [],
            'directory_structure': {},
            'code_files': [],
            'config_files': [],
            'doc_files': [],
            'language_stats': defaultdict(int)
        }
        
        files_processed = 0
        
        for root, dirs, files in os.walk(repo_path):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            rel_root = Path(root).relative_to(repo_path)
            
            for file in files:
                if files_processed >= max_files:
                    break
                    
                file_path = Path(root) / file
                try:
                    stat = file_path.stat()
                    file_size = stat.st_size
                    
                    analysis['total_files'] += 1
                    analysis['total_size'] += file_size
                    files_processed += 1
                    
                    # File extension analysis
                    ext = file_path.suffix.lower()
                    analysis['file_types'][ext or 'no_extension'] += 1
                    
                    # Categorize files
                    rel_path = str(file_path.relative_to(repo_path))
                    file_info = {
                        'path': rel_path,
                        'size': file_size,
                        'size_human': _format_size(file_size)
                    }
                    
                    if ext in code_extensions:
                        analysis['code_files'].append(file_info)
                        analysis['language_stats'][ext] += 1
                    elif ext in config_extensions:
                        analysis['config_files'].append(file_info)
                    elif ext in doc_extensions:
                        analysis['doc_files'].append(file_info)
                    
                    # Track largest files
                    analysis['largest_files'].append(file_info)
                    
                except (OSError, PermissionError):
                    continue
            
            if files_processed >= max_files:
                break
        
        # Sort and limit largest files
        analysis['largest_files'].sort(key=lambda x: x['size'], reverse=True)
        analysis['largest_files'] = analysis['largest_files'][:10]
        
        # Sort code files by size
        analysis['code_files'].sort(key=lambda x: x['size'], reverse=True)
        analysis['config_files'].sort(key=lambda x: x['size'], reverse=True)
        analysis['doc_files'].sort(key=lambda x: x['size'], reverse=True)
        
        # Generate summary
        return _format_repository_analysis(analysis, include_content, repo_path)
        
    except Exception as e:
        return f"❌ Error analyzing repository: {e}"


def get_file_content_summary(file_path: str, max_lines: int = 50) -> str:
    """Get a summary of file content with line count and preview."""
    try:
        path = Path(file_path)
        if not path.exists():
            return f"❌ File {file_path} does not exist"
        
        if not path.is_file():
            return f"❌ {file_path} is not a file"
        
        # Check if it's a text file by trying to read it
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(path, 'r', encoding='latin-1') as f:
                    lines = f.readlines()
            except:
                return f"📄 {file_path}\n   Binary file ({_format_size(path.stat().st_size)})"
        
        total_lines = len(lines)
        file_size = path.stat().st_size
        
        preview_lines = min(max_lines, total_lines)
        preview = ''.join(lines[:preview_lines])
        
        summary = f"📄 {file_path}\n"
        summary += f"   📏 {total_lines} lines, {_format_size(file_size)}\n"
        
        if preview_lines < total_lines:
            summary += f"   👀 Preview (first {preview_lines} lines):\n"
        else:
            summary += f"   📝 Content:\n"
        
        # Add line numbers to preview
        preview_with_numbers = ""
        for i, line in enumerate(lines[:preview_lines], 1):
            preview_with_numbers += f"   {i:3d}: {line}"
        
        summary += preview_with_numbers
        
        if preview_lines < total_lines:
            summary += f"   ... ({total_lines - preview_lines} more lines)\n"
        
        return summary
        
    except Exception as e:
        return f"❌ Error reading {file_path}: {e}"


def _format_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def _format_repository_analysis(analysis: Dict[str, Any], include_content: bool, repo_path: Path) -> str:
    """Format the repository analysis into a readable report."""
    report = f"📊 Repository Analysis: {analysis['path']}\n"
    report += "=" * 60 + "\n\n"
    
    # Summary statistics
    report += f"📈 Summary:\n"
    report += f"   📁 Total files: {analysis['total_files']}\n"
    report += f"   💾 Total size: {_format_size(analysis['total_size'])}\n"
    report += f"   💻 Code files: {len(analysis['code_files'])}\n"
    report += f"   ⚙️  Config files: {len(analysis['config_files'])}\n"
    report += f"   📚 Doc files: {len(analysis['doc_files'])}\n\n"
    
    # Language statistics
    if analysis['language_stats']:
        report += "🌐 Languages detected:\n"
        for ext, count in sorted(analysis['language_stats'].items(), key=lambda x: x[1], reverse=True):
            lang_name = _ext_to_language(ext)
            report += f"   {ext}: {count} files ({lang_name})\n"
        report += "\n"
    
    # Largest files
    if analysis['largest_files']:
        report += "📋 Largest files:\n"
        for file_info in analysis['largest_files'][:5]:
            report += f"   📄 {file_info['path']} ({file_info['size_human']})\n"
        report += "\n"
    
    # Key code files
    if analysis['code_files']:
        report += "💻 Key code files:\n"
        for file_info in analysis['code_files'][:10]:
            report += f"   🐍 {file_info['path']} ({file_info['size_human']})\n"
        report += "\n"
    
    # Configuration files
    if analysis['config_files']:
        report += "⚙️  Configuration files:\n"
        for file_info in analysis['config_files'][:5]:
            report += f"   ⚙️  {file_info['path']} ({file_info['size_human']})\n"
        report += "\n"
    
    # File type distribution
    if analysis['file_types']:
        report += "📊 File type distribution:\n"
        sorted_types = sorted(analysis['file_types'].items(), key=lambda x: x[1], reverse=True)
        for ext, count in sorted_types[:10]:
            report += f"   {ext}: {count} files\n"
        report += "\n"
    
    return report


def _ext_to_language(ext: str) -> str:
    """Convert file extension to language name."""
    lang_map = {
        '.py': 'Python',
        '.js': 'JavaScript', 
        '.ts': 'TypeScript',
        '.java': 'Java',
        '.cpp': 'C++',
        '.c': 'C',
        '.cs': 'C#',
        '.go': 'Go',
        '.rs': 'Rust',
        '.php': 'PHP',
        '.rb': 'Ruby',
        '.yaml': 'YAML',
        '.yml': 'YAML',
        '.json': 'JSON',
        '.toml': 'TOML',
        '.md': 'Markdown'
    }
    return lang_map.get(ext, 'Unknown')


if LANGCHAIN_AVAILABLE:
    register_tool(StructuredTool.from_function(create_project, name="create_project", description="Create simple project structure."))
    register_tool(StructuredTool.from_function(analyze_repository, name="analyze_repository", description="Analyze repository structure, file sizes, languages, and provide comprehensive overview."))
    register_tool(StructuredTool.from_function(get_file_content_summary, name="get_file_content_summary", description="Get detailed file content summary with line count and preview."))
else:
    class _CreateProjectTool:
        name = "create_project"
        description = "Create simple project structure."
        def __call__(self, project_name: str, project_type: str = "python"):  # pragma: no cover
            return create_project(project_name, project_type=project_type)
    
    class _AnalyzeRepositoryTool:
        name = "analyze_repository"
        description = "Analyze repository structure, file sizes, languages, and provide comprehensive overview."
        def __call__(self, path: str = ".", max_files: int = 100, include_content: bool = False):  # pragma: no cover
            return analyze_repository(path, max_files, include_content)
    
    class _GetFileContentSummaryTool:
        name = "get_file_content_summary"
        description = "Get detailed file content summary with line count and preview."
        def __call__(self, file_path: str, max_lines: int = 50):  # pragma: no cover
            return get_file_content_summary(file_path, max_lines)
    
    register_tool(_CreateProjectTool())
    register_tool(_AnalyzeRepositoryTool())
    register_tool(_GetFileContentSummaryTool())

__all__ = ["create_project", "analyze_repository", "get_file_content_summary"]
