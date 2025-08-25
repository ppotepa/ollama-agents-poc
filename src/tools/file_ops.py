"""File operation tools."""
from __future__ import annotations

import os, glob, shutil
import stat
import time
from datetime import datetime
from pathlib import Path

try:
    from langchain.tools import StructuredTool
    from src.tools.registry import register_tool
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False
    from src.tools.registry import register_tool


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    size = size_bytes
    unit_index = 0
    
    while size >= 1024 and unit_index < len(size_names) - 1:
        size /= 1024.0
        unit_index += 1
    
    if unit_index == 0:
        return f"{size:,} {size_names[unit_index]}"
    else:
        return f"{size:.1f} {size_names[unit_index]}"


def _format_file_permissions(filepath: str) -> str:
    """Format file permissions in Windows-style format."""
    try:
        file_stat = os.stat(filepath)
        mode = file_stat.st_mode
        
        # Basic permissions
        perms = []
        if os.path.isdir(filepath):
            perms.append("D")  # Directory
        elif os.path.isfile(filepath):
            perms.append("-")  # Regular file
        else:
            perms.append("?")  # Unknown
            
        # Read/Write attributes (simplified for Windows)
        if os.access(filepath, os.R_OK):
            perms.append("R")
        else:
            perms.append("-")
            
        if os.access(filepath, os.W_OK):
            perms.append("W")
        else:
            perms.append("-")
            
        # Hidden attribute
        if os.name == 'nt':  # Windows
            import ctypes
            attrs = ctypes.windll.kernel32.GetFileAttributesW(filepath)
            if attrs != -1 and attrs & 2:  # FILE_ATTRIBUTE_HIDDEN
                perms.append("H")
            else:
                perms.append("-")
        else:
            if os.path.basename(filepath).startswith('.'):
                perms.append("H")
            else:
                perms.append("-")
                
        return "".join(perms)
    except:
        return "----"


def _format_file_date(filepath: str) -> str:
    """Format file modification date."""
    try:
        mtime = os.path.getmtime(filepath)
        dt = datetime.fromtimestamp(mtime)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return "Unknown"


def get_file_info(filepath: str) -> str:
    """Get detailed information about a single file or directory."""
    if not os.path.exists(filepath):
        return f"❌ Path {filepath} does not exist"
    
    try:
        stat_result = os.stat(filepath)
        path_obj = Path(filepath)
        
        info_lines = []
        info_lines.append(f"📁 Path: {os.path.abspath(filepath)}")
        
        if os.path.isfile(filepath):
            info_lines.append(f"📄 Type: File")
            info_lines.append(f"📏 Size: {_format_file_size(stat_result.st_size)} ({stat_result.st_size:,} bytes)")
        elif os.path.isdir(filepath):
            info_lines.append(f"📁 Type: Directory")
            try:
                # Count items in directory
                items = list(os.listdir(filepath))
                files = sum(1 for item in items if os.path.isfile(os.path.join(filepath, item)))
                dirs = sum(1 for item in items if os.path.isdir(os.path.join(filepath, item)))
                info_lines.append(f"📊 Contents: {files} files, {dirs} directories")
            except PermissionError:
                info_lines.append(f"📊 Contents: Access denied")
        
        info_lines.append(f"🔐 Permissions: {_format_file_permissions(filepath)}")
        info_lines.append(f"📅 Modified: {_format_file_date(filepath)}")
        info_lines.append(f"📅 Created: {datetime.fromtimestamp(stat_result.st_ctime).strftime('%Y-%m-%d %H:%M:%S')}")
        info_lines.append(f"📅 Accessed: {datetime.fromtimestamp(stat_result.st_atime).strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(info_lines)
        
    except Exception as e:
        return f"❌ Error getting file info: {e}"


def write_file(filepath: str, contents: str) -> str:
    """Write contents to a file with detailed feedback."""
    try:
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        
        # Check if file exists for feedback
        file_existed = os.path.exists(filepath)
        old_size = 0
        if file_existed:
            old_size = os.path.getsize(filepath)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(contents)
        
        new_size = len(contents.encode('utf-8'))
        abs_path = os.path.abspath(filepath)
        
        if file_existed:
            size_change = new_size - old_size
            change_str = f"({size_change:+} bytes)" if size_change != 0 else "(no size change)"
            return f"✅ File updated: {abs_path}\n📏 Size: {_format_file_size(new_size)} {change_str}\n📅 Modified: {_format_file_date(filepath)}"
        else:
            return f"✅ File created: {abs_path}\n📏 Size: {_format_file_size(new_size)}\n📅 Created: {_format_file_date(filepath)}"
            
    except Exception as e:
        return f"❌ Error writing file {filepath}: {e}"


def read_file(filepath: str, show_info: bool = True) -> str:
    """Read file contents with optional file information."""
    if not os.path.exists(filepath):
        return f"❌ File {filepath} does not exist"
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        if not show_info:
            return content
        
        # Add file information
        size = os.path.getsize(filepath)
        lines = content.count('\n') + 1 if content else 0
        chars = len(content)
        
        info_header = f"📄 File: {os.path.abspath(filepath)}\n"
        info_header += f"📏 Size: {_format_file_size(size)} ({lines:,} lines, {chars:,} characters)\n"
        info_header += f"📅 Modified: {_format_file_date(filepath)}\n"
        info_header += "─" * 50 + "\n"
        
        return info_header + content
        
    except UnicodeDecodeError:
        return f"❌ Cannot read {filepath}: File appears to be binary"
    except Exception as e:
        return f"❌ Error reading file {filepath}: {e}"


def append_file(filepath: str, contents: str) -> str:
    """Append contents to a file with detailed feedback."""
    try:
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        
        old_size = 0
        if os.path.exists(filepath):
            old_size = os.path.getsize(filepath)
        
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(contents)
        
        new_size = os.path.getsize(filepath)
        appended_size = len(contents.encode('utf-8'))
        
        return f"✅ Appended to {os.path.abspath(filepath)}\n📏 Added: {_format_file_size(appended_size)}\n📏 Total size: {_format_file_size(new_size)}\n📅 Modified: {_format_file_date(filepath)}"
        
    except Exception as e:
        return f"❌ Error appending to file {filepath}: {e}"


def list_files(directory: str = ".", pattern: str = "*", detailed: bool = True) -> str:
    """List files and directories with detailed information similar to dir command."""
    if not os.path.exists(directory):
        return f"❌ Directory {directory} does not exist"
    
    try:
        # Get all items matching the pattern
        search_pattern = os.path.join(directory, pattern)
        all_items = glob.glob(search_pattern)
        
        if not all_items:
            return f"📂 No items found matching pattern '{pattern}' in {directory}"
        
        if not detailed:
            # Simple listing (original behavior)
            items = []
            for item in all_items:
                if os.path.isfile(item):
                    items.append(os.path.relpath(item))
                elif os.path.isdir(item):
                    items.append(os.path.relpath(item) + "/")
            return "\n".join(items)
        
        # Detailed listing (dir-style)
        result_lines = []
        result_lines.append(f"📁 Directory of {os.path.abspath(directory)}")
        result_lines.append("")
        
        # Header
        result_lines.append(f"{'Date/Time':<20} {'Perms':<6} {'Size':<12} {'Name'}")
        result_lines.append("-" * 60)
        
        # Sort items: directories first, then files
        directories = []
        files = []
        
        for item in all_items:
            if os.path.isdir(item):
                directories.append(item)
            else:
                files.append(item)
        
        directories.sort()
        files.sort()
        
        total_files = 0
        total_size = 0
        
        # List directories first
        for item in directories:
            name = os.path.basename(item)
            date_str = _format_file_date(item)
            perms = _format_file_permissions(item)
            
            result_lines.append(f"{date_str:<20} {perms:<6} {'<DIR>':<12} {name}/")
        
        # List files
        for item in files:
            try:
                name = os.path.basename(item)
                date_str = _format_file_date(item)
                perms = _format_file_permissions(item)
                size = os.path.getsize(item)
                size_str = _format_file_size(size)
                
                result_lines.append(f"{date_str:<20} {perms:<6} {size_str:<12} {name}")
                total_files += 1
                total_size += size
                
            except (OSError, IOError):
                # Handle files that can't be accessed
                result_lines.append(f"{'Unknown':<20} {'----':<6} {'<ERROR>':<12} {os.path.basename(item)}")
        
        # Summary
        result_lines.append("-" * 60)
        result_lines.append(f"� {len(directories)} directories, {total_files} files")
        result_lines.append(f"📏 Total size: {_format_file_size(total_size)} ({total_size:,} bytes)")
        
        return "\n".join(result_lines)
        
    except Exception as e:
        return f"❌ Error listing directory: {e}"


def list_files_recurse(directory: str = ".", pattern: str = "*", detailed: bool = True, max_depth: int = 10) -> str:
    """Recursively list all files and directories with detailed information."""
    if not os.path.exists(directory):
        return f"❌ Directory {directory} does not exist"
    
    if max_depth <= 0:
        return f"❌ Max depth must be greater than 0 (provided: {max_depth})"
    
    try:
        result_lines = []
        result_lines.append(f"📁 Recursive Directory Listing of {os.path.abspath(directory)}")
        result_lines.append(f"🔍 Pattern: {pattern}, Max Depth: {max_depth}")
        result_lines.append("")
        
        if detailed:
            result_lines.append(f"{'Path':<60} {'Date/Time':<20} {'Perms':<6} {'Size':<12}")
            result_lines.append("=" * 120)
        
        total_files = 0
        total_directories = 0
        total_size = 0
        current_depth = 0
        
        def _walk_directory(current_dir: str, depth: int, prefix: str = ""):
            """Recursively walk directory structure."""
            nonlocal total_files, total_directories, total_size
            
            if depth > max_depth:
                return
            
            try:
                # Get items in current directory
                search_pattern = os.path.join(current_dir, pattern)
                items = glob.glob(search_pattern)
                
                if not items and depth == 0:
                    # If no items match pattern in root, try to list all items
                    items = glob.glob(os.path.join(current_dir, "*"))
                
                # Separate directories and files
                directories = []
                files = []
                
                for item in items:
                    if os.path.isdir(item):
                        directories.append(item)
                    elif os.path.isfile(item):
                        files.append(item)
                
                directories.sort()
                files.sort()
                
                # Process directories first
                for dir_path in directories:
                    total_directories += 1
                    rel_path = os.path.relpath(dir_path, directory)
                    display_path = f"{prefix}📁 {rel_path}/"
                    
                    if detailed:
                        date_str = _format_file_date(dir_path)
                        perms = _format_file_permissions(dir_path)
                        result_lines.append(f"{display_path:<60} {date_str:<20} {perms:<6} {'<DIR>':<12}")
                    else:
                        result_lines.append(display_path)
                    
                    # Recursively process subdirectory
                    if depth < max_depth:
                        _walk_directory(dir_path, depth + 1, prefix + "  ")
                
                # Process files
                for file_path in files:
                    try:
                        total_files += 1
                        rel_path = os.path.relpath(file_path, directory)
                        display_path = f"{prefix}📄 {rel_path}"
                        
                        if detailed:
                            date_str = _format_file_date(file_path)
                            perms = _format_file_permissions(file_path)
                            size = os.path.getsize(file_path)
                            size_str = _format_file_size(size)
                            total_size += size
                            
                            result_lines.append(f"{display_path:<60} {date_str:<20} {perms:<6} {size_str:<12}")
                        else:
                            result_lines.append(display_path)
                            
                    except (OSError, IOError) as e:
                        # Handle files that can't be accessed
                        display_path = f"{prefix}❌ {os.path.relpath(file_path, directory)}"
                        if detailed:
                            result_lines.append(f"{display_path:<60} {'Error':<20} {'----':<6} {'<ERROR>':<12}")
                        else:
                            result_lines.append(display_path)
                
            except Exception as e:
                result_lines.append(f"{prefix}❌ Error accessing {current_dir}: {e}")
        
        # Start recursive walk
        _walk_directory(directory, 0)
        
        # Summary
        result_lines.append("=" * 120)
        result_lines.append(f"📊 Summary:")
        result_lines.append(f"   📁 {total_directories} directories")
        result_lines.append(f"   📄 {total_files} files")
        if detailed:
            result_lines.append(f"   📏 Total size: {_format_file_size(total_size)} ({total_size:,} bytes)")
        result_lines.append(f"   🔍 Max depth reached: {min(max_depth, current_depth)}")
        
        return "\n".join(result_lines)
        
    except Exception as e:
        return f"❌ Error during recursive listing: {e}"


def delete_file(filepath: str) -> str:
    """Delete a file or directory with detailed feedback."""
    if not os.path.exists(filepath):
        return f"❌ Path {filepath} does not exist"
    
    try:
        abs_path = os.path.abspath(filepath)
        
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            os.remove(filepath)
            return f"🗑️ Deleted file: {abs_path}\n📏 Freed space: {_format_file_size(size)}"
            
        elif os.path.isdir(filepath):
            # Calculate directory size before deletion
            total_size = 0
            file_count = 0
            dir_count = 0
            
            for root, dirs, files in os.walk(filepath):
                dir_count += len(dirs)
                for file in files:
                    file_count += 1
                    try:
                        total_size += os.path.getsize(os.path.join(root, file))
                    except:
                        pass
            
            shutil.rmtree(filepath)
            return f"🗑️ Deleted directory: {abs_path}\n📊 Removed: {file_count} files, {dir_count} directories\n📏 Freed space: {_format_file_size(total_size)}"
            
    except PermissionError:
        return f"❌ Permission denied: Cannot delete {filepath}"
    except Exception as e:
        return f"❌ Error deleting {filepath}: {e}"


def copy_file(source: str, destination: str) -> str:
    """Copy a file or directory with detailed feedback."""
    if not os.path.exists(source):
        return f"❌ Source {source} does not exist"
    
    try:
        abs_source = os.path.abspath(source)
        abs_dest = os.path.abspath(destination)
        
        if os.path.isfile(source):
            parent = os.path.dirname(destination)
            if parent:
                os.makedirs(parent, exist_ok=True)
            
            source_size = os.path.getsize(source)
            shutil.copy2(source, destination)  # copy2 preserves metadata
            
            return f"📋 Copied file: {abs_source} → {abs_dest}\n📏 Size: {_format_file_size(source_size)}\n📅 Copied: {_format_file_date(destination)}"
            
        elif os.path.isdir(source):
            # Calculate directory contents
            total_size = 0
            file_count = 0
            dir_count = 0
            
            for root, dirs, files in os.walk(source):
                dir_count += len(dirs)
                for file in files:
                    file_count += 1
                    try:
                        total_size += os.path.getsize(os.path.join(root, file))
                    except:
                        pass
            
            shutil.copytree(source, destination, dirs_exist_ok=True)
            
            return f"📋 Copied directory: {abs_source} → {abs_dest}\n📊 Copied: {file_count} files, {dir_count} directories\n📏 Total size: {_format_file_size(total_size)}"
            
    except PermissionError:
        return f"❌ Permission denied: Cannot copy {source}"
    except FileExistsError:
        return f"❌ Destination {destination} already exists"
    except Exception as e:
        return f"❌ Error copying {source}: {e}"


if LANGCHAIN_AVAILABLE:
    for fn in [write_file, read_file, append_file, list_files, list_files_recurse, delete_file, copy_file, get_file_info]:
        register_tool(StructuredTool.from_function(fn, name=fn.__name__, description=fn.__doc__ or fn.__name__))
else:
    class _Wrap:
        def __init__(self, fn):
            self._fn = fn
            self.name = fn.__name__
            self.description = fn.__doc__ or fn.__name__
        def __call__(self, *a, **kw):  # pragma: no cover
            return self._fn(*a, **kw)
    for _f in [write_file, read_file, append_file, list_files, list_files_recurse, delete_file, copy_file, get_file_info]:
        register_tool(_Wrap(_f))

__all__ = ["write_file", "read_file", "append_file", "list_files", "list_files_recurse", "delete_file", "copy_file", "get_file_info"]
