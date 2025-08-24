"""Repository Analysis Command - Analyzes project structure and languages."""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import Counter
import subprocess


class RepositoryAnalysisCommand:
    """Command to analyze repository structure, languages, and technologies."""
    
    def __init__(self):
        """Initialize the repository analysis command."""
        self.language_extensions = {
            # Programming Languages
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.jsx': 'React JSX',
            '.tsx': 'React TSX',
            '.java': 'Java',
            '.c': 'C',
            '.cpp': 'C++',
            '.cxx': 'C++',
            '.cc': 'C++',
            '.h': 'C/C++ Header',
            '.hpp': 'C++ Header',
            '.cs': 'C#',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.go': 'Go',
            '.rs': 'Rust',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.scala': 'Scala',
            '.r': 'R',
            '.m': 'Objective-C',
            '.mm': 'Objective-C++',
            
            # Web Technologies
            '.html': 'HTML',
            '.htm': 'HTML',
            '.css': 'CSS',
            '.scss': 'SCSS',
            '.sass': 'Sass',
            '.less': 'Less',
            '.vue': 'Vue.js',
            
            # Configuration & Data
            '.json': 'JSON',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.xml': 'XML',
            '.toml': 'TOML',
            '.ini': 'INI',
            '.cfg': 'Config',
            '.conf': 'Config',
            
            # Documentation
            '.md': 'Markdown',
            '.rst': 'reStructuredText',
            '.txt': 'Text',
            
            # Shell & Scripts
            '.sh': 'Shell Script',
            '.bash': 'Bash Script',
            '.ps1': 'PowerShell',
            '.bat': 'Batch Script',
            '.cmd': 'Command Script',
            
            # Database
            '.sql': 'SQL',
            
            # Other
            '.dockerfile': 'Dockerfile',
            '.gitignore': 'Git Ignore',
        }
    
    def execute(self, query: str, parameters: Dict[str, Any]) -> str:
        """Execute the repository analysis command.
        
        Args:
            query: The original user query
            parameters: Command parameters
            
        Returns:
            Analysis result as formatted string
        """
        print(f"🔍 Analyzing repository structure...")
        
        # Determine repository path (current directory or specified path)
        repo_path = Path.cwd()
        
        # Perform analysis
        analysis_result = {
            "repository_path": str(repo_path),
            "structure": self._analyze_structure(repo_path),
            "languages": self._analyze_languages(repo_path),
            "dependencies": self._analyze_dependencies(repo_path),
            "git_info": self._analyze_git_info(repo_path) if parameters.get("include_git_info") else None,
            "summary": {}
        }
        
        # Generate summary
        analysis_result["summary"] = self._generate_summary(analysis_result)
        
        # Format and return result
        return self._format_analysis_result(analysis_result)
    
    def _analyze_structure(self, repo_path: Path) -> Dict[str, Any]:
        """Analyze the directory structure of the repository."""
        structure = {
            "total_files": 0,
            "total_directories": 0,
            "directory_tree": {},
            "largest_files": [],
            "file_size_distribution": {}
        }
        
        file_sizes = []
        
        try:
            for root, dirs, files in os.walk(repo_path):
                # Skip hidden directories and common ignore patterns
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]
                
                structure["total_directories"] += len(dirs)
                structure["total_files"] += len(files)
                
                # Analyze files in this directory
                for file in files:
                    if file.startswith('.'):
                        continue
                        
                    file_path = Path(root) / file
                    try:
                        file_size = file_path.stat().st_size
                        file_sizes.append((str(file_path.relative_to(repo_path)), file_size))
                    except (OSError, ValueError):
                        continue
            
            # Find largest files
            file_sizes.sort(key=lambda x: x[1], reverse=True)
            structure["largest_files"] = file_sizes[:10]  # Top 10 largest files
            
            # File size distribution
            if file_sizes:
                sizes = [size for _, size in file_sizes]
                structure["file_size_distribution"] = {
                    "total_size": sum(sizes),
                    "average_size": sum(sizes) / len(sizes),
                    "median_size": sorted(sizes)[len(sizes) // 2] if sizes else 0
                }
        
        except Exception as e:
            structure["error"] = str(e)
        
        return structure
    
    def _analyze_languages(self, repo_path: Path) -> Dict[str, Any]:
        """Analyze the programming languages used in the repository."""
        language_stats = Counter()
        file_counts = Counter()
        
        try:
            for root, dirs, files in os.walk(repo_path):
                # Skip hidden directories and common ignore patterns
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]
                
                for file in files:
                    if file.startswith('.'):
                        continue
                    
                    file_path = Path(root) / file
                    ext = file_path.suffix.lower()
                    
                    # Special cases for files without extensions
                    if not ext:
                        if file.lower() in ['dockerfile', 'makefile', 'rakefile']:
                            ext = f'.{file.lower()}'
                        elif file.lower() == 'readme':
                            ext = '.md'
                    
                    if ext in self.language_extensions:
                        language = self.language_extensions[ext]
                        try:
                            file_size = file_path.stat().st_size
                            language_stats[language] += file_size
                            file_counts[language] += 1
                        except OSError:
                            continue
        
        except Exception as e:
            return {"error": str(e)}
        
        # Calculate percentages
        total_size = sum(language_stats.values())
        language_breakdown = {}
        
        for language, size in language_stats.most_common():
            percentage = (size / total_size * 100) if total_size > 0 else 0
            language_breakdown[language] = {
                "bytes": size,
                "files": file_counts[language],
                "percentage": round(percentage, 2)
            }
        
        return {
            "languages": language_breakdown,
            "primary_language": language_stats.most_common(1)[0][0] if language_stats else None,
            "total_language_files": sum(file_counts.values()),
            "total_code_size": total_size
        }
    
    def _analyze_dependencies(self, repo_path: Path) -> Dict[str, Any]:
        """Analyze project dependencies from various config files."""
        dependencies = {}
        
        # Python dependencies
        requirements_files = [
            'requirements.txt', 'requirements-dev.txt', 'dev-requirements.txt',
            'Pipfile', 'pyproject.toml', 'setup.py'
        ]
        
        for req_file in requirements_files:
            req_path = repo_path / req_file
            if req_path.exists():
                dependencies['python'] = self._parse_python_dependencies(req_path)
                break
        
        # Node.js dependencies
        package_json = repo_path / 'package.json'
        if package_json.exists():
            dependencies['nodejs'] = self._parse_nodejs_dependencies(package_json)
        
        # Other dependency files
        other_deps = {
            'Gemfile': 'ruby',
            'go.mod': 'go',
            'Cargo.toml': 'rust',
            'composer.json': 'php',
            'pom.xml': 'java_maven',
            'build.gradle': 'java_gradle'
        }
        
        for dep_file, lang in other_deps.items():
            dep_path = repo_path / dep_file
            if dep_path.exists():
                dependencies[lang] = {"file_found": dep_file, "parsed": False}
        
        return dependencies
    
    def _parse_python_dependencies(self, file_path: Path) -> Dict[str, Any]:
        """Parse Python dependencies from requirements files."""
        try:
            if file_path.name == 'requirements.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                deps = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Simple parsing - just get package name
                        package = line.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0]
                        deps.append(package.strip())
                
                return {"file": file_path.name, "dependencies": deps, "count": len(deps)}
            
            return {"file": file_path.name, "parsed": False}
        
        except Exception as e:
            return {"file": file_path.name, "error": str(e)}
    
    def _parse_nodejs_dependencies(self, file_path: Path) -> Dict[str, Any]:
        """Parse Node.js dependencies from package.json."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                package_data = json.load(f)
            
            deps = package_data.get('dependencies', {})
            dev_deps = package_data.get('devDependencies', {})
            
            return {
                "file": "package.json",
                "dependencies": list(deps.keys()),
                "dev_dependencies": list(dev_deps.keys()),
                "total_deps": len(deps) + len(dev_deps)
            }
        
        except Exception as e:
            return {"file": "package.json", "error": str(e)}
    
    def _analyze_git_info(self, repo_path: Path) -> Dict[str, Any]:
        """Analyze Git repository information."""
        git_info = {}
        
        try:
            # Check if it's a git repository
            git_dir = repo_path / '.git'
            if not git_dir.exists():
                return {"is_git_repo": False}
            
            git_info["is_git_repo"] = True
            
            # Get current branch
            try:
                result = subprocess.run(['git', 'branch', '--show-current'], 
                                      capture_output=True, text=True, cwd=repo_path, timeout=5)
                if result.returncode == 0:
                    git_info["current_branch"] = result.stdout.strip()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            # Get remote information
            try:
                result = subprocess.run(['git', 'remote', '-v'], 
                                      capture_output=True, text=True, cwd=repo_path, timeout=5)
                if result.returncode == 0:
                    remotes = result.stdout.strip().split('\n')
                    git_info["remotes"] = [remote.split()[0] for remote in remotes if remote]
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        except Exception as e:
            git_info["error"] = str(e)
        
        return git_info
    
    def _generate_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the analysis."""
        summary = {}
        
        # Repository overview
        structure = analysis.get("structure", {})
        summary["repository_size"] = {
            "total_files": structure.get("total_files", 0),
            "total_directories": structure.get("total_directories", 0)
        }
        
        # Primary language
        languages = analysis.get("languages", {})
        if languages.get("primary_language"):
            summary["primary_language"] = languages["primary_language"]
            
        # Technologies detected
        technologies = []
        if languages.get("languages"):
            technologies.extend(list(languages["languages"].keys())[:5])  # Top 5 languages
        
        deps = analysis.get("dependencies", {})
        if "python" in deps:
            technologies.append("Python Environment")
        if "nodejs" in deps:
            technologies.append("Node.js Environment")
        
        summary["technologies"] = technologies
        
        # Project type detection
        project_types = []
        if "Python" in technologies:
            project_types.append("Python Project")
        if "JavaScript" in technologies or "TypeScript" in technologies:
            project_types.append("JavaScript/Web Project")
        if "React JSX" in technologies or "React TSX" in technologies:
            project_types.append("React Application")
        
        summary["project_types"] = project_types
        
        return summary
    
    def _format_analysis_result(self, analysis: Dict[str, Any]) -> str:
        """Format the analysis result as a readable string."""
        result = []
        result.append("🔍 Repository Analysis Report")
        result.append("=" * 50)
        
        # Summary section
        summary = analysis.get("summary", {})
        result.append("\n📊 Summary:")
        result.append(f"   Repository: {analysis['repository_path']}")
        
        if summary.get("primary_language"):
            result.append(f"   Primary Language: {summary['primary_language']}")
        
        repo_size = summary.get("repository_size", {})
        result.append(f"   Files: {repo_size.get('total_files', 0)}")
        result.append(f"   Directories: {repo_size.get('total_directories', 0)}")
        
        if summary.get("project_types"):
            result.append(f"   Project Type: {', '.join(summary['project_types'])}")
        
        # Languages section
        languages = analysis.get("languages", {})
        if languages.get("languages"):
            result.append("\n💻 Languages:")
            for lang, stats in list(languages["languages"].items())[:5]:
                result.append(f"   {lang}: {stats['files']} files ({stats['percentage']:.1f}%)")
        
        # Technologies section
        if summary.get("technologies"):
            result.append(f"\n🛠️  Technologies: {', '.join(summary['technologies'])}")
        
        # Dependencies section
        deps = analysis.get("dependencies", {})
        if deps:
            result.append("\n📦 Dependencies:")
            for lang, dep_info in deps.items():
                if isinstance(dep_info, dict) and "count" in dep_info:
                    result.append(f"   {lang.title()}: {dep_info['count']} packages")
                elif isinstance(dep_info, dict) and "total_deps" in dep_info:
                    result.append(f"   {lang.title()}: {dep_info['total_deps']} packages")
        
        # Git information
        git_info = analysis.get("git_info")
        if git_info and git_info.get("is_git_repo"):
            result.append("\n🔗 Git Repository:")
            if git_info.get("current_branch"):
                result.append(f"   Current Branch: {git_info['current_branch']}")
            if git_info.get("remotes"):
                result.append(f"   Remotes: {', '.join(git_info['remotes'])}")
        
        result.append("\n✅ Analysis Complete")
        
        return "\n".join(result)
