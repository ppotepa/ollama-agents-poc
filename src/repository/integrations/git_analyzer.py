"""Git integration service for repository analysis.

Single Responsibility: Extract Git repository information and metadata.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


class GitAnalyzer:
    """Analyzes Git repository information and metadata.

    Responsibility: Extract Git-specific information like branch, commits, remotes.
    """

    def __init__(self, timeout_seconds: int = 5):
        self.timeout = timeout_seconds

    def is_git_repository(self, repo_path: Path) -> bool:
        """Check if the given path is a Git repository.

        Args:
            repo_path: Path to check

        Returns:
            True if path is a Git repository
        """
        git_dir = repo_path / '.git'
        return git_dir.exists()

    def run_git_command(self, repo_path: Path, command: list[str]) -> str | None:
        """Run a git command and return the output.

        Args:
            repo_path: Repository path
            command: Git command as list of strings

        Returns:
            Command output or None if failed
        """
        try:
            result = subprocess.run(
                ['git'] + command,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode == 0:
                return result.stdout.strip()
            return None

        except Exception:
            return None

    def get_current_branch(self, repo_path: Path) -> str | None:
        """Get the current Git branch.

        Args:
            repo_path: Repository path

        Returns:
            Current branch name or None if not available
        """
        return self.run_git_command(repo_path, ['branch', '--show-current'])

    def get_remote_url(self, repo_path: Path, remote: str = 'origin') -> str | None:
        """Get the remote URL for a given remote name.

        Args:
            repo_path: Repository path
            remote: Remote name (default: 'origin')

        Returns:
            Remote URL or None if not available
        """
        return self.run_git_command(repo_path, ['remote', 'get-url', remote])

    def get_commit_count(self, repo_path: Path, branch: str | None = None) -> int | None:
        """Get the total number of commits.

        Args:
            repo_path: Repository path
            branch: Branch name (default: current branch)

        Returns:
            Number of commits or None if not available
        """
        command = ['rev-list', '--count']
        command.append(branch or 'HEAD')

        result = self.run_git_command(repo_path, command)
        if result:
            try:
                return int(result)
            except ValueError:
                return None
        return None

    def get_last_commit_hash(self, repo_path: Path, short: bool = True) -> str | None:
        """Get the hash of the last commit.

        Args:
            repo_path: Repository path
            short: Return short hash if True

        Returns:
            Commit hash or None if not available
        """
        format_option = '--short' if short else ''
        command = ['rev-parse'] + ([format_option] if format_option else []) + ['HEAD']
        return self.run_git_command(repo_path, command)

    def get_last_commit_message(self, repo_path: Path) -> str | None:
        """Get the message of the last commit.

        Args:
            repo_path: Repository path

        Returns:
            Last commit message or None if not available
        """
        return self.run_git_command(repo_path, ['log', '-1', '--pretty=format:%s'])

    def get_last_commit_author(self, repo_path: Path) -> str | None:
        """Get the author of the last commit.

        Args:
            repo_path: Repository path

        Returns:
            Last commit author or None if not available
        """
        return self.run_git_command(repo_path, ['log', '-1', '--pretty=format:%an'])

    def get_last_commit_date(self, repo_path: Path) -> str | None:
        """Get the date of the last commit.

        Args:
            repo_path: Repository path

        Returns:
            Last commit date or None if not available
        """
        return self.run_git_command(repo_path, ['log', '-1', '--pretty=format:%ai'])

    def get_repository_status(self, repo_path: Path) -> dict[str, Any]:
        """Get the working directory status.

        Args:
            repo_path: Repository path

        Returns:
            Dictionary with status information
        """
        status = {}

        # Check if there are uncommitted changes
        result = self.run_git_command(repo_path, ['status', '--porcelain'])
        if result is not None:
            status['has_uncommitted_changes'] = bool(result.strip())
            status['uncommitted_files_count'] = len(result.strip().split('\n')) if result.strip() else 0

        # Check if there are untracked files
        result = self.run_git_command(repo_path, ['ls-files', '--others', '--exclude-standard'])
        if result is not None:
            status['has_untracked_files'] = bool(result.strip())
            status['untracked_files_count'] = len(result.strip().split('\n')) if result.strip() else 0

        return status

    def analyze_git_repository(self, repo_path: Path) -> dict[str, Any] | None:
        """Perform comprehensive Git repository analysis.

        Args:
            repo_path: Repository path

        Returns:
            Dictionary with all Git information or None if not a Git repo
        """
        if not self.is_git_repository(repo_path):
            return None

        git_info = {}

        # Basic repository information
        if branch := self.get_current_branch(repo_path):
            git_info['branch'] = branch

        if remote_url := self.get_remote_url(repo_path):
            git_info['remote_url'] = remote_url

        if commit_count := self.get_commit_count(repo_path):
            git_info['commit_count'] = commit_count

        # Last commit information
        if commit_hash := self.get_last_commit_hash(repo_path):
            git_info['last_commit_hash'] = commit_hash

        if commit_message := self.get_last_commit_message(repo_path):
            git_info['last_commit_message'] = commit_message

        if commit_author := self.get_last_commit_author(repo_path):
            git_info['last_commit_author'] = commit_author

        if commit_date := self.get_last_commit_date(repo_path):
            git_info['last_commit_date'] = commit_date

        # Repository status
        status = self.get_repository_status(repo_path)
        git_info.update(status)

        return git_info if git_info else None
