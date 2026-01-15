import os
import shlex
import subprocess
from typing import List, Optional

from app.settings import require_env


def _run(command: List[str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["GH_TOKEN"] = require_env("GITHUB_TOKEN")
    # non-interactive
    env["GH_PROMPT_DISABLED"] = "1"
    return subprocess.run(command, env=env, capture_output=True, text=True, check=False)


def comment_issue(repository: str, issue_number: int, body: str) -> None:
    cmd = [
        "gh",
        "issue",
        "comment",
        f"{issue_number}",
        "--repo",
        repository,
        "--body",
        body,
    ]
    _run(cmd)


def create_pr(repository: str, title: str, body: str, head: str, base: str = "main") -> None:
    cmd = [
        "gh",
        "pr",
        "create",
        "--repo",
        repository,
        "--title",
        title,
        "--body",
        body,
        "--head",
        head,
        "--base",
        base,
    ]
    _run(cmd)


def clone_repository(repository: str, target_dir: Optional[str] = None) -> str:
    """Clone a repository using GitHub CLI.
    
    Args:
        repository: Repository full name (e.g., "owner/repo")
        target_dir: Target directory to clone into (optional)
    
    Returns:
        Path to cloned repository
    """
    if target_dir:
        cmd = ["gh", "repo", "clone", repository, target_dir]
    else:
        cmd = ["gh", "repo", "clone", repository]
    result = _run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to clone repository {repository}: {result.stderr}")
    # Extract directory name from repository if target_dir not specified
    if not target_dir:
        repo_name = repository.split("/")[-1]
        return repo_name
    return target_dir


def create_branch(repository: str, branch_name: str, base: str = "main") -> None:
    """Create a new branch in the repository using GitHub API.
    
    Args:
        repository: Repository full name (e.g., "owner/repo")
        branch_name: Name of the new branch
        base: Base branch to create from (default: "main")
    """
    # First get the SHA of the base branch
    get_sha_cmd = [
        "gh",
        "api",
        f"repos/{repository}/git/ref/heads/{base}",
        "-q",
        ".object.sha",
    ]
    sha_result = _run(get_sha_cmd)
    if sha_result.returncode != 0:
        raise RuntimeError(f"Failed to get SHA of base branch {base}: {sha_result.stderr}")
    
    base_sha = sha_result.stdout.strip()
    if not base_sha:
        raise RuntimeError(f"Empty SHA returned for base branch {base}")
    
    # Create the new branch by creating a ref
    create_cmd = [
        "gh",
        "api",
        f"repos/{repository}/git/refs",
        "-X",
        "POST",
        "-f",
        f"ref=refs/heads/{branch_name}",
        "-f",
        f"sha={base_sha}",
    ]
    result = _run(create_cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create branch {branch_name}: {result.stderr}")


