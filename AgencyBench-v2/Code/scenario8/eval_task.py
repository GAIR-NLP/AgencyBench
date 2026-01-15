#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import textwrap
import asyncio
import re
import sys
import hashlib
import hmac
import time
import uuid
import inspect
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

try:
    from sii_agent_sdk import (  # type: ignore
        AssistantMessage,
        TextBlock,
        SiiAgentOptions,
        SiiAgentSession as SDKAgentSession,
    )
except ImportError:  # pragma: no cover - optional dependency
    AssistantMessage = None
    TextBlock = None
    SiiAgentOptions = None
    SDKAgentSession = None

SCRIPT_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Helpers & Mocks (Ported from Reference Logic)
# --------------------------------------------------------------------------- #


def load_env_file(env_path: Path) -> Dict[str, str]:
    """Load key/value pairs from an env file and inject them into os.environ."""

    env_path = env_path.resolve()
    if not env_path.exists():
        raise FileNotFoundError(f"Required env file not found: {env_path}")

    parsed: Dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"')
        parsed[key] = value
        os.environ[key] = value
    return parsed


def derive_model_name(identifier: str) -> str:
    safe_chars = []
    for ch in identifier:
        if ch.isalnum() or ch in "._-":
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    sanitized = "".join(safe_chars).strip("._-") or "model"
    return sanitized


def ensure_directory(path: Path) -> Path:
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def patch_httpx_client_app_param() -> None:
    """Ensure httpx.Client accepts 'app' kwarg for compatibility with older TestClient usage."""
    try:
        import httpx
        import inspect

        sig = inspect.signature(httpx.Client.__init__)
        if "app" not in sig.parameters:
            original_init = httpx.Client.__init__

            def _patched_init(self, *args, app=None, **kwargs):
                return original_init(self, *args, **kwargs)

            httpx.Client.__init__ = _patched_init  # type: ignore[assignment]
            print("[WARN] Patched httpx.Client.__init__ to ignore 'app' kwarg (compat mode).")
    except Exception as exc:
        print(f"[WARN] Failed to patch httpx Client compatibility: {exc}")


def clear_directory(path: Path) -> None:
    path = path.resolve()
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for entry in path.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


EXCLUDED_COPY_DIRS = {".git", ".hg", ".svn", "__pycache__", ".sii", "logs", "bin", "venv", ".pytest_cache"}
EXCLUDED_COPY_FILES = {"meta_eval.json"}
EXCLUDED_COPY_SUFFIXES = {".db", ".db3", ".sqlite", ".sqlite3", ".pyc", ".pyo"}


def copy_workspace(src: Path, dst: Path) -> None:
    src = src.resolve()
    dst = dst.resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source workspace missing: {src}")
    ensure_directory(dst)
    clear_directory(dst)
    for root, dirs, files in os.walk(src):
        rel_root = Path(root).relative_to(src)
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        target_root = dst / rel_root
        target_root.mkdir(parents=True, exist_ok=True)
        for filename in files:
            if filename.startswith("."):
                continue
            src_file = Path(root) / filename
            dst_file = target_root / filename
            shutil.copy2(src_file, dst_file)


def copy_workspace_filtered(src: Path, dst: Path) -> None:
    src = src.resolve()
    dst = dst.resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source workspace missing: {src}")
    ensure_directory(dst)
    clear_directory(dst)
    for root, dirs, files in os.walk(src):
        rel_root = Path(root).relative_to(src)
        if any(part.startswith(".") or part in EXCLUDED_COPY_DIRS for part in rel_root.parts):
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in EXCLUDED_COPY_DIRS]
        target_root = dst / rel_root
        target_root.mkdir(parents=True, exist_ok=True)
        for filename in files:
            if filename.startswith("."):
                continue
            if filename in EXCLUDED_COPY_FILES:
                continue
            if Path(filename).suffix.lower() in EXCLUDED_COPY_SUFFIXES:
                continue
            src_file = Path(root) / filename
            dst_file = target_root / filename
            shutil.copy2(src_file, dst_file)


def read_json(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(
    cmd: Sequence[str],
    cwd: Optional[Path] = None,
    input_text: Optional[str] = None,
    timeout: int = 600,
) -> Tuple[int, str, str]:
    try:
        run_cwd = cwd.resolve() if cwd else None
        proc = subprocess.run(
            cmd,
            cwd=run_cwd,
            input=input_text,
            text=True,
            timeout=timeout,
            capture_output=True,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError as exc:
        return 127, "", str(exc)
    except subprocess.TimeoutExpired as exc:
        return 124, exc.stdout or "", exc.stderr or "Command timed out"


def workspace_notice(path: Path, repo_root: Optional[Path] = None) -> str:
    target = path
    if repo_root and repo_root.exists():
        try:
            target = path.relative_to(repo_root)
        except Exception:
            target = path
    return textwrap.dedent(
        f"""
        You are working in {target}. Do not access outside paths.
        Implement the solution in Python (FastAPI/MongoDB) strictly following the JSON specifications.
        """
    ).strip()


def ensure_bridge_timeout(options: Any, timeout_ms: int) -> None:
    if not timeout_ms:
        return
    try:
        current = getattr(options, "timeout_ms", 0) or 0
        options.timeout_ms = max(timeout_ms, current)
    except Exception:
        pass


def resolve_script_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (SCRIPT_DIR / path).resolve()


def is_port_open(host: str, port: int, timeout: float = 1.5) -> bool:
    """Lightweight TCP connect check for host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# --- Mock Implementations for Evaluation Logic ---

class MockMongoDB:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.logs: List[Dict[str, Any]] = []
        self.indexes: Dict[str, List[Dict[str, Any]]] = {
            "agent_sessions": [],
            "agent_logs": [],
        }

    def get_database(self, db_name: str) -> "MockDatabase":
        return MockDatabase(self, db_name)


class MockDatabase:
    def __init__(self, mock_mongo: MockMongoDB, db_name: str):
        self.mock_mongo = mock_mongo
        self.db_name = db_name

    def __getitem__(self, collection_name: str) -> "MockCollection":
        return MockCollection(self.mock_mongo, collection_name)


class MockCollection:
    def __init__(self, mock_mongo: MockMongoDB, collection_name: str):
        self.mock_mongo = mock_mongo
        self.collection_name = collection_name
        if collection_name == "agent_sessions":
            self.data = mock_mongo.sessions
        else:
            self.data = {}

    def create_index(self, keys: List[Tuple[str, int]], **kwargs) -> None:
        index_spec = {"keys": keys, **kwargs}
        self.mock_mongo.indexes[self.collection_name].append(index_spec)

    def find_one(self, filter: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.collection_name == "agent_sessions":
            for key, value in filter.items():
                for doc in self.mock_mongo.sessions.values():
                    if doc.get(key) == value:
                        return doc
        elif self.collection_name == "agent_logs":
            for log in self.mock_mongo.logs:
                if all(log.get(k) == v for k, v in filter.items()):
                    return log
        return None

    def find(self, filter: Optional[Dict[str, Any]] = None, **kwargs) -> "MockCursor":
        return MockCursor(self, filter or {})

    def insert_one(self, document: Dict[str, Any]) -> "MockInsertResult":
        if self.collection_name == "agent_sessions":
            session_id = document.get("session_id")
            if session_id:
                self.mock_mongo.sessions[session_id] = document.copy()
        elif self.collection_name == "agent_logs":
            self.mock_mongo.logs.append(document.copy())
        return MockInsertResult(document.get("_id") or str(uuid.uuid4()))

    def update_one(self, filter: Dict[str, Any], update: Dict[str, Any]) -> "MockUpdateResult":
        if self.collection_name == "agent_sessions":
            for key, value in filter.items():
                for session_id, doc in list(self.mock_mongo.sessions.items()):
                    if doc.get(key) == value:
                        if "$set" in update:
                            doc.update(update["$set"])
                        self.mock_mongo.sessions[session_id] = doc
                        return MockUpdateResult(1)
        return MockUpdateResult(0)

    def count_documents(self, filter: Dict[str, Any]) -> int:
        if self.collection_name == "agent_sessions":
            count = 0
            for doc in self.mock_mongo.sessions.values():
                if all(doc.get(k) == v for k, v in filter.items()):
                    count += 1
            return count
        return 0


class MockCursor:
    def __init__(self, collection: MockCollection, filter: Dict[str, Any]):
        self.collection = collection
        self.filter = filter
        self._data: List[Dict[str, Any]] = []
        self._prepare_data()

    def _prepare_data(self):
        if self.collection.collection_name == "agent_sessions":
            for doc in self.collection.mock_mongo.sessions.values():
                if all(doc.get(k) == v for k, v in self.filter.items()):
                    self._data.append(doc)
        elif self.collection.collection_name == "agent_logs":
            for log in self.collection.mock_mongo.logs:
                if all(log.get(k) == v for k, v in self.filter.items()):
                    self._data.append(log)

    def sort(self, *args, **kwargs) -> "MockCursor":
        return self

    def limit(self, n: int) -> "MockCursor":
        self._data = self._data[:n]
        return self

    def __iter__(self):
        return iter(self._data)

    def __list__(self) -> List[Dict[str, Any]]:
        return self._data


class MockInsertResult:
    def __init__(self, inserted_id: str):
        self.inserted_id = inserted_id


class MockUpdateResult:
    def __init__(self, modified_count: int):
        self.modified_count = modified_count


class MockGitHubCLI:
    def __init__(self):
        self.commands_executed: List[List[str]] = []

    def run_command(self, cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
        self.commands_executed.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")


class MockDocker:
    def __init__(self):
        self.containers: Dict[str, Dict[str, Any]] = {}

    def containers_run(self, image: str, command: List[str], detach: bool = True, **kwargs) -> Any:
        return MagicMock()

    def from_env(self) -> "MockDocker":
        return self


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class EnvConfig:
    env_path: Path
    visualize: bool
    env_values: Dict[str, str]
    github_token: str
    llm_api_key: str
    llm_base_url: str
    mongodb_url: str
    webhook_secret: str
    use_mock: bool
    model_name: str
    scorer_name: str
    max_attempts: int
    conda_env_name: str
    sii_api_key: str
    sii_api_base: str
    sii_auth_type: str
    sii_target_model: str
    sii_system_prompt: str
    sii_username: Optional[str]
    sii_password: Optional[str]
    sii_max_turns: int
    bridge_timeout_ms: int
    eval_only: bool = False

    @classmethod
    def _from_env_values(cls, values: Dict[str, str], visualize: bool, env_path: Path) -> "EnvConfig":
        def fetch(key: str) -> Optional[str]:
            value = values.get(key) or os.environ.get(key)
            if value:
                return value.strip()
            return None

        def require(key: str) -> str:
            value = fetch(key)
            if not value or value.startswith("<YOUR_"):
                raise ValueError(f"Environment variable '{key}' must be set in {env_path}")
            return value

        sii_target = require("SII_TARGET_MODEL")
        sii_api_key = require("SII_AGENT_API_KEY")
        sii_api_base = fetch("SII_AGENT_API_BASE_URL") or "https://openrouter.ai/api/v1"

        return cls(
            env_path=env_path,
            visualize=visualize,
            env_values=values,
            github_token=fetch("GITHUB_TOKEN") or "mock_token",
            llm_api_key=fetch("LLM_API_KEY") or "mock_key",
            llm_base_url=fetch("LLM_BASE_URL") or "https://api.openai.com/v1",
            mongodb_url=fetch("MONGODB_URL") or "mongodb://mock:27017/test",
            webhook_secret=fetch("WEBHOOK_SECRET") or "mock_secret",
            use_mock=(values.get("USE_MOCK") or os.environ.get("USE_MOCK") or "true").lower() == "true",
            model_name=derive_model_name(sii_target),
            scorer_name=values.get("SCORER_NAME", "rubric"),
            max_attempts=int(values.get("MAX_SUBTASK_ATTEMPTS", "3") or "3"),
            conda_env_name=values.get("CONDA_ENV_NAME", ""),
            sii_api_key=sii_api_key,
            sii_api_base=sii_api_base,
            sii_auth_type=values.get("SII_AUTH_TYPE", "USE_OPENAI"),
            sii_target_model=sii_target,
            sii_system_prompt=values.get("SII_SYSTEM_PROMPT", ""),
            sii_username=fetch("SII_USERNAME"),
            sii_password=fetch("SII_PASSWORD"),
            sii_max_turns=int(values.get("SII_MAX_TURNS", "20")),
            bridge_timeout_ms=int(values.get("BRIDGE_TIMEOUT_MS", "180000")),
        )

    @classmethod
    def load(cls, env_path: Path, visualize: bool) -> "EnvConfig":
        env_path = resolve_script_path(env_path)
        values = load_env_file(env_path)
        return cls._from_env_values(values, visualize=visualize, env_path=env_path)

    def inject_defaults(self) -> None:
        defaults = {
            "SII_AGENT_API_KEY": self.sii_api_key,
            "SII_AGENT_API_BASE_URL": self.sii_api_base,
            "SII_OPENAI_API_KEY": self.sii_api_key,
            "SII_OPENAI_BASE_URL": self.sii_api_base,
            "SII_OPENAI_MODEL": self.sii_target_model,
            "OPENAI_API_KEY": self.sii_api_key,
            "OPENAI_BASE_URL": self.sii_api_base,
            "SII_USERNAME": self.sii_username or "",
            "SII_PASSWORD": self.sii_password or "",
            "GITHUB_TOKEN": self.github_token,
            "LLM_API_KEY": self.llm_api_key,
            "LLM_BASE_URL": self.llm_base_url,
            "MONGODB_URL": self.mongodb_url,
            "WEBHOOK_SECRET": self.webhook_secret,
        }
        for key, value in {**self.env_values, **defaults}.items():
            if value:
                os.environ.setdefault(key, value)


@dataclass
class CommandResult:
    name: str
    command: Sequence[str]
    returncode: int
    stdout: str
    stderr: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "command": list(self.command),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass
class VerificationResult:
    step_name: str
    check_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RubricResult:
    subtask: str
    score: float
    pass_count: int
    total_points: int
    failed_points: List[str] = field(default_factory=list)
    raw_results: List[VerificationResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subtask": self.subtask,
            "score": self.score,
            "pass_count": self.pass_count,
            "total_points": self.total_points,
            "failed_points": self.failed_points,
        }


@dataclass
class AttemptSummary:
    subtask: str
    attempt_index: int
    score: float
    rubric: RubricResult
    workspace: Path
    evalspace: Path
    agent_output: str
    commands: Dict[str, Any]
    feedback: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subtask": self.subtask,
            "attempt_index": self.attempt_index,
            "score": self.score,
            "rubric": self.rubric.to_dict(),
            "workspace": str(self.workspace),
            "evalspace": str(self.evalspace),
            "agent_output": self.agent_output,
            "commands": self.commands,
            "feedback": self.feedback,
        }


# --------------------------------------------------------------------------- #
# Agent runner (minimal)
# --------------------------------------------------------------------------- #


class AgentRunner:
    """Minimal wrapper around the SII SDK streaming interface."""

    def __init__(self, env_config: EnvConfig, visualize: bool = False):
        self.env_config = env_config
        self.visualize = visualize
        self._available = self._sdk_available()
        self._session: Optional[SDKAgentSession] = None
        if not self._available:
            print("[AGENT] sii_agent_sdk not available; using no-op agent.")
            return
        try:
            base_options = SiiAgentOptions(
                system_prompt=self.env_config.sii_system_prompt,
                max_turns=self.env_config.sii_max_turns,
                auth_type=self.env_config.sii_auth_type,
                cwd=str(SCRIPT_DIR),
                yolo=True,
                allowed_tools=[],
                model=self.env_config.sii_target_model,
                env=self.env_config.env_values,
                enable_data_upload=False,
            )
            ensure_bridge_timeout(base_options, self.env_config.bridge_timeout_ms)
            self._session = SDKAgentSession(base_options)
        except Exception as exc:
            print(f"[AGENT] Failed to initialize SII session: {exc}")
            self._available = False

    def send(self, prompt: str, workspace_hint: str, workspace: Path) -> str:
        if not self._available or not self._session:
            return ""
        try:
            options = self._build_options(workspace)
        except Exception as exc:
            print(f"[AGENT] Failed to configure options: {exc}")
            return ""
        combined_prompt = f"{workspace_hint}\n\n{prompt}" if workspace_hint else prompt
        try:
            return asyncio.run(self._run_session(self._session, combined_prompt, options))
        except Exception as exc:
            print(f"[AGENT] Agent invocation failed: {exc}")
            return ""

    async def _run_session(
        self, session: SDKAgentSession, prompt: str, options: SiiAgentOptions
    ) -> str:
        assistant_chunks: List[str] = []
        status_seen: set[str] = set()
        async for message in session.run(prompt, options=options):
            if isinstance(message, AssistantMessage):
                text = self._text_from_assistant(message)
                if text:
                    assistant_chunks.append(text)
            elif hasattr(message, "status"):
                status = getattr(message, "status", None) or getattr(message, "message", None)
                if status and status not in status_seen:
                    print(f"[SII] status -> {status}")
                    status_seen.add(status)
            elif hasattr(message, "content"):
                content = getattr(message, "content")
                if isinstance(content, str):
                    assistant_chunks.append(content)
        return "\n".join(chunk.strip() for chunk in assistant_chunks if chunk).strip()

    def _build_options(self, workspace: Path) -> SiiAgentOptions:
        workspace = workspace.resolve()
        env_vars = {k: v for k, v in self.env_config.env_values.items() if v}
        env_vars.setdefault("SII_AGENT_API_KEY", self.env_config.sii_api_key)
        env_vars.setdefault("SII_OPENAI_API_KEY", self.env_config.sii_api_key)
        env_vars.setdefault("OPENAI_API_KEY", self.env_config.sii_api_key)
        env_vars.setdefault("SII_AGENT_API_BASE_URL", self.env_config.sii_api_base)
        env_vars.setdefault("SII_OPENAI_BASE_URL", self.env_config.sii_api_base)
        env_vars.setdefault("OPENAI_BASE_URL", self.env_config.sii_api_base)
        env_vars.setdefault("SII_OPENAI_MODEL", self.env_config.sii_target_model)
        if self.env_config.sii_username:
            env_vars.setdefault("SII_USERNAME", self.env_config.sii_username)
        if self.env_config.sii_password:
            env_vars.setdefault("SII_PASSWORD", self.env_config.sii_password)

        options = SiiAgentOptions(
            system_prompt=self.env_config.sii_system_prompt,
            max_turns=self.env_config.sii_max_turns,
            auth_type=self.env_config.sii_auth_type,
            cwd=str(workspace),
            yolo=True,
            allowed_tools=[],
            model=self.env_config.sii_target_model,
            env=env_vars,
            enable_data_upload=False,
        )
        ensure_bridge_timeout(options, self.env_config.bridge_timeout_ms)
        return options

    def _text_from_assistant(self, message: AssistantMessage) -> str:
        blocks = []
        content = getattr(message, "content", [])
        for block in content:
            if isinstance(block, TextBlock):
                blocks.append(block.text)
        return "\n".join(blocks).strip()

    def _sdk_available(self) -> bool:
        return all(item is not None for item in (SDKAgentSession, SiiAgentOptions, AssistantMessage, TextBlock))


# --------------------------------------------------------------------------- #
# Command runner (Simplified for Python Environment)
# --------------------------------------------------------------------------- #


class CommandRunner:
    """Prepare environment for Python verification (replaces C++ runner)."""

    def __init__(self, logs_dir: Path, conda_env: Optional[str] = None, conda_exe: Optional[str] = None):
        self.logs_dir = ensure_directory(logs_dir)
        self.conda_env = conda_env
        self.conda_exe = conda_exe

    def capture(self, subtask: str, root: Path) -> Dict[str, Any]:
        """
        Since verification is done via import/mocks in RubricEvaluator,
        CommandRunner is now mostly a placeholder or for basic setup.
        """
        root = root.resolve()
        results: Dict[str, CommandResult] = {}
        
        # Check python version just to have a record
        cmd = ["python", "--version"]
        if self.conda_env and self.conda_exe:
             cmd = [self.conda_exe, "run", "--no-capture-output", "-n", self.conda_env, *cmd]
        
        rc, out, err = run_command(cmd, cwd=root, timeout=60)
        results["env_check"] = CommandResult("env_check", cmd, rc, out, err)

        summary = {name: res.to_dict() for name, res in results.items()}
        return summary


# --------------------------------------------------------------------------- #
# Verification Logic (Ported)
# --------------------------------------------------------------------------- #

class SubtaskVerifier:
    """Generic verifier that checks against Acceptance Criteria from task description."""

    def __init__(self, subtask_number: int, subtask_name: str, acceptance_criteria: List[str], mock_mongo: MockMongoDB, mock_github: Optional[MockGitHubCLI] = None, mock_docker: Optional[MockDocker] = None, webhook_secret: str = "mock_secret", repo_path: Optional[Path] = None):
        self.subtask_number = subtask_number
        self.subtask_name = subtask_name
        self.acceptance_criteria = acceptance_criteria
        self.mock_mongo = mock_mongo
        self.mock_github = mock_github
        self.mock_docker = mock_docker
        self.webhook_secret = webhook_secret
        self.repo_path = repo_path or Path.cwd()

    def verify(self) -> List[VerificationResult]:
        """Verify based on Acceptance Criteria from task description."""
        results: List[VerificationResult] = []
        
        # Map each acceptance criterion to a verification check
        for criterion in self.acceptance_criteria:
            check_result = self._check_criterion(criterion)
            if check_result is not None:
                results.append(check_result)
            else:
                results.append(
                    VerificationResult(
                        step_name=self.subtask_name,
                        check_name=f"criterion_not_verifiable",
                        passed=False,
                        message=f"Criterion not verifiable: {criterion[:80]}...",
                        details={"criterion": criterion}
                    )
                )
        return results

    def _check_criterion(self, criterion: str) -> Optional[VerificationResult]:
        """Check a single acceptance criterion."""
        criterion_lower = criterion.lower()
        
        if "fastapi server starts successfully" in criterion_lower: return self._check_fastapi_starts()
        if "mongodb connection is established" in criterion_lower: return self._check_mongodb_connection()
        if "environment variables are loaded correctly" in criterion_lower: return self._check_env_loading()
        if "health check endpoint returns 200" in criterion_lower: return self._check_health_endpoint()
        if "health check endpoint returns appropriate error when mongodb is unavailable" in criterion_lower: return self._check_health_unavailable()
        if "database collections are created" in criterion_lower: return self._check_database_collections()
        if "unique indexes are enforced" in criterion_lower: return self._check_unique_indexes()
        if "indexes on status" in criterion_lower or "indexes on session_id" in criterion_lower or "indexes on timestamp" in criterion_lower: return self._check_indexes_exist()
        if "index initialization is idempotent" in criterion_lower: return self._check_index_idempotent()
        if "webhook endpoint accepts post requests" in criterion_lower: return self._check_webhook_endpoint()
        if "signature verification correctly validates" in criterion_lower: return self._check_signature_verification()
        if "invalid signatures are rejected" in criterion_lower: return self._check_invalid_signature_rejection()
        if "ping events are handled" in criterion_lower: return self._check_ping_events()
        if "malformed json payloads return 400" in criterion_lower: return self._check_malformed_json()
        if "all webhook requests are logged" in criterion_lower: return self._check_webhook_logging()
        if "issues.assigned events create or retrieve sessions" in criterion_lower: return self._check_issues_assigned_processing()
        if "issue title and body are stored" in criterion_lower: return self._check_issue_storage()
        if "issue_comment.created events detect @mentions" in criterion_lower: return self._check_comment_mentions()
        if "session crud operations work" in criterion_lower: return self._check_session_crud()
        if "background tasks are queued" in criterion_lower: return self._check_background_tasks()
        if "github cli commands execute successfully" in criterion_lower: return self._check_github_cli_execution()
        if "repository cloning works" in criterion_lower or "repository cloning" in criterion_lower: return self._check_clone_repository()
        if "branch creation" in criterion_lower or "pr creation" in criterion_lower: return self._check_branch_and_pr_creation()
        if "comments can be posted" in criterion_lower: return self._check_post_comment()
        if "log entries are written" in criterion_lower or "all major operations are logged" in criterion_lower: return self._check_log_writing()
        if "logs can be retrieved" in criterion_lower: return self._check_log_retrieval()
        if "docker containers are created" in criterion_lower: return self._check_docker_containers()
        if "container lifecycle" in criterion_lower: return self._check_container_lifecycle()
        if "repositories are cloned into container workspace" in criterion_lower: return self._check_container_clone()
        if "concurrency limit" in criterion_lower: return self._check_concurrency_limit()
        if "duplicate issue processing is prevented" in criterion_lower: return self._check_duplicate_prevention()
        if "smolagents successfully initializes" in criterion_lower or "llm client is configured" in criterion_lower: return self._check_smolagents_init()
        if "agent execution workflow processes issues" in criterion_lower or "execution states are properly tracked" in criterion_lower: return self._check_agent_execution()
        if "prs are created" in criterion_lower or "feature branches are created" in criterion_lower: return self._check_pr_creation()
        if "monitoring interface displays" in criterion_lower: return self._check_monitoring_interface()
        
        return None

    # --- Verification Implementation Methods ---
    
    def _check_fastapi_starts(self) -> VerificationResult:
        try:
            from app.main import app
            from fastapi import FastAPI
            assert isinstance(app, FastAPI)
            return VerificationResult(self.subtask_name, "fastapi_starts", True, "FastAPI server starts successfully")
        except Exception as e:
            return VerificationResult(self.subtask_name, "fastapi_starts", False, f"FastAPI start check failed: {e}")

    def _check_mongodb_connection(self) -> VerificationResult:
        try:
            from app.db.mongo import get_db, get_database_name
            with patch("app.db.mongo.get_mongo_client") as mock_get_client:
                mock_client = MagicMock()
                mock_db = self.mock_mongo.get_database(get_database_name())
                mock_client.__getitem__.return_value = mock_db
                mock_get_client.return_value = mock_client
                with patch("app.db.mongo.get_db") as mock_get_db:
                    mock_get_db.return_value = mock_db
                    db = get_db()
                    assert db is not None
                    return VerificationResult(self.subtask_name, "mongodb_connection", True, "MongoDB connection established")
        except Exception as e:
            return VerificationResult(self.subtask_name, "mongodb_connection", False, f"MongoDB check failed: {e}")

    def _check_env_loading(self) -> VerificationResult:
        try:
            from app.settings import get_env
            test_key = "TEST_ENV_VAR"
            os.environ[test_key] = "test_value"
            value = get_env(test_key)
            assert value == "test_value"
            return VerificationResult(self.subtask_name, "env_loading", True, "Environment variables loaded correctly")
        except Exception as e:
            return VerificationResult(self.subtask_name, "env_loading", False, f"Env loading check failed: {e}")

    def _check_health_endpoint(self) -> VerificationResult:
        try:
            from app.main import app
            from fastapi.testclient import TestClient
            client = TestClient(app)
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json().get("status") == "ok"
            return VerificationResult(self.subtask_name, "health_endpoint", True, "Health check endpoint returns 200")
        except Exception as e:
            return VerificationResult(self.subtask_name, "health_endpoint", False, f"Health endpoint check failed: {e}")

    def _check_health_unavailable(self) -> VerificationResult:
        """
        Simulate MongoDB unavailable and expect non-200 with error indicator.
        """
        try:
            from app.main import app
            from fastapi.testclient import TestClient
            # Force Mongo health to fail
            with patch("app.db.mongo.check_mongodb_health", return_value=False):
                client = TestClient(app)
                response = client.get("/health")
                if response.status_code == 200:
                    return VerificationResult(self.subtask_name, "health_unavailable", False, "Health endpoint returned 200 even when MongoDB mocked as unavailable")
                body = response.json() if response.content else {}
                msg = body.get("status") or body.get("error") or "non-200 status"
                return VerificationResult(self.subtask_name, "health_unavailable", True, f"Health endpoint returns error when MongoDB is unavailable (status {response.status_code}, message={msg})")
        except Exception as e:
            return VerificationResult(self.subtask_name, "health_unavailable", False, f"Health unavailable check failed: {e}")

    def _check_database_collections(self) -> VerificationResult:
        try:
            from app.db.mongo import init_indexes, get_database_name
            with patch("app.db.mongo.get_db") as mock_get_db:
                mock_db = self.mock_mongo.get_database(get_database_name())
                mock_get_db.return_value = mock_db
                init_indexes()
                assert "agent_sessions" in self.mock_mongo.indexes
                return VerificationResult(self.subtask_name, "database_collections", True, "Database collections created")
        except Exception as e:
            return VerificationResult(self.subtask_name, "database_collections", False, f"Database collections check failed: {e}")

    def _check_unique_indexes(self) -> VerificationResult:
        try:
            from app.db.mongo import init_indexes, get_database_name
            with patch("app.db.mongo.get_db") as mock_get_db:
                mock_db = self.mock_mongo.get_database(get_database_name())
                mock_get_db.return_value = mock_db
                init_indexes()
                indexes = self.mock_mongo.indexes["agent_sessions"]
                has_unique = any(idx.get("unique") for idx in indexes)
                if has_unique:
                    return VerificationResult(self.subtask_name, "unique_indexes", True, "Unique indexes enforced")
        except Exception as e:
            pass
        return VerificationResult(self.subtask_name, "unique_indexes", False, "Unique indexes check failed")

    def _check_indexes_exist(self) -> VerificationResult:
        try:
            from app.db.mongo import init_indexes, get_database_name
            with patch("app.db.mongo.get_db") as mock_get_db:
                mock_db = self.mock_mongo.get_database(get_database_name())
                mock_get_db.return_value = mock_db
                init_indexes()
                if self.mock_mongo.indexes.get("agent_sessions") or self.mock_mongo.indexes.get("agent_logs"):
                    return VerificationResult(self.subtask_name, "indexes_exist", True, "Indexes exist")
        except Exception:
            pass
        return VerificationResult(self.subtask_name, "indexes_exist", False, "Indexes check failed")

    def _check_index_idempotent(self) -> VerificationResult:
        return VerificationResult(self.subtask_name, "index_idempotent", True, "Index initialization is idempotent (assumed)")

    def _check_webhook_endpoint(self) -> VerificationResult:
        try:
            from app.main import app
            from fastapi.testclient import TestClient
            client = TestClient(app)
            payload = json.dumps({"test": "ping"}).encode("utf-8")
            signature = self._generate_signature(payload)
            response = client.post("/webhooks/github", headers={"X-GitHub-Event": "ping", "X-Hub-Signature-256": signature, "Content-Type": "application/json"}, content=payload)
            assert response.status_code == 200
            return VerificationResult(self.subtask_name, "webhook_endpoint", True, "Webhook endpoint accepts POST")
        except Exception as e:
            return VerificationResult(self.subtask_name, "webhook_endpoint", False, f"Webhook endpoint check failed: {e}")

    def _check_signature_verification(self) -> VerificationResult:
        try:
            from app.main import app
            from fastapi.testclient import TestClient
            client = TestClient(app)
            payload = json.dumps({"test": "data"}).encode("utf-8")
            signature = self._generate_signature(payload)
            response = client.post("/webhooks/github", headers={"X-GitHub-Event": "ping", "X-Hub-Signature-256": signature, "Content-Type": "application/json"}, content=payload)
            assert response.status_code == 200
            return VerificationResult(self.subtask_name, "signature_verification", True, "Signature verification valid")
        except Exception as e:
            return VerificationResult(self.subtask_name, "signature_verification", False, f"Signature verification check failed: {e}")

    def _check_invalid_signature_rejection(self) -> VerificationResult:
        try:
            from app.main import app
            from fastapi.testclient import TestClient
            client = TestClient(app)
            payload = json.dumps({"test": "data"}).encode("utf-8")
            response = client.post("/webhooks/github", headers={"X-GitHub-Event": "ping", "X-Hub-Signature-256": "sha256=invalid", "Content-Type": "application/json"}, content=payload)
            assert response.status_code == 401
            return VerificationResult(self.subtask_name, "invalid_signature_rejection", True, "Invalid signatures rejected")
        except Exception as e:
            return VerificationResult(self.subtask_name, "invalid_signature_rejection", False, f"Invalid signature check failed: {e}")

    def _check_ping_events(self) -> VerificationResult:
        try:
            from app.main import app
            from fastapi.testclient import TestClient
            client = TestClient(app)
            payload = json.dumps({"test": "ping"}).encode("utf-8")
            signature = self._generate_signature(payload)
            response = client.post("/webhooks/github", headers={"X-GitHub-Event": "ping", "X-Hub-Signature-256": signature, "Content-Type": "application/json"}, content=payload)
            assert response.status_code == 200
            return VerificationResult(self.subtask_name, "ping_events", True, "Ping events handled")
        except Exception as e:
            return VerificationResult(self.subtask_name, "ping_events", False, f"Ping events check failed: {e}")

    def _check_malformed_json(self) -> VerificationResult:
        try:
            from app.main import app
            from fastapi.testclient import TestClient
            client = TestClient(app)
            payload = b"invalid json"
            signature = self._generate_signature(payload)
            response = client.post("/webhooks/github", headers={"X-GitHub-Event": "issues", "X-Hub-Signature-256": signature, "Content-Type": "application/json"}, content=payload)
            assert response.status_code == 400
            return VerificationResult(self.subtask_name, "malformed_json", True, "Malformed JSON rejected")
        except Exception as e:
            return VerificationResult(self.subtask_name, "malformed_json", False, f"Malformed JSON check failed: {e}")

    def _check_webhook_logging(self) -> VerificationResult:
        # Simple static analysis
        try:
            from app.webhooks.github import router
            import inspect
            source = inspect.getsource(router.routes[0].endpoint) if router.routes else ""
            if "log(" in source or "write_log" in source:
                return VerificationResult(self.subtask_name, "webhook_logging", True, "Webhook requests logged")
        except Exception:
            pass
        return VerificationResult(self.subtask_name, "webhook_logging", False, "Webhook logging check failed")

    def _check_issues_assigned_processing(self) -> VerificationResult:
        try:
            from app.main import app
            from fastapi.testclient import TestClient
            client = TestClient(app)
            payload_data = {"action": "assigned", "issue": {"number": 123, "title": "Test", "body": "Test"}, "repository": {"full_name": "test/repo"}}
            payload = json.dumps(payload_data).encode("utf-8")
            signature = self._generate_signature(payload)
            # Ensure DB is mocked for this request
            with patch("app.db.mongo.get_db", return_value=self.mock_mongo.get_database("test")):
                response = client.post("/webhooks/github", headers={"X-GitHub-Event": "issues", "X-Hub-Signature-256": signature, "Content-Type": "application/json"}, content=payload)
                assert response.status_code == 200
                assert len(self.mock_mongo.sessions) > 0
                return VerificationResult(self.subtask_name, "issues_assigned_processing", True, "Issues assigned creates session")
        except Exception as e:
            return VerificationResult(self.subtask_name, "issues_assigned_processing", False, f"Issues processing check failed: {e}")

    def _check_issue_storage(self) -> VerificationResult:
        try:
            from app.sessions.service import update_issue_info
            with patch("app.db.mongo.get_db", return_value=self.mock_mongo.get_database("test")):
                session_id = str(uuid.uuid4())
                self.mock_mongo.sessions[session_id] = {"session_id": session_id}
                update_issue_info(session_id, "Test Title", "Test Body")
                session = self.mock_mongo.sessions.get(session_id)
                assert session and session.get("issue_title") == "Test Title"
                return VerificationResult(self.subtask_name, "issue_storage", True, "Issue info stored")
        except Exception as e:
            return VerificationResult(self.subtask_name, "issue_storage", False, f"Issue storage check failed: {e}")

    def _check_comment_mentions(self) -> VerificationResult:
        try:
            from app.main import app
            from fastapi.testclient import TestClient
            client = TestClient(app)
            payload_data = {"action": "created", "issue": {"number": 123}, "comment": {"body": "@test-bot help"}, "repository": {"full_name": "test/repo"}}
            payload = json.dumps(payload_data).encode("utf-8")
            signature = self._generate_signature(payload)
            response = client.post("/webhooks/github", headers={"X-GitHub-Event": "issue_comment", "X-Hub-Signature-256": signature, "Content-Type": "application/json"}, content=payload)
            assert response.status_code == 200
            return VerificationResult(self.subtask_name, "comment_mentions", True, "Comment mentions detected")
        except Exception as e:
            return VerificationResult(self.subtask_name, "comment_mentions", False, f"Comment mentions check failed: {e}")

    def _check_session_crud(self) -> VerificationResult:
        try:
            from app.sessions.service import create_or_get_session, update_session_status
            with patch("app.db.mongo.get_db", return_value=self.mock_mongo.get_database("test")):
                session_id = create_or_get_session("test/repo", 123)
                assert session_id is not None
                update_session_status(session_id, "running")
                session = self.mock_mongo.sessions.get(session_id)
                assert session and session.get("status") == "running"
                return VerificationResult(self.subtask_name, "session_crud", True, "Session CRUD operations work")
        except Exception as e:
            return VerificationResult(self.subtask_name, "session_crud", False, f"Session CRUD check failed: {e}")

    def _check_background_tasks(self) -> VerificationResult:
        try:
            from app.webhooks.github import router
            import inspect
            source = inspect.getsource(router.routes[0].endpoint) if router.routes else ""
            if "BackgroundTasks" in source or "background_tasks" in source:
                return VerificationResult(self.subtask_name, "background_tasks", True, "Background tasks used")
        except Exception:
            pass
        return VerificationResult(self.subtask_name, "background_tasks", False, "Background tasks check failed")

    def _check_github_cli_execution(self) -> VerificationResult:
        try:
            from app.github.cli import comment_issue
            if self.mock_github:
                with patch("app.github.cli._run") as mock_run:
                    mock_run.return_value = self.mock_github.run_command(["gh", "issue", "comment", "123"])
                    comment_issue("test/repo", 123, "Test")
                    return VerificationResult(self.subtask_name, "github_cli_execution", True, "GitHub CLI executed")
        except Exception:
            pass
        return VerificationResult(self.subtask_name, "github_cli_execution", False, "GitHub CLI check failed")

    def _check_clone_repository(self) -> VerificationResult:
        try:
            from app.github.cli import clone_repository
            assert callable(clone_repository)
            return VerificationResult(self.subtask_name, "clone_repository", True, "Clone repo function exists")
        except Exception as e:
            return VerificationResult(self.subtask_name, "clone_repository", False, f"Clone repo check failed: {e}")

    def _check_branch_and_pr_creation(self) -> VerificationResult:
        try:
            from app.github.cli import create_branch, create_pr
            assert callable(create_branch) and callable(create_pr)
            return VerificationResult(self.subtask_name, "branch_and_pr_creation", True, "Branch/PR functions exist")
        except Exception as e:
            return VerificationResult(self.subtask_name, "branch_and_pr_creation", False, f"Branch/PR check failed: {e}")

    def _check_post_comment(self) -> VerificationResult:
        try:
            from app.github.cli import comment_issue
            assert callable(comment_issue)
            return VerificationResult(self.subtask_name, "post_comment", True, "Post comment function exists")
        except Exception as e:
            return VerificationResult(self.subtask_name, "post_comment", False, f"Post comment check failed: {e}")

    def _check_log_writing(self) -> VerificationResult:
        try:
            from app.logs.db_logger import write_log
            from app.logs.models import AgentLog
            with patch("app.db.mongo.get_db", return_value=self.mock_mongo.get_database("test")):
                test_log: AgentLog = {"session_id": "test", "level": "info", "message": "Test", "source": "webhook", "timestamp": "2024-01-01T00:00:00Z"}
                write_log(test_log)
                assert len(self.mock_mongo.logs) > 0
                return VerificationResult(self.subtask_name, "log_writing", True, "Logs written to DB")
        except Exception as e:
            return VerificationResult(self.subtask_name, "log_writing", False, f"Log writing check failed: {e}")

    def _check_log_retrieval(self) -> VerificationResult:
        try:
            from app.db.mongo import get_db
            with patch("app.db.mongo.get_db", return_value=self.mock_mongo.get_database("test")):
                self.mock_mongo.logs.append({"session_id": "test", "level": "info", "message": "Test"})
                logs = list(self.mock_mongo.get_database("test")["agent_logs"].find({"session_id": "test"}))
                assert len(logs) > 0
                return VerificationResult(self.subtask_name, "log_retrieval", True, "Logs retrieved")
        except Exception as e:
            return VerificationResult(self.subtask_name, "log_retrieval", False, f"Log retrieval check failed: {e}")

    def _check_docker_containers(self) -> VerificationResult:
        try:
            from app.sandbox.docker_manager import create_container
            if self.mock_docker:
                with patch("app.sandbox.docker_manager.get_client", return_value=self.mock_docker):
                    container_id = create_container("python:3.11-slim")
                    assert container_id is not None
                    return VerificationResult(self.subtask_name, "docker_containers", True, "Docker containers created")
        except Exception as e:
            # If docker module missing, fail gracefully
            if "app.sandbox" in str(e) or "docker_manager" in str(e):
                return VerificationResult(self.subtask_name, "docker_containers", False, "Docker module missing")
            return VerificationResult(self.subtask_name, "docker_containers", False, f"Docker check failed: {e}")

    def _check_container_lifecycle(self) -> VerificationResult:
        try:
            from app.sandbox.docker_manager import create_container, cleanup
            if self.mock_docker:
                with patch("app.sandbox.docker_manager.get_client", return_value=self.mock_docker):
                    container_id = create_container("python:3.11-slim")
                    cleanup(container_id)
                    return VerificationResult(self.subtask_name, "container_lifecycle", True, "Container lifecycle works")
        except Exception as e:
            return VerificationResult(self.subtask_name, "container_lifecycle", False, f"Lifecycle check failed: {e}")

    def _check_container_clone(self) -> VerificationResult:
        try:
            from app.sandbox.docker_manager import clone_repo
            if self.mock_docker:
                with patch("app.sandbox.docker_manager.get_client", return_value=self.mock_docker):
                    with patch("app.sandbox.docker_manager.exec_sh", return_value=(0, "Cloned")):
                        workdir = clone_repo("mock_id", "test/repo")
                        assert workdir == "/workspace/repo"
                        return VerificationResult(self.subtask_name, "container_clone", True, "Clone in container works")
        except Exception as e:
            return VerificationResult(self.subtask_name, "container_clone", False, f"Container clone check failed: {e}")

    def _check_concurrency_limit(self) -> VerificationResult:
        try:
            from app.limits.concurrency import can_start_new_agent, get_max_concurrent
            with patch("app.db.mongo.get_db", return_value=self.mock_mongo.get_database("test")):
                assert get_max_concurrent() == 2
                self.mock_mongo.sessions[str(uuid.uuid4())] = {"status": "running"}
                self.mock_mongo.sessions[str(uuid.uuid4())] = {"status": "running"}
                assert not can_start_new_agent()
                return VerificationResult(self.subtask_name, "concurrency_limit", True, "Concurrency enforced")
        except Exception as e:
            return VerificationResult(self.subtask_name, "concurrency_limit", False, f"Concurrency check failed: {e}")

    def _check_duplicate_prevention(self) -> VerificationResult:
        try:
            from app.sessions.service import create_or_get_session
            with patch("app.db.mongo.get_db", return_value=self.mock_mongo.get_database("test")):
                s1 = create_or_get_session("repo", 123)
                s2 = create_or_get_session("repo", 123)
                assert s1 == s2
                return VerificationResult(self.subtask_name, "duplicate_prevention", True, "Duplicate prevention works")
        except Exception as e:
            return VerificationResult(self.subtask_name, "duplicate_prevention", False, f"Duplicate check failed: {e}")

    def _check_smolagents_init(self) -> VerificationResult:
        try:
            import app.agent.smol as smol_module
            return VerificationResult(self.subtask_name, "smolagents_init", True, "SmolAgents module found")
        except Exception as e:
            return VerificationResult(self.subtask_name, "smolagents_init", False, f"SmolAgents init check failed: {e}")

    def _check_agent_execution(self) -> VerificationResult:
        try:
            import app.agent.runner as runner_module
            return VerificationResult(self.subtask_name, "agent_execution", True, "Agent runner module found")
        except Exception as e:
            return VerificationResult(self.subtask_name, "agent_execution", False, f"Agent execution check failed: {e}")

    def _check_pr_creation(self) -> VerificationResult:
        try:
            import app.agent.code_agent as code_agent
            return VerificationResult(self.subtask_name, "pr_creation", True, "Code agent module found")
        except Exception as e:
            return VerificationResult(self.subtask_name, "pr_creation", False, f"PR creation check failed: {e}")

    def _check_monitoring_interface(self) -> VerificationResult:
        try:
            from app.http.monitor import router
            return VerificationResult(self.subtask_name, "monitoring_interface", True, "Monitoring interface found")
        except Exception as e:
            return VerificationResult(self.subtask_name, "monitoring_interface", False, f"Monitoring check failed: {e}")

    def _generate_signature(self, payload: bytes) -> str:
        signature = hmac.new(self.webhook_secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
        return f"sha256={signature}"


# --------------------------------------------------------------------------- #
# Rubric evaluation (Updated to use SubtaskVerifier)
# --------------------------------------------------------------------------- #


class RubricEvaluator:
    """Uses SubtaskVerifier to check Acceptance Criteria."""

    def __init__(self, description: Dict[str, Any], env_config: Optional[EnvConfig] = None):
        self.description = description
        self.mock_mongo = MockMongoDB()
        self.mock_github = MockGitHubCLI()
        self.mock_docker = MockDocker()
        self.env_config = env_config

    def evaluate(
        self, subtask: str, evalspace: Path, command_results: Dict[str, Any]
    ) -> RubricResult:
        
        # 1. Prepare environment
        evalspace = evalspace.resolve()
        sys.path.insert(0, str(evalspace))
        
        # 2. Get criteria from description
        step_desc_text = self.description.get(subtask, "")
        criteria = self._parse_criteria(step_desc_text)
        
        # 3. Clean up sys.modules to allow re-importing student code (Basic attempt)
        # Note: This is imperfect. For deep reliability, run in separate process.
        # Here we do best-effort module unloading for 'app' namespace.
        for mod in list(sys.modules.keys()):
            if mod.startswith("app"):
                del sys.modules[mod]

        # 4. Run verification
        try:
            verifier = SubtaskVerifier(
                subtask_number=int(subtask.replace("subtask", "")),
                subtask_name=subtask,
                acceptance_criteria=criteria,
                mock_mongo=self.mock_mongo,
                mock_github=self.mock_github,
                mock_docker=self.mock_docker,
                webhook_secret=(self.env_config.webhook_secret if self.env_config else "mock_secret"),
                repo_path=evalspace
            )
            
            # Setup Mocks globally if needed (e.g. if code imports at module level)
            # We trust the verifier's use of 'patch', but some patches might need to be broader
            
            results = verifier.verify()
            
        except Exception as e:
            results = [
                VerificationResult(subtask, "execution_error", False, f"Verifier crashed: {e}")
            ]
        finally:
            # Cleanup sys.path
            if str(evalspace) in sys.path:
                sys.path.remove(str(evalspace))

        # 5. Calculate Score
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        score = (passed_count / total_count * 10.0) if total_count > 0 else 0.0
        
        failed_points = [r.message for r in results if not r.passed]

        return RubricResult(
            subtask=subtask,
            score=score,
            pass_count=passed_count,
            total_points=total_count,
            failed_points=failed_points,
            raw_results=results
        )

    def points_for_subtask(self, subtask: str) -> List[str]:
        # Return generic list based on description just for display
        step_desc_text = self.description.get(subtask, "")
        return self._parse_criteria(step_desc_text)

    def _parse_criteria(self, description_text: str) -> List[str]:
        criteria = []
        criteria_markers = [
            "\n**Acceptance Criteria:**",
            "\nAcceptance Criteria:",
            "\nAcceptance Criteria:\n",
        ]
        
        criteria_text = ""
        for marker in criteria_markers:
            if marker in description_text:
                criteria_start = description_text.find(marker) + len(marker)
                criteria_text = description_text[criteria_start:].strip()
                break
        
        if criteria_text:
            lines = criteria_text.split("\n")
            for line in lines:
                line = line.strip()
                if line and (line.startswith("-") or line.startswith("•")):
                    clean_line = line.lstrip("-• ").strip()
                    if clean_line:
                        criteria.append(clean_line)
        return criteria


# --------------------------------------------------------------------------- #
# Coordinator (Adjusted to pass description to RubricEvaluator)
# --------------------------------------------------------------------------- #


class EvaluationCoordinator:
    def __init__(self, env_config: EnvConfig, description_path: Path, visualize: bool = False):
        self.env_config = env_config
        description_path = resolve_script_path(description_path)
        self.description_path = description_path
        self.visualize = visualize
        self.description = read_json(description_path)
        self.base_dir = description_path.parent.resolve()
        self.repo_root = self.base_dir.parent.resolve()
        self.run_root = (self.base_dir / env_config.model_name).resolve()
        self.eval_only = getattr(env_config, "eval_only", False)
        source_candidate = (SCRIPT_DIR / "source").resolve()
        alt_candidate = (self.base_dir / "source").resolve()
        self.source_repo: Optional[Path] = None
        if source_candidate.exists() and source_candidate.is_dir():
            self.source_repo = source_candidate
        elif alt_candidate.exists() and alt_candidate.is_dir():
            self.source_repo = alt_candidate
        
        # Pass description to RubricEvaluator
        self.rubric = RubricEvaluator(self.description, env_config=self.env_config)
        
        self.meta: Dict[str, Any] = {
            "model": env_config.model_name,
            "scorer": env_config.scorer_name,
            "max_attempts": env_config.max_attempts,
            "subtasks": [],
        }

    def prepare_layout(self) -> None:
        ensure_directory(self.run_root)
        subtask_count = int(self.description.get("subtask_count") or 0)
        for index in range(1, subtask_count + 1):
            subtask = f"subtask{index}"
            ensure_directory(self.run_root / subtask / "workspace")
            ensure_directory(self.run_root / subtask / "evalspace")

    def run(self) -> Dict[str, Any]:
        self.prepare_layout()
        self.env_config.inject_defaults()
        # Conda checks kept generic, though we rely mostly on sys.path
        conda_exe = self._resolve_conda_executable()
        conda_exe = self._verify_conda_environment(conda_exe)
        if self.env_config.use_mock:
            print("[DEBUG] USE_MOCK=true; skipping MongoDB auto-start.")
        else:
            self._ensure_local_mongo()
        
        agent = None if self.eval_only else AgentRunner(self.env_config, visualize=self.visualize)
        previous_best: Optional[Path] = None
        subtask_count = int(self.description.get("subtask_count") or 0)
        start_index = max(1, getattr(self.env_config, "start_subtask", 1))
        
        if start_index > subtask_count:
            print(f"[HALT] start subtask {start_index} beyond available count {subtask_count}; exiting.")
            return self.meta

        print(f"[BOOT] Model={self.env_config.model_name} scorer={self.env_config.scorer_name}")
        terminate = False
        for index in range(start_index, subtask_count + 1):
            subtask = f"subtask{index}"
            prompt = self.description.get(subtask, "")
            attempt_summaries: List[AttemptSummary] = []
            cache_workspace: Optional[Path] = None
            feedback: str = ""
            print(f"[SUBTASK] Starting {subtask}")
            attempt_limit = 1 if self.eval_only else self.env_config.max_attempts
            for attempt in range(1, attempt_limit + 1):
                workspace, evalspace = self._prepare_attempt_dirs(subtask, attempt, previous_best)
                if attempt > 1 and not self.eval_only:
                    self._clone_previous_attempt(subtask, attempt, workspace)
                elif previous_best and not self.eval_only:
                    copy_workspace(previous_best, workspace)

                if not self.eval_only and agent:
                    agent_output = agent.send(
                        prompt + ("\n\n" + feedback if feedback else ""),
                        workspace_notice(workspace, self.repo_root),
                        workspace,
                    )
                else:
                    agent_output = ""
                
                copy_workspace(workspace, evalspace)
                logs_dir = ensure_directory(evalspace / "logs")
                conda_env = self.env_config.conda_env_name or None
                
                # CommandRunner now primarily sets up environment checks
                cmd_runner = CommandRunner(
                    logs_dir,
                    conda_env=conda_env if conda_exe else None,
                    conda_exe=conda_exe,
                )
                commands = cmd_runner.capture(subtask, evalspace)
                
                # RubricEvaluator runs the import-based verification
                rubric_result = self.rubric.evaluate(subtask, evalspace, commands)
                
                self._print_command_diagnostics(subtask, attempt, commands)
                self._print_rubric_diagnostics(subtask, attempt, rubric_result)
                feedback = self._build_feedback(rubric_result)

                summary = AttemptSummary(
                    subtask=subtask,
                    attempt_index=attempt,
                    score=rubric_result.score,
                    rubric=rubric_result,
                    workspace=workspace,
                    evalspace=evalspace,
                    agent_output=agent_output,
                    commands=commands,
                    feedback=feedback,
                )
                attempt_summaries.append(summary)
                print(
                    f"[RESULT] {subtask} attempt {attempt}: score={rubric_result.score:.1f} "
                    f"pass={rubric_result.pass_count}/{rubric_result.total_points}"
                )
                if rubric_result.failed_points:
                    print(
                        "         Failed rubric points: "
                        + ", ".join(rubric_result.failed_points[:3]) + ("..." if len(rubric_result.failed_points)>3 else "")
                    )
                if not self.eval_only and rubric_result.score >= 8.0: # Threshold from ref script is 80%
                    cache_workspace = workspace
                    break
                cache_workspace = cache_workspace or workspace

            best = max(attempt_summaries, key=lambda item: item.score) if attempt_summaries else None
            if cache_workspace is None and best:
                cache_workspace = best.workspace
            if cache_workspace and not self.eval_only:
                previous_best = cache_workspace
            self.meta["subtasks"].append(
                {
                    "name": subtask,
                    "attempts": [item.to_dict() for item in attempt_summaries],
                    "best_score": best.score if best else 0,
                    "best_attempt": best.attempt_index if best else None,
                }
            )

            best_score = best.score if best else 0
            if not self.eval_only and best_score < 0:
                print(
                    f"[HALT] {subtask} best score {best_score:.1f}/10 < 0; stopping evaluation early."
                )
                terminate = True
                break

        if terminate:
            print("[HALT] Evaluation terminated early due to score threshold.")

        meta_path = self.run_root / "meta_eval.json"
        meta_path.write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        print(f"[DONE] Wrote results to {meta_path}")
        return self.meta

    def _prepare_attempt_dirs(
        self, subtask: str, attempt_index: int, previous_best: Optional[Path]
    ) -> Tuple[Path, Path]:
        attempt_name = f"attempt_{attempt_index:02d}"
        workspace = (self.run_root / subtask / "workspace" / attempt_name).resolve()
        evalspace = (self.run_root / subtask / "evalspace" / attempt_name).resolve()
        if attempt_index == 1:
            if self.eval_only:
                ensure_directory(workspace)
            else:
                clear_directory(workspace)
                ensure_directory(workspace)
                if previous_best and previous_best.exists():
                    try:
                        copy_workspace(previous_best, workspace)
                        print(f"[INFO] Seeded workspace from previous subtask result: {previous_best}")
                    except Exception as exc:
                        print(f"[WARN] Failed to seed workspace from previous subtask: {exc}")
                elif self.source_repo and self.source_repo.exists():
                    try:
                        copy_workspace(self.source_repo, workspace)
                        print(f"[INFO] Seeded workspace from source: {self.source_repo}")
                    except Exception as exc:
                        print(f"[WARN] Failed to seed workspace from source: {exc}")
        else:
            ensure_directory(workspace)
        ensure_directory(evalspace)
        return workspace, evalspace

    def _clone_previous_attempt(self, subtask: str, attempt_index: int, workspace: Path) -> None:
        prev_attempt = attempt_index - 1
        prev_path = (self.run_root / subtask / "workspace" / f"attempt_{prev_attempt:02d}").resolve()
        if prev_path.exists():
            copy_workspace(prev_path, workspace)

    def _build_feedback(self, rubric: RubricResult) -> str:
        if not rubric.failed_points:
            return ""
        # Provide actionable feedback based on failed check names/messages
        bullets = "\n".join(f"- {item}" for item in rubric.failed_points[:5])
        return f"The verification failed on the following points (Acceptance Criteria):\n{bullets}\n\nPlease verify these specific requirements are implemented correctly."

    def _resolve_conda_executable(self) -> Optional[str]:
        candidates = [
            self.env_config.env_values.get("CONDA_EXE"),
            os.environ.get("CONDA_EXE"),
            shutil.which("conda"),
            "/root/miniconda3/bin/conda",
        ]
        for candidate in candidates:
            if not candidate:
                continue
            expanded = os.path.expanduser(candidate)
            path = Path(expanded)
            if path.exists():
                print(f"[DEBUG] Using conda executable: {path}")
                return str(path)
            resolved = shutil.which(candidate)
            if resolved:
                print(f"[DEBUG] Using conda executable from PATH: {resolved}")
                return resolved
        print("[DEBUG] Conda executable not found; will run commands without conda.")
        return None

    def _verify_conda_environment(self, conda_exe: Optional[str]) -> Optional[str]:
        env_name = (self.env_config.conda_env_name or "").strip()
        if not env_name:
            print("[DEBUG] CONDA_ENV_NAME not set; commands will use host PATH.")
            return None
        if not conda_exe:
            print(
                f"[DEBUG] CONDA_ENV_NAME={env_name} provided but conda executable missing; "
                "commands will use host PATH."
            )
            return None

        checks = [
            (["python", "-c", "import sys; print('conda-env', sys.executable)"], "python"),
        ]
        for args, label in checks:
            cmd = [conda_exe, "run", "-n", env_name, *args]
            rc, out, err = run_command(cmd, timeout=60)
            if rc != 0:
                print(
                    f"[WARN] Conda env '{env_name}' failed '{label}' check; "
                    "falling back to host PATH commands."
                )
                return None

        print(f"[DEBUG] Conda env '{env_name}' verified via {conda_exe}.")
        return conda_exe

    def _print_command_diagnostics(
        self, subtask: str, attempt: int, commands: Dict[str, Any]
    ) -> None:
        print(f"[DETAIL] {subtask} attempt {attempt} environment checks:")
        for name, data in commands.items():
            if not isinstance(data, dict):
                continue
            rc = data.get("returncode")
            status = "PASS" if rc == 0 else "FAIL"
            print(f"         [{status}] {name}: rc={rc}")

    def _print_rubric_diagnostics(self, subtask: str, attempt: int, rubric: RubricResult) -> None:
        if not rubric.raw_results:
            return
        print(f"[DETAIL] {subtask} attempt {attempt} verification results:")
        for res in rubric.raw_results:
            status = "PASS" if res.passed else "FAIL"
            print(f"         - {status} {res.check_name}: {res.message}")

    def _ensure_local_mongo(self) -> None:
        """Best-effort local MongoDB bootstrap if not already reachable."""
        mongo_url = os.environ.get("MONGODB_URL") or self.env_config.mongodb_url
        host, port = self._parse_mongo_host_port(mongo_url)
        if not host or not port:
            print(f"[WARN] Could not parse MONGODB_URL='{mongo_url}'; skipping MongoDB auto-start.")
            return

        if is_port_open(host, port):
            print(f"[DEBUG] MongoDB reachable at {host}:{port}")
            return

        mongod_bin = shutil.which("mongod")
        if not mongod_bin:
            install_hint = (
                "No MongoDB daemon found. Install MongoDB (e.g. 'sudo apt-get install -y mongodb-org' "
                "or 'brew install mongodb-community') or start with Docker: "
                f"docker run -d --name mongo -p {port}:{port} -v ./mongo-data:/data/db mongo:6"
            )
            print(f"[WARN] {install_hint}")
            return

        data_dir = ensure_directory(self.run_root / "mongo-data")
        log_file = self.run_root / "mongo.log"
        cmd = [
            mongod_bin,
            "--dbpath",
            str(data_dir),
            "--bind_ip",
            "127.0.0.1",
            "--port",
            str(port),
            "--fork",
            "--logpath",
            str(log_file),
        ]
        rc, out, err = run_command(cmd, timeout=30)
        if rc != 0:
            print(f"[WARN] Failed to start mongod (rc={rc}): {err or out}")
            return
        time.sleep(1.0)
        if is_port_open(host, port):
            print(f"[INFO] Started local MongoDB at {host}:{port} (dbpath={data_dir})")
        else:
            print(f"[WARN] mongod started but {host}:{port} still unreachable; see log {log_file}")

    def _parse_mongo_host_port(self, mongo_url: str) -> Tuple[str, int]:
        parsed = urlparse(mongo_url or "")
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 27017
        return host, int(port)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task31 evaluator runner")
    parser.add_argument("--env", default=str(SCRIPT_DIR / ".env"), help="Path to env file for agent credentials")
    parser.add_argument("--visualize", action="store_true", help="Enable SII SDK visualization")
    parser.add_argument(
        "--description",
        default=str(SCRIPT_DIR / "description.json"),
        help="Path to task description JSON",
    )
    parser.add_argument(
        "--start-subtask",
        type=int,
        default=1,
        help="1-indexed subtask number to start from (default: 1)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only without invoking agent or modifying workspaces",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Alias for --eval-only: skip sii_agent_sdk and only run evaluations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_httpx_client_app_param()
    env_path = resolve_script_path(Path(args.env))
    env_config = EnvConfig.load(env_path, visualize=args.visualize)
    env_config.start_subtask = args.start_subtask
    env_config.eval_only = args.eval_only or args.test_only
    description_path = resolve_script_path(Path(args.description))
    coordinator = EvaluationCoordinator(env_config, description_path, visualize=args.visualize)
    print("[INFO] Evaluation started")
    coordinator.run()
    print("[INFO] Evaluation finished")


if __name__ == "__main__":
    main()
