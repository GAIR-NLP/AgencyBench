#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import textwrap
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
# Helpers
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


EXCLUDED_COPY_DIRS = {".git", ".hg", ".svn", "__pycache__", ".sii", "logs", "bin"}
EXCLUDED_COPY_FILES = {"meta_eval.json"}
EXCLUDED_COPY_SUFFIXES = {".db", ".db3", ".sqlite", ".sqlite3", ".txt"}


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
        # Skip hidden or excluded directories entirely
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
    """Run a shell command and capture stdout/stderr."""

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
        你必须在{target}中工作，禁止访问其他任何路径。请保持当前结构，不要修改其他子任务。
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


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class EnvConfig:
    env_path: Path
    visualize: bool
    env_values: Dict[str, str]
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
class RubricResult:
    subtask: str
    score: int
    pass_count: int
    total_points: int
    failed_points: List[str] = field(default_factory=list)

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
    score: int
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
# Command runner (compile/run Java)
# --------------------------------------------------------------------------- #


class CommandRunner:
    """Execute compile/run commands and persist outputs into logs."""

    def __init__(self, logs_dir: Path, conda_env: Optional[str] = None, conda_exe: Optional[str] = None):
        self.logs_dir = ensure_directory(logs_dir)
        self.conda_env = conda_env
        self.conda_exe = conda_exe
        self._warned_conda_missing = False

    def capture(self, subtask: str, root: Path) -> Dict[str, Any]:
        root = root.resolve()
        results: Dict[str, CommandResult] = {}
        compile_result = self.compile_java(root)
        results["compile"] = compile_result

        run_inputs = self._sample_inputs(subtask)
        for idx, input_text in enumerate(run_inputs, start=1):
            result = self.run_java(root, input_text, name=f"run_{idx}")
            results[result.name] = result

        planned_path = self.logs_dir / f"{subtask}_commands.txt"
        planned_path.write_text(
            "\n".join(run_inputs) if run_inputs else "No scripted inputs", encoding="utf-8"
        )
        self._persist(results)
        summary = {name: res.to_dict() for name, res in results.items()}
        summary["planned_inputs_file"] = str(planned_path)
        return summary

    def compile_java(self, root: Path) -> CommandResult:
        sources = [
            path
            for path in root.rglob("*.java")
            if "bin" not in path.parts and not any(part.startswith(".") for part in path.parts)
        ]
        if not sources:
            return CommandResult(
                name="compile",
                command=["javac"],
                returncode=1,
                stdout="",
                stderr="No Java sources discovered",
            )
        bin_dir = ensure_directory(root / "bin")
        cmd = self._wrap_cmd(["javac", "-d", str(bin_dir)] + [str(src) for src in sources])
        rc, out, err = run_command(cmd, cwd=root, timeout=600)
        return CommandResult(name="compile", command=cmd, returncode=rc, stdout=out, stderr=err)

    def run_java(self, root: Path, input_text: str, name: str = "run") -> CommandResult:
        bin_dir = root / "bin"
        cmd = self._wrap_cmd(["java", "-cp", str(bin_dir), "Main"])
        payload = input_text
        if not payload.endswith("\n"):
            payload += "\n"
        payload += "\n\n"  # extra blank lines keep Scanner.nextLine() from hitting EOF immediately
        rc, out, err = run_command(cmd, cwd=root, input_text=payload, timeout=600)
        return CommandResult(name=name, command=cmd, returncode=rc, stdout=out, stderr=err)

    def _sample_inputs(self, subtask: str) -> List[str]:
        if subtask == "subtask1":
            return [
                "1\nagent_user\nagent_pass\nagent_pass\n3\n",
                "2\nagent_user\nbad\n2\nagent_user\nagent_pass\n3\n",
            ]
        if subtask == "subtask2":
            return [
                (
                    "1\nsub2_user\nsub2_pass\nsub2_pass\n"
                    "2\nsub2_user\nsub2_pass\n"
                    "1\n"  # enter Manage Tasks
                    "1\nTask One\nDescription One\n"  # add task 1
                    "1\nTask Two\nDescription Two\n"  # add task 2
                    "2\n"  # list tasks (should show both)
                    "3\n1\n"  # mark task 1 completed
                    "2\n"
                    "4\n2\n"  # delete task 2
                    "2\n"
                    "5\n"  # back to main menu
                    "3\n"  # exit application
                ),
                (
                    "2\nsub2_user\nsub2_pass\n"
                    "1\n"  # enter Manage Tasks
                    "2\n"  # list persisted tasks
                    "5\n"  # back to main menu
                    "3\n"
                ),
            ]
        if subtask == "subtask3":
            return [
                (
                    "1\nsub3_user\nsub3_pass\nsub3_pass\n"
                    "2\nsub3_user\nsub3_pass\n"
                    "1\nHigh Priority Task\nFinish quarterly report\nHigh\n2025-12-31\nwork,urgent\n"
                    "1\nLow Priority Task\nClean workspace\nLow\n2026-01-15\nhome,chore\n"
                    "2\n"
                    "/filter priority=high\n"
                    "/filter tag=work\n"
                    "/sort deadline\n"
                    "5\n"
                    "3\n"
                ),
                (
                    "2\nsub3_user\nsub3_pass\n"
                    "/filter priority=high\n"
                    "5\n"
                    "3\n"
                ),
            ]
        if subtask == "subtask4":
            return [
                (
                    "1\nsub4_user\nsub4_pass\nsub4_pass\n"
                    "2\nsub4_user\nsub4_pass\n"
                    "1\nTeam Meeting\nDiscuss roadmap\nMedium\n2025-11-10\nwork,meeting\n"
                    "1\nCode Cleanup\nRefactor modules\nLow\n2026-01-20\nwork,maintenance\n"
                    "2\n"
                    "/search meeting\n"
                    "/archive 1\n"
                    "2\n"
                    "/search meeting\n"
                    "/unarchive 1\n"
                    "5\n"
                    "3\n"
                )
            ]
        if subtask == "subtask5":
            return [
                (
                    "1\nsub5_user\nsub5_pass\nsub5_pass\n"
                    "2\nsub5_user\nsub5_pass\n"
                    "1\nShared Task One\nSync across users\nMedium\n2025-12-01\nteam,shared\n"
                    "/refresh\n"
                    "5\n"
                    "3\n"
                ),
                (
                    "2\nsub5_user\nsub5_pass\n"
                    "/refresh\n"
                    "2\n"
                    "5\n"
                    "3\n"
                ),
            ]
        return []

    def _persist(self, results: Dict[str, CommandResult]) -> None:
        for name, result in results.items():
            log_path = self.logs_dir / f"{name}.txt"
            body = textwrap.dedent(
                f"""
                [command] {' '.join(result.command)}
                [returncode] {result.returncode}

                [stdout]
                {result.stdout}

                [stderr]
                {result.stderr}
                """
            ).strip()
            log_path.write_text(body, encoding="utf-8")

    def _wrap_cmd(self, cmd: Sequence[str]) -> List[str]:
        if not self.conda_env:
            return list(cmd)
        if not self.conda_exe:
            if not self._warned_conda_missing:
                print(
                    f"[DEBUG] CONDA_ENV_NAME={self.conda_env} specified but conda executable missing; "
                    "running commands on host PATH."
                )
                self._warned_conda_missing = True
            return list(cmd)
        return [self.conda_exe, "run", "--no-capture-output", "-n", self.conda_env, *cmd]


# --------------------------------------------------------------------------- #
# Rubric evaluation (heuristic checks)
# --------------------------------------------------------------------------- #


class RubricEvaluator:
    POINTS = {
        "subtask1": [
            "register-login persistence",
            "wrong password rejection",
            "restart persistence",
            "main menu exit",
        ],
        "subtask2": [
            "multiple tasks listed",
            "status updated",
            "delete reflected",
            "persistence across restarts",
            "user scoping",
        ],
        "subtask3": [
            "priority stored",
            "tag filtering",
            "deadline sorting",
            "legacy compatibility",
            "multi-tag support",
        ],
        "subtask4": [
            "substring search",
            "archiving hides default list",
            "archived searchable",
            "unarchive flow",
            "title and description scan",
        ],
        "subtask5": [
            "concurrent adds visible",
            "no corruption",
            "refresh pulls latest",
            "locking prevents races",
            "responsiveness",
        ],
    }

    def evaluate(
        self, subtask: str, evalspace: Path, command_results: Dict[str, Any]
    ) -> RubricResult:
        dispatch = {
            "subtask1": self._eval_subtask1,
            "subtask2": self._eval_subtask2,
            "subtask3": self._eval_subtask3,
            "subtask4": self._eval_subtask4,
            "subtask5": self._eval_subtask5,
        }
        handler = dispatch.get(subtask)
        if not handler:
            return RubricResult(subtask=subtask, score=0, pass_count=0, total_points=0)
        return handler(evalspace, command_results)

    def points_for_subtask(self, subtask: str) -> List[str]:
        return self.POINTS.get(subtask, [])

    def _eval_subtask1(self, evalspace: Path, command_results: Dict[str, Any]) -> RubricResult:
        user_file = evalspace / "data" / "users.txt"
        log_text = self._aggregate_logs(command_results)
        cases: List[Tuple[str, bool]] = []
        cases.append(("register-login persistence", self._has_credentials(user_file)))
        cases.append(("wrong password rejection", "wrong" in log_text.lower() or "invalid" in log_text.lower()))
        cases.append(("restart persistence", self._has_credentials(user_file)))
        cases.append(("main menu exit", "menu" in log_text.lower() or "exit" in log_text.lower()))
        return self._score("subtask1", cases, total=4)

    def _eval_subtask2(self, evalspace: Path, command_results: Dict[str, Any]) -> RubricResult:
        data_dir = evalspace / "data"
        log_text = self._aggregate_logs(command_results).lower()
        task_files = list(data_dir.glob("*_tasks.*")) if data_dir.exists() else []
        file_lines = sum(len(path.read_text(encoding="utf-8", errors="ignore").splitlines()) for path in task_files if path.is_file())
        id_mentions = log_text.count("id:")
        cases: List[Tuple[str, bool]] = []
        cases.append(("multiple tasks listed", file_lines >= 2 or id_mentions >= 2))
        cases.append(("status updated", "marked completed" in log_text or "status: completed" in log_text))
        cases.append(("delete reflected", "deleted" in log_text or "removed" in log_text))
        cases.append(("persistence across restarts", bool(task_files)))
        cases.append(("user scoping", any("sub2_user" in path.name for path in task_files)))
        return self._score("subtask2", cases, total=5)

    def _eval_subtask3(self, evalspace: Path, command_results: Dict[str, Any]) -> RubricResult:
        log_text = self._aggregate_logs(command_results)
        data_text = self._read_all_data(evalspace)
        combined = (log_text + "\n" + data_text).lower()
        cases: List[Tuple[str, bool]] = []
        cases.append(("priority stored", "priority" in combined and "high" in combined))
        cases.append(("tag filtering", "/filter priority=high" in combined and "/filter tag=work" in combined))
        cases.append(("deadline sorting", "/sort deadline" in combined))
        cases.append(("legacy compatibility", "default" in combined or "pending" in combined))
        cases.append(("multi-tag support", "work,urgent" in combined or "multiple tags" in combined))
        return self._score("subtask3", cases, total=5)

    def _eval_subtask4(self, evalspace: Path, command_results: Dict[str, Any]) -> RubricResult:
        combined = (self._aggregate_logs(command_results) + "\n" + self._read_all_data(evalspace)).lower()
        cases: List[Tuple[str, bool]] = []
        cases.append(("substring search", "/search" in combined and "meeting" in combined))
        cases.append(("archiving hides default list", "/archive" in combined and "hiding" in combined))
        cases.append(("archived searchable", "archived" in combined and "search" in combined))
        cases.append(("unarchive flow", "/unarchive" in combined))
        cases.append(("title and description scan", "title" in combined and "description" in combined))
        return self._score("subtask4", cases, total=5)

    def _eval_subtask5(self, evalspace: Path, command_results: Dict[str, Any]) -> RubricResult:
        combined = (self._aggregate_logs(command_results) + "\n" + self._read_all_data(evalspace)).lower()
        cases: List[Tuple[str, bool]] = []
        cases.append(("concurrent adds visible", "shared task" in combined and "refresh" in combined))
        cases.append(("no corruption", "lock" in combined or "synchronized" in combined))
        cases.append(("refresh pulls latest", "/refresh" in combined and "latest" in combined))
        cases.append(("locking prevents races", "lock" in combined))
        cases.append(("responsiveness", "responsive" in combined or "success" in combined))
        return self._score("subtask5", cases, total=5)

    def _score(self, name: str, cases: List[Tuple[str, bool]], total: int) -> RubricResult:
        pass_count = sum(1 for _, passed in cases if passed)
        failed = [title for title, passed in cases if not passed]
        score = round(10 * (pass_count / total)) if total else 0
        return RubricResult(
            subtask=name,
            score=score,
            pass_count=pass_count,
            total_points=total,
            failed_points=failed,
        )

    def _aggregate_logs(self, command_results: Dict[str, Any]) -> str:
        pieces: List[str] = []
        for item in command_results.values():
            if isinstance(item, dict):
                stdout = item.get("stdout") or ""
                stderr = item.get("stderr") or ""
                pieces.append(str(stdout))
                pieces.append(str(stderr))
        return "\n".join(pieces)

    def _has_credentials(self, user_file: Path) -> bool:
        if not user_file.exists():
            return False
        for line in user_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            if ":" in line and line.strip():
                return True
        return False

    def _read_all_data(self, evalspace: Path) -> str:
        data_dir = evalspace / "data"
        if not data_dir.exists():
            return ""
        fragments: List[str] = []
        for file in data_dir.rglob("*"):
            if file.is_file():
                fragments.append(file.read_text(encoding="utf-8", errors="ignore"))
        return "\n".join(fragments)


# --------------------------------------------------------------------------- #
# Coordinator
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
        self.rubric = RubricEvaluator()
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
        conda_exe = self._resolve_conda_executable()
        conda_exe = self._verify_conda_environment(conda_exe)
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
                copy_workspace_filtered(workspace, evalspace)
                logs_dir = ensure_directory(evalspace / "logs")
                conda_env = self.env_config.conda_env_name or None
                cmd_runner = CommandRunner(
                    logs_dir,
                    conda_env=conda_env if conda_exe else None,
                    conda_exe=conda_exe,
                )
                commands = cmd_runner.capture(subtask, evalspace)
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
                    f"[RESULT] {subtask} attempt {attempt}: score={rubric_result.score} "
                    f"pass={rubric_result.pass_count}/{rubric_result.total_points}"
                )
                if rubric_result.failed_points:
                    print(
                        "         Failed rubric points: "
                        + ", ".join(rubric_result.failed_points)
                    )
                if not self.eval_only and rubric_result.score == 10:
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
                    f"[HALT] {subtask} best score {best_score}/10 < 0; stopping evaluation early."
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
                if previous_best is None:
                    ensure_directory(workspace)
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
        bullets = "\n".join(f"- {item}" for item in rubric.failed_points)
        return f"Focus on fixing these rubric gaps:\n{bullets}"

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
            (["javac", "-version"], "javac"),
        ]
        for args, label in checks:
            cmd = [conda_exe, "run", "-n", env_name, *args]
            rc, out, err = run_command(cmd, timeout=450)
            print(f"[DEBUG] Conda check ({label}) rc={rc}")
            if out.strip():
                print(f"[DEBUG] stdout: {out.strip()}")
            if err.strip():
                print(f"[DEBUG] stderr: {err.strip()}")
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
        print(f"[DETAIL] {subtask} attempt {attempt} command diagnostics:")
        inputs_file = commands.get("planned_inputs_file")
        if inputs_file:
            print(f"         Inputs script: {inputs_file}")
        for name, data in commands.items():
            if name == "planned_inputs_file" or not isinstance(data, dict):
                continue
            cmd_list = data.get("command") or []
            cmd_str = " ".join(str(part) for part in cmd_list)
            rc = data.get("returncode")
            status = "PASS" if rc == 0 else "FAIL"
            stdout = (data.get("stdout") or "").strip()
            stderr = (data.get("stderr") or "").strip()
            stdout_fmt = self._truncate_output(stdout)
            stderr_fmt = self._truncate_output(stderr)
            print(f"         [{status}] {name}: rc={rc} cmd={cmd_str}")
            if stdout_fmt:
                print(f"           stdout: {stdout_fmt}")
            if stderr_fmt:
                print(f"           stderr: {stderr_fmt}")

    @staticmethod
    def _truncate_output(text: str, limit: int = 200) -> str:
        if not text:
            return ""
        cleaned = " ".join(text.split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 3] + "..."

    def _print_rubric_diagnostics(self, subtask: str, attempt: int, rubric: RubricResult) -> None:
        points = self.rubric.points_for_subtask(subtask)
        if not points:
            return
        print(f"[DETAIL] {subtask} attempt {attempt} rubric checks:")
        for point in points:
            status = "PASS" if point not in rubric.failed_points else "FAIL"
            print(f"         - {point}: {status}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task38 evaluator runner")
    parser.add_argument("--env", default=".env", help="Path to env file for agent credentials")
    parser.add_argument("--visualize", action="store_true", help="Enable SII SDK visualization")
    parser.add_argument("--description", default="/description.json", help="Path to task description JSON")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_path = resolve_script_path(Path(args.env))
    env_config = EnvConfig.load(env_path, visualize=args.visualize)
    env_config.start_subtask = args.start_subtask
    env_config.eval_only = args.eval_only
    description_path = resolve_script_path(Path(args.description))
    coordinator = EvaluationCoordinator(env_config, description_path, visualize=args.visualize)
    coordinator.run()


if __name__ == "__main__":
    main()
