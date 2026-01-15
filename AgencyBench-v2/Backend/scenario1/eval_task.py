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
        You are working in {target}. Do not access outside paths.
        Implement the solution in C++17 strictly following the JSON specifications.
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
# Command runner (Adapted for C++ Project)
# --------------------------------------------------------------------------- #


class CommandRunner:
    """Execute compile/run commands for C++ Chat App and persist outputs."""

    def __init__(self, logs_dir: Path, conda_env: Optional[str] = None, conda_exe: Optional[str] = None):
        self.logs_dir = ensure_directory(logs_dir)
        self.conda_env = conda_env
        self.conda_exe = conda_exe
        self._warned_conda_missing = False

    def capture(self, subtask: str, root: Path) -> Dict[str, Any]:
        root = root.resolve()
        results: Dict[str, CommandResult] = {}
        
        # 1. Compile (Make)
        compile_result = self.compile_cpp(root, subtask)
        results["compile"] = compile_result

        # 2. Probe (Version or Selftest)
        probe_result = self.run_probe(root, subtask)
        results["probe"] = probe_result

        # 3. Interactive Runs (Demo persistence)
        run_inputs = self._sample_inputs(subtask)
        for idx, input_text in enumerate(run_inputs, start=1):
            result = self.run_app(root, input_text, name=f"run_{idx}")
            results[result.name] = result

        # 4. Unit Test (Task 5 specific)
        if subtask == "subtask5":
            test_queue_res = self.run_unit_test(root, "test_queue")
            results["test_queue"] = test_queue_res

        planned_path = self.logs_dir / f"{subtask}_commands.txt"
        planned_path.write_text(
            "\n".join(run_inputs) if run_inputs else "No scripted inputs", encoding="utf-8"
        )
        self._persist(results)
        summary = {name: res.to_dict() for name, res in results.items()}
        summary["planned_inputs_file"] = str(planned_path)
        return summary

    def compile_cpp(self, root: Path, subtask: str) -> CommandResult:
        target = "task" + subtask.replace("subtask", "")
        cmd = self._wrap_cmd(["make", target])
        rc, out, err = run_command(cmd, cwd=root, timeout=600)
        return CommandResult(name="compile", command=cmd, returncode=rc, stdout=out, stderr=err)

    def run_probe(self, root: Path, subtask: str) -> CommandResult:
        """Run --version or --selftest checks."""
        bin_path = "./bin/chat_app"
        if subtask == "subtask1":
            cmd = self._wrap_cmd([bin_path, "--version"])
        elif subtask == "subtask2":
            cmd = self._wrap_cmd([bin_path, "--selftest", "friend-menu"])
        elif subtask == "subtask3":
            cmd = self._wrap_cmd([bin_path, "--selftest", "chat-loop"])
        elif subtask == "subtask4":
            cmd = self._wrap_cmd([bin_path, "--selftest", "alias-search"])
        elif subtask == "subtask5":
            cmd = self._wrap_cmd([bin_path, "--selftest", "concurrency"])
        else:
            return CommandResult(name="probe", command=[], returncode=0, stdout="", stderr="")

        rc, out, err = run_command(cmd, cwd=root, timeout=60)
        return CommandResult(name="probe", command=cmd, returncode=rc, stdout=out, stderr=err)

    def run_unit_test(self, root: Path, target: str) -> CommandResult:
        """Run make test_queue for Task 5."""
        cmd = self._wrap_cmd(["make", target])
        rc, out, err = run_command(cmd, cwd=root, timeout=120)
        return CommandResult(name=target, command=cmd, returncode=rc, stdout=out, stderr=err)

    def run_app(self, root: Path, input_text: str, name: str = "run") -> CommandResult:
        cmd = self._wrap_cmd(["./bin/chat_app"])
        # Ensure input ends with newline
        if input_text and not input_text.endswith("\n"):
            input_text += "\n"
        
        rc, out, err = run_command(cmd, cwd=root, input_text=input_text, timeout=60)
        return CommandResult(name=name, command=cmd, returncode=rc, stdout=out, stderr=err)

    def _sample_inputs(self, subtask: str) -> List[str]:
        # Strict inputs based on "Action: " prompts
        if subtask == "subtask1":
            return [
                # 1. Register and Exit
                "1\nuser_task1\nSecr3t!\nSecr3t!\n3\n",
                # 2. Restart and Login (Verification run)
                "2\nuser_task1\nSecr3t!\n3\n"
            ]
        if subtask == "subtask2":
            return [
                # 1. Register/Login, Add Friends, Remove Friend
                (
                    "1\nuser_task2\npass\npass\n"  # Register
                    "2\nuser_task2\npass\n"        # Login
                    "A\nalice\n"                   # Add alice
                    "A\nbob\n"                     # Add bob
                    "R\nbob\n"                     # Remove bob
                    "L\n"                          # List
                    "Q\n"                          # Logout
                    "3\n"                          # Exit
                ),
                # 2. Restart, Login, Verify Persistence, Enter Chat
                (
                    "2\nuser_task2\npass\n"        # Login
                    "L\n"                          # List (check persistence)
                    "C\nalice\n"                   # Enter chat
                    "/back\n"                      # Return
                    "Q\n"                          # Logout
                    "3\n"
                ),
            ]
        if subtask == "subtask3":
            return [
                # 1. Register, Setup Friend, Chat Loop
                (
                    "1\nuser_task3\npass\npass\n"
                    "2\nuser_task3\npass\n"
                    "A\nbot_friend\n"
                    "C\nbot_friend\n"
                    "Hello\n"
                    "How are you?\n"
                    "bye\n"
                    "/history 1\n"
                    "/history 2\n"
                    "/exit\n"
                    "Q\n"
                    "3\n"
                ),
                # 2. Restart, Check History Persistence
                (
                    "2\nuser_task3\npass\n"
                    "C\nbot_friend\n"
                    "/history 1\n"
                    "/history 2\n"
                    "/exit\n"
                    "Q\n"
                    "3\n"
                )
            ]
        if subtask == "subtask4":
            return [
                # 1. Register, Friend, Alias, Search
                (
                    "1\nuser_task4\npass\npass\n"
                    "2\nuser_task4\npass\n"
                    "A\nsearch_target\n"
                    "set-alias search_target TargetOne\n"
                    "C\nsearch_target\n"
                    "hello world\n"
                    "SHELL test\n"
                    "/exit\n"
                    "/search HEL\n"  # Should match hello
                    "Q\n"
                    "3\n"
                )
            ]
        if subtask == "subtask5":
            return [
                # 1. Register, Add Friend, Wait for concurrency
                (
                    "1\nuser_task5\npass\npass\n"
                    "2\nuser_task5\npass\n"
                    "A\nconc_friend\n"
                    "C\nconc_friend\n"
                    "hi from user\n"
                    "/exit\n"
                    # No explicit wait in input, reliance on python timeout or app behavior
                    # Just exit cleanly to save log
                    "Q\n"
                    "3\n"
                )
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
# Rubric evaluation (Adapted for C++ Chat App)
# --------------------------------------------------------------------------- #


class RubricEvaluator:
    POINTS = {
        "subtask1": [
            "Build success (Score_B)",
            "Persistence verification (Score_P)",
            "Menu semantics (Score_M)",
        ],
        "subtask2": [
            "Build success (Score_B2)",
            "Friend persistence (Score_F)",
            "Status rendering (Score_S)",
            "Chat entry hook (Score_C)",
        ],
        "subtask3": [
            "Build success (Score_B3)",
            "Live chat flow (Score_CF)",
            "History pagination (Score_HP)",
            "Persistence check (Score_HS)",
        ],
        "subtask4": [
            "Build success (Score_B4)",
            "Alias persistence (Score_AP)",
            "Search accuracy (Score_SA)",
            "Highlight fidelity (Score_HF)",
        ],
        "subtask5": [
            "Build success (Score_B5)",
            "Thread simulation (Score_TS)",
            "Queue integrity (Score_QI)",
            "History consistency (Score_HC)",
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

    # --- Subtask 1: Auth & Basics ---
    def _eval_subtask1(self, evalspace: Path, command_results: Dict[str, Any]) -> RubricResult:
        cases: List[Tuple[str, bool]] = []
        
        compile_rc = command_results.get("compile", {}).get("returncode")
        probe = command_results.get("probe", {})
        version_out = probe.get("stdout", "").strip()
        score_b = (
            compile_rc == 0
            and probe.get("returncode") == 0
            and version_out == "1.0.0"
        )
        cases.append(("Build success (Score_B)", score_b))

        # Score_P: Persistence (users.json + session_state.json)
        users_file = evalspace / "data" / "users.json"
        session_file = evalspace / "logs" / "session_state.json"
        score_p = False
        if users_file.exists():
            try:
                users_data = json.loads(users_file.read_text())
                for user in users_data:
                    uname = user.get("username")
                    salt = user.get("salt", "")
                    hashed = user.get("hash", "")
                    if uname == "user_task1" and salt and re.fullmatch(r"[0-9a-fA-F]{16,}", salt) and re.fullmatch(r"[0-9a-fA-F]{64}", hashed) and hashed != "Secr3t!":
                        score_p = True
                        break
            except Exception:
                score_p = False
        if score_p and session_file.exists():
            try:
                session_data = json.loads(session_file.read_text())
                score_p = session_data.get("last_user") == "user_task1"
            except Exception:
                score_p = False
        cases.append(("Persistence verification (Score_P)", score_p))

        # Score_M: Menu semantics (strict prompts + login success)
        run1 = command_results.get("run_1", {})
        stdout1 = run1.get("stdout", "")
        prompts_ok = all(token in stdout1 for token in ("Action: ", "Username: ", "Password: "))
        run2 = command_results.get("run_2", {})
        stdout2 = run2.get("stdout", "")
        login_ok = "Welcome user_task1" in stdout2 and run2.get("returncode") == 0
        cases.append(("Menu semantics (Score_M)", prompts_ok and login_ok))

        weights = [0.3, 0.5, 0.2]
        return self._score_manual("subtask1", cases, self._apply_weights(cases, weights))

    # --- Subtask 2: Friends ---
    def _eval_subtask2(self, evalspace: Path, command_results: Dict[str, Any]) -> RubricResult:
        cases: List[Tuple[str, bool]] = []

        # Score_B2: Selftest
        probe = command_results.get("probe", {})
        compile_rc = command_results.get("compile", {}).get("returncode")
        valid_b2 = False
        if compile_rc == 0 and probe.get("returncode") == 0:
            try:
                json.loads(probe.get("stdout", ""))
                valid_b2 = True
            except Exception:
                valid_b2 = False
        cases.append(("Build success (Score_B2)", valid_b2))

        # Score_F: Persistence (Check friends file)
        friend_file = evalspace / "data" / "friends_user_task2.json"
        valid_f = False
        if friend_file.exists():
            try:
                fdata = json.loads(friend_file.read_text())
                # Check for alice (should be there) and bob (should be removed)
                names = [f.get("user") for f in fdata]
                statuses = [f.get("status") for f in fdata if f.get("user") == "alice"]
                if "alice" in names and "bob" not in names and all(status in ("online", "offline") for status in statuses):
                    valid_f = True
            except Exception:
                valid_f = False
        cases.append(("Friend persistence (Score_F)", valid_f))

        # Score_S: Status Rendering (Regex match)
        run1 = command_results.get("run_1", {}).get("stdout", "")
        # Regex: space hyphen space name space [status]
        valid_s = bool(re.search(r"\s+-\s+\w+\s+\[(online|offline)\]", run1))
        cases.append(("Status rendering (Score_S)", valid_s))

        # Score_C: Chat Entry Header
        run2 = command_results.get("run_2", {}).get("stdout", "")
        valid_c = "--- Chat with alice ---" in run2
        cases.append(("Chat entry hook (Score_C)", valid_c))

        weights = [0.25, 0.35, 0.2, 0.2]
        return self._score_manual("subtask2", cases, self._apply_weights(cases, weights))

    # --- Subtask 3: Chat Loop & History ---
    def _eval_subtask3(self, evalspace: Path, command_results: Dict[str, Any]) -> RubricResult:
        cases: List[Tuple[str, bool]] = []
        
        # Score_B3: Selftest
        probe = command_results.get("probe", {})
        compile_rc = command_results.get("compile", {}).get("returncode")
        cases.append(("Build success (Score_B3)", compile_rc == 0 and probe.get("returncode") == 0))

        # Score_CF: Alternating Flow (Simple check in logs)
        run1 = command_results.get("run_1", {}).get("stdout", "")
        # Look for pattern > ... [Bot] ... >
        valid_cf = "[Bot]" in run1 and run1.count(">") >= 2
        cases.append(("Live chat flow (Score_CF)", valid_cf))

        # Score_HP: History Pagination
        # In run 1 we sent messages. In run 2 we checked /history 1 and /history 2
        run2 = command_results.get("run_2", {}).get("stdout", "")
        history_lines = re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} \| .* \| .*", run2)
        valid_hp = 0 < len(history_lines) <= 10
        cases.append(("History pagination (Score_HP)", valid_hp))

        # Score_HS: Persistence/Regex
        # Check logs/history/user_task3/bot_friend.log
        hist_dir = evalspace / "data" / "history" / "user_task3"
        valid_hs = False
        if hist_dir.exists():
            for f in hist_dir.glob("*.log"):
                content = f.read_text()
                # ISO8601 | sender | text
                if re.search(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} \| .+ \| .+$", content, flags=re.MULTILINE):
                    valid_hs = True
        cases.append(("Persistence check (Score_HS)", valid_hs))

        weights = [0.2, 0.35, 0.25, 0.2]
        return self._score_manual("subtask3", cases, self._apply_weights(cases, weights))

    # --- Subtask 4: Alias & Search ---
    def _eval_subtask4(self, evalspace: Path, command_results: Dict[str, Any]) -> RubricResult:
        cases: List[Tuple[str, bool]] = []
        
        # Score_B4
        probe = command_results.get("probe", {})
        compile_rc = command_results.get("compile", {}).get("returncode")
        cases.append(("Build success (Score_B4)", compile_rc == 0 and probe.get("returncode") == 0))

        # Score_AP: Alias Persistence
        # Check data/aliases_user_task4.json
        alias_file = evalspace / "data" / "aliases_user_task4.json"
        valid_ap = False
        if alias_file.exists():
            try:
                data = json.loads(alias_file.read_text())
                if data.get("search_target") == "TargetOne":
                    valid_ap = True
            except Exception:
                valid_ap = False
        cases.append(("Alias persistence (Score_AP)", valid_ap))

        # Score_SA: Search Accuracy
        run1 = command_results.get("run_1", {}).get("stdout", "")
        # We searched for HEL, should find hello and SHELL
        run1_lower = run1.lower()
        valid_sa = "hello" in run1_lower and "shell" in run1_lower
        cases.append(("Search accuracy (Score_SA)", valid_sa))

        # Score_HF: Highlighting
        # Should see [[...]]
        valid_hf = bool(re.search(r"\[\[\s*hel", run1, flags=re.IGNORECASE)) and "]]" in run1
        cases.append(("Highlight fidelity (Score_HF)", valid_hf))

        weights = [0.2, 0.35, 0.35, 0.1]
        return self._score_manual("subtask4", cases, self._apply_weights(cases, weights))

    # --- Subtask 5: Concurrency ---
    def _eval_subtask5(self, evalspace: Path, command_results: Dict[str, Any]) -> RubricResult:
        cases: List[Tuple[str, bool]] = []

        # Score_B5
        probe = command_results.get("probe", {})
        compile_rc = command_results.get("compile", {}).get("returncode")
        cases.append(("Build success (Score_B5)", compile_rc == 0 and probe.get("returncode") == 0))

        # Score_TS: Thread Simulation
        # Check selftest output for processed count > 0
        valid_ts = False
        try:
            out_json = json.loads(probe.get("stdout", "{}"))
            if out_json.get("processed", 0) > 0 or out_json.get("autoMessages", 0) > 0:
                valid_ts = True
        except Exception:
            valid_ts = False
        cases.append(("Thread simulation (Score_TS)", valid_ts))

        # Score_QI: Queue Integrity (unit test)
        qt = command_results.get("test_queue", {})
        cases.append(("Queue integrity (Score_QI)", qt.get("returncode") == 0))

        # Score_HC: History Consistency
        # Difficult to prove force-kill in this script structure without specialized runner,
        # but we can check if history exists and is well-formed.
        hist_dir = evalspace / "data" / "history"
        valid_hc = False
        if hist_dir.exists():
            for f in hist_dir.rglob("*.log"):
                content = f.read_text()
                if not content.strip():
                    continue
                lines = content.splitlines()
                pattern = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} \| .+ \| .+$")
                if all(pattern.match(line) for line in lines):
                    valid_hc = True
                    # Require trailing newline to avoid partial line
                    if not content.endswith("\n"):
                        valid_hc = False
                    break
        cases.append(("History consistency (Score_HC)", valid_hc))

        weights = [0.2, 0.35, 0.25, 0.2]
        return self._score_manual("subtask5", cases, self._apply_weights(cases, weights))

    def _score_manual(self, name: str, cases: List[Tuple[str, bool]], score: int) -> RubricResult:
        pass_count = sum(1 for _, passed in cases if passed)
        failed = [title for title, passed in cases if not passed]
        total = len(cases)
        return RubricResult(
            subtask=name,
            score=min(10, max(0, score)),
            pass_count=pass_count,
            total_points=total,
            failed_points=failed,
        )

    def _apply_weights(self, cases: List[Tuple[str, bool]], weights: Sequence[float]) -> int:
        total = 0.0
        for (_, passed), weight in zip(cases, weights):
            total += (1.0 if passed else 0.0) * weight
        return round(10 * total)

    def _aggregate_logs(self, command_results: Dict[str, Any]) -> str:
        pieces: List[str] = []
        for item in command_results.values():
            if isinstance(item, dict):
                stdout = item.get("stdout") or ""
                stderr = item.get("stderr") or ""
                pieces.append(str(stdout))
                pieces.append(str(stderr))
        return "\n".join(pieces)


# --------------------------------------------------------------------------- #
# Coordinator (Unchanged)
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
            (["g++", "--version"], "g++"),
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
    parser = argparse.ArgumentParser(description="Task11 evaluator runner")
    parser.add_argument("--env", default=".env", help="Path to env file for agent credentials")
    parser.add_argument("--visualize", action="store_true", help="Enable SII SDK visualization")
    parser.add_argument("--description", default="description.json", help="Path to task description JSON")
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
