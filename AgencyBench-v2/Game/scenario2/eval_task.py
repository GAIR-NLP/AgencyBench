"""Automated evaluation pipeline for AgencyBench task8.

This script orchestrates the following loop for each subtask described in
`AgencyBench_v2/description.json`:

1. Load environment configuration from `.env` (model endpoints, API keys).
2. Invoke the target model via the SII Agent SDK to complete the subtask in the
   designated workspace directory.
3. Collect evidence (screenshots, videos, DOM dumps, logs) using the sandbox
   automation utilities described in `basic_usage.md`.
4. Run two evaluators:
   - Text-only scoring through OpenRouter (`anthropic/claude-haiku-4.5`).
   - Vision/video scoring through the provider configured in `.env` (Google Gemini by default).
5. Merge the two scores; if either evaluator returns `no`, feed the feedback
   back into the SII Agent and repeat until success or attempt limit.
6. Persist detailed metadata into `meta_eval.json` under `AgencyBench_v2/task8`.

The script is intentionally verbose to ease debugging. Runtime logs describe the
current subtask, attempt, evaluator outputs, and artifact locations. All logic
is kept in this single file per requirements.
"""

from __future__ import annotations

import pdb
import argparse
import asyncio
import base64
import json
import mimetypes
import os
import random
import shutil
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re

from agent_sandbox import Sandbox


def _suppress_event_loop_closed_errors() -> None:
    try:
        import asyncio.base_subprocess as _base_subprocess
    except ImportError:
        return

    original_del = getattr(_base_subprocess.BaseSubprocessTransport, "__del__", None)

    if original_del is None:
        return

    # Avoid double patching
    if getattr(original_del, "_sii_patched", False):
        return

    def _patched_del(self, *args, **kwargs):
        try:
            original_del(self, *args, **kwargs)
        except RuntimeError as exc:
            if "Event loop is closed" in str(exc):
                return
            raise

    setattr(_patched_del, "_sii_patched", True)
    _base_subprocess.BaseSubprocessTransport.__del__ = _patched_del


_suppress_event_loop_closed_errors()
try:
    from google import genai
    from google.genai import errors as genai_errors
except ImportError:  # Lazy import to allow non-Google vision providers.
    genai = None  # type: ignore[assignment]
    genai_errors = None  # type: ignore[assignment]
import httpx
from openai import OpenAI, AzureOpenAI
from playwright.async_api import async_playwright
from sii_agent_sdk import (
    AssistantMessage,
    Message,
    SiiAgentOptions,
    TextBlock,
    SiiAgentSession as SDKAgentSession,
)


###############################################################################
# Utility helpers
###############################################################################


def load_env_file(env_path: Path) -> Dict[str, str]:
    """Load key=value pairs from the given env file and inject into os.environ."""

    if not env_path.exists():
        raise FileNotFoundError(f"Required env file not found: {env_path}")

    parsed: Dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('\"')
        parsed[key] = value
        os.environ[key] = value
    return parsed


def ensure_conda_env(expected_env: str) -> None:
    """Warn if the current conda environment does not match the expectation."""

    active_env = os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("VIRTUAL_ENV")
    if active_env and expected_env and expected_env not in active_env:
        print(
            f"[WARN] Active environment '{active_env}' does not match expected '{expected_env}'."
        )


def ensure_bridge_timeout(options: SiiAgentOptions, timeout_ms: int) -> None:
    if timeout_ms:
        options.timeout_ms = max(timeout_ms, options.timeout_ms or 0)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_workspace(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source workspace missing: {src}")
    ensure_directory(dst)
    clear_directory(dst)
    for root, dirs, files in os.walk(src):
        rel_root = Path(root).relative_to(src)
        if _contains_hidden(rel_root.parts):
            dirs[:] = []
            continue
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        target_root = dst / rel_root
        target_root.mkdir(parents=True, exist_ok=True)
        for filename in files:
            if filename.startswith('.'):
                continue
            src_file = Path(root) / filename
            dst_file = target_root / filename
            shutil.copy2(src_file, dst_file)


def clear_directory(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def _contains_hidden(segments: Iterable[str]) -> bool:
    return any(part.startswith('.') for part in segments if part)


def derive_model_name(identifier: str) -> str:
    raw = identifier.split("/")[-1].strip()
    if not raw:
        raw = identifier.strip()
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    sanitized = sanitized.strip("._-") or "model"
    return sanitized


def enforce_int_score(value: Any) -> int:
    """Coerce an arbitrary score value into an integer within [0, 10]."""

    original = value
    coerced: Optional[int] = None

    if isinstance(value, bool) or value is None:
        coerced = 0
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if not numeric.is_integer():
            print(f"[EVAL][WARN] Non-integer score {numeric}; rounding to nearest int")
        coerced = int(round(numeric))
    elif isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            numeric = float(match.group())
            if not numeric.is_integer():
                print(f"[EVAL][WARN] Non-integer score '{value}'; rounding to nearest int")
            coerced = int(round(numeric))
    if coerced is None:
        print(f"[EVAL][WARN] Unrecognized score '{original}'; defaulting to 0")
        coerced = 0

    if coerced < 0 or coerced > 10:
        clamped = max(0, min(10, coerced))
        if clamped != coerced:
            print(f"[EVAL][WARN] Score {coerced} out of range; clamped to {clamped}")
        coerced = clamped

    return coerced


def extract_generation_plan(text: str) -> Dict[str, Any]:


    # pdb.set_trace()

    if not text:
        raise ValueError("Assistant response is empty; cannot extract plan")

    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    json_block = _find_json_block(stripped)
    if not json_block:
        raise ValueError(f"Failed to locate JSON plan in assistant response: {stripped[:200]}")

    try:
        return json.loads(json_block)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON plan: {exc}") from exc


def _find_json_block(text: str) -> Optional[str]:
    fence_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]

            if in_string:
                if escape:
                    escape = False
                    continue
                if char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue

            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]

        start = text.find("{", start + 1)
    return None


def apply_plan_to_workspace(plan: Dict[str, Any], workspace: Path) -> None:
    files = plan.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Plan JSON must contain a non-empty 'files' list")

    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    deleted: List[Path] = []
    for file_obj in files:
        if not isinstance(file_obj, dict):
            raise ValueError("Each file entry must be an object")
        path_str = file_obj.get("path")
        if not path_str:
            raise ValueError("File entry missing 'path'")
        target_path = (workspace / Path(path_str)).resolve()
        if not str(target_path).startswith(str(workspace)):
            raise ValueError(f"Invalid file path outside workspace: {path_str}")
        delete_requested = bool(file_obj.get("delete"))
        if delete_requested:
            if target_path.exists():
                if target_path.is_dir():
                    shutil.rmtree(target_path)
                else:
                    target_path.unlink()
                deleted.append(target_path)
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)

        encoding = (file_obj.get("encoding") or "utf-8").lower()
        content = file_obj.get("content")
        if content is None:
            raise ValueError(f"File entry for {path_str} missing 'content'")

        if encoding == "base64":
            data = base64.b64decode(content)
            target_path.write_bytes(data)
        else:
            target_path.write_text(str(content), encoding=encoding, errors="ignore")
        written.append(target_path)

    print("[SII] Applied files:")
    for path in written:
        print(f"       - {path.relative_to(workspace)} ({path.stat().st_size} bytes)")
    for path in deleted:
        print(f"       - deleted {path.relative_to(workspace)}")


###############################################################################
# Data classes for structured state
###############################################################################


@dataclass
class EnvConfig:
    conda_env: str
    sandbox_base_url: str
    sii_api_key: str
    sii_api_base: str
    sii_auth_type: str
    sii_target_model: str
    target_model_name: str
    sii_system_prompt: str
    sii_username: str
    sii_password: str
    sii_max_turns: int
    eval_text_model: str
    text_eval_api_key: str
    text_eval_api_base: str
    vision_model: str
    vision_provider: str
    google_api_key: Optional[str]
    qwen_api_base: Optional[str]
    qwen_api_key: Optional[str]
    max_attempts: int
    bridge_timeout_ms: int
    sandbox_timeout_sec: int
    gzy_openai_mode: bool

    @staticmethod
    def _as_bool(value: Optional[str], default: bool = False) -> bool:
        if value is None:
            return default
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default

    @classmethod
    def from_env(cls, data: Dict[str, str]) -> "EnvConfig":
        def fetch(key: str) -> Optional[str]:
            value = data.get(key) or os.environ.get(key)
            if value:
                return value.strip()
            return None

        def require(key: str) -> str:
            value = fetch(key)
            if not value or value.startswith("<YOUR_"):
                raise ValueError(f"Environment variable '{key}' must be set in .env")
            return value

        target_model = require("SII_TARGET_MODEL")
        vision_provider = data.get("EVAL_VISION_PROVIDER", "google").strip().lower()
        if vision_provider not in {"google", "qwen"}:
            raise ValueError(
                "EVAL_VISION_PROVIDER must be either 'google' or 'qwen' in .env"
            )

        sii_api_key = require("SII_AGENT_API_KEY")
        sii_api_base = fetch("SII_AGENT_API_BASE_URL") or "https://openrouter.ai/api/v1"
        text_eval_api_key = fetch("EVAL_TEXT_API_KEY") or sii_api_key
        text_eval_api_base = fetch("EVAL_TEXT_API_BASE_URL") or sii_api_base

        def optional(key: str) -> Optional[str]:
            return fetch(key)

        config = cls(
            conda_env=data.get("CONDA_ENV_NAME", ""),
            sandbox_base_url=data.get("SANDBOX_BASE_URL", "http://localhost:8080"),
            sii_api_key=sii_api_key,
            sii_api_base=sii_api_base,
            sii_auth_type=data.get("SII_AUTH_TYPE", "USE_OPENAI"),
            sii_target_model=target_model,
            target_model_name=derive_model_name(target_model),
            sii_system_prompt=data.get(
                "SII_SYSTEM_PROMPT",
                "You are a senior front-end engineer tasked with delivering polished 2048 puzzle interfaces.",
            ),
            sii_username=require("SII_USERNAME"),
            sii_password=require("SII_PASSWORD"),
            sii_max_turns=int(data.get("SII_MAX_TURNS", "20")),
            eval_text_model=require("EVAL_TEXT_MODEL"),
            text_eval_api_key=text_eval_api_key,
            text_eval_api_base=text_eval_api_base,
            vision_model=require("VISION_MODEL"),
            vision_provider=vision_provider,
            google_api_key=optional("GOOGLE_API_KEY"),
            qwen_api_base=optional("QWEN_VISION_BASE_URL"),
            qwen_api_key=optional("QWEN_VISION_API_KEY"),
            max_attempts=int(data.get("MAX_SUBTASK_ATTEMPTS", "3")),
            bridge_timeout_ms=int(data.get("BRIDGE_TIMEOUT_MS", "180000")),
            sandbox_timeout_sec=int(data.get("SANDBOX_TIMEOUT_SEC", "20")),
            gzy_openai_mode=cls._as_bool(
                data.get("GZY_OPENAI_MODE"),
                default="yunstorm" in (text_eval_api_base.lower()),
            ),
        )
        if config.vision_provider == "google":
            if not config.google_api_key:
                raise ValueError("GOOGLE_API_KEY must be set when EVAL_VISION_PROVIDER=google")
        else:
            if not config.qwen_api_base or not config.qwen_api_key:
                raise ValueError(
                    "QWEN_VISION_BASE_URL and QWEN_VISION_API_KEY must be set when EVAL_VISION_PROVIDER=qwen"
                )
        return config

    def inject_defaults(self) -> None:
        os.environ.setdefault("SII_AGENT_API_KEY", self.sii_api_key)
        os.environ.setdefault("SII_AGENT_API_BASE_URL", self.sii_api_base)
        if self.google_api_key:
            os.environ.setdefault("GOOGLE_API_KEY", self.google_api_key)
        if self.qwen_api_base:
            os.environ.setdefault("QWEN_VISION_BASE_URL", self.qwen_api_base)
        if self.qwen_api_key:
            os.environ.setdefault("QWEN_VISION_API_KEY", self.qwen_api_key)
        os.environ.setdefault("SANDBOX_BASE_URL", self.sandbox_base_url)
        os.environ.setdefault("EVAL_TEXT_API_KEY", self.text_eval_api_key)
        os.environ.setdefault("EVAL_TEXT_API_BASE_URL", self.text_eval_api_base)
        os.environ.setdefault("SII_OPENAI_API_KEY", self.sii_api_key)
        os.environ.setdefault("SII_OPENAI_BASE_URL", self.sii_api_base)
        os.environ.setdefault("SII_OPENAI_MODEL", self.sii_target_model)
        os.environ.setdefault("SII_USERNAME", self.sii_username)
        os.environ.setdefault("SII_PASSWORD", self.sii_password)
        if self.gzy_openai_mode:
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
            os.environ.setdefault("AZURE_OPENAI_ENDPOINT", self.text_eval_api_base)
            os.environ.setdefault("AZURE_OPENAI_API_KEY", self.text_eval_api_key)
            os.environ.setdefault("AZURE_OPENAI_API_VERSION", api_version)
            os.environ.setdefault("OPENAI_API_TYPE", "azure")


@dataclass
class ArtifactBundle:
    attempt_index: int
    workspace_path: Path
    evalspace_path: Path
    screenshot_paths: List[Path] = field(default_factory=list)
    video_paths: List[Path] = field(default_factory=list)
    dom_snapshot: Optional[str] = None
    debug_logs: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluatorResult:
    model: str
    score: float
    passed: bool
    reason: str
    confidence: float
    raw_response: Dict[str, Any] = field(default_factory=dict)
    input_prompt: str = ""
    input_files: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AttemptRecord:
    attempt_index: int
    prompt: str
    agent_messages: List[Dict[str, Any]]
    assistant_text: str
    artifacts: ArtifactBundle
    text_result: EvaluatorResult
    vision_result: EvaluatorResult
    status: str


###############################################################################
# Core automation components
###############################################################################


class SiiAgentSession:
    def __init__(self, env_config: EnvConfig, workspace: Optional[Path] = None):
        self.env = env_config
        self.workspace: Optional[Path] = workspace.resolve() if workspace else None
        self._session: Optional[SDKAgentSession] = None
        self._options: Optional[SiiAgentOptions] = None
        self._original_cwd: Path = Path.cwd()

    def set_workspace(self, workspace: Path, *, reset_history: bool = False) -> None:
        self.workspace = workspace.resolve()
        ensure_directory(self.workspace)
        if Path.cwd() != self.workspace:
            os.chdir(self.workspace)

        env_vars = {
            "OPENAI_API_KEY": self.env.sii_api_key,
            "OPENAI_BASE_URL": self.env.sii_api_base,
            "SII_OPENAI_API_KEY": self.env.sii_api_key,
            "SII_OPENAI_BASE_URL": self.env.sii_api_base,
            "SII_OPENAI_MODEL": self.env.sii_target_model,
            "SII_USERNAME": self.env.sii_username,
            "SII_PASSWORD": self.env.sii_password,
        }
        if self.env.gzy_openai_mode:
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
            env_vars.update(
                {
                    "AZURE_OPENAI_ENDPOINT": self.env.sii_api_base,
                    "AZURE_OPENAI_API_KEY": self.env.sii_api_key,
                    "AZURE_OPENAI_API_VERSION": api_version,
                    "OPENAI_API_VERSION": api_version,
                    "OPENAI_API_TYPE": "azure",
                }
            )

        options = SiiAgentOptions(
            system_prompt=self.env.sii_system_prompt,
            max_turns=self.env.sii_max_turns,
            auth_type=self.env.sii_auth_type,
            cwd=str(self.workspace),
            yolo=True,
            allowed_tools=[],
            model=self.env.sii_target_model,
            env=env_vars,
        )
        ensure_bridge_timeout(options, self.env.bridge_timeout_ms)
        self._options = options

        if self._session is None or reset_history:
            self._session = SDKAgentSession(options)

    async def send(self, user_message: str) -> Tuple[List[Dict[str, Any]], str]:
        if self.workspace is None or self._session is None or self._options is None:
            raise RuntimeError("Workspace not initialised for SiiAgentSession")

        transcript: List[Dict[str, Any]] = []
        status_seen: set[str] = set()
        assistant_chunks: List[str] = []
        index = 0
        async for message in self._session.run(user_message, options=self._options):
            index += 1
            payload = self._normalize_message(message, index)
            transcript.append(payload)
            msg_type = payload.get("type")

            if msg_type == "status":
                status = payload.get("status") or payload.get("message")
                if status and status not in status_seen:
                    print(f"[SII] status -> {status}")
                    status_seen.add(status)
            elif msg_type == "assistant" and payload.get("text"):
                assistant_chunks.append(payload["text"])
            elif msg_type == "tool_result":
                content = payload.get("content")
                if isinstance(content, str):
                    assistant_chunks.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, str):
                            assistant_chunks.append(item)
                        elif isinstance(item, dict):
                            text_value = item.get("text")
                            if text_value:
                                assistant_chunks.append(str(text_value))

        assistant_text = "\n".join(chunk.strip() for chunk in assistant_chunks if chunk).strip()
        print(f"[SII] assistant response length: {len(assistant_text)} chars")

        return transcript, assistant_text

    def restore_cwd(self) -> None:
        try:
            if self._original_cwd and self._original_cwd.exists():
                os.chdir(self._original_cwd)
        except Exception:
            pass

    @staticmethod
    def _normalize_message(message: Message, index: int) -> Dict[str, Any]:
        if isinstance(message, AssistantMessage):
            texts = [block.text for block in message.content if isinstance(block, TextBlock)]
            return {
                "index": index,
                "type": "assistant",
                "text": "\n".join(texts),
            }
        if isinstance(message, TextBlock):
            return {"index": index, "type": "text", "text": message.text}
        if hasattr(message, "type"):
            data = message.__dict__.copy()
            data["index"] = index
            return data
        return {"index": index, "type": "unknown", "content": str(message)}


class SandboxWorkspace:
    def __init__(
        self,
        base_url: str,
        model_slug: str,
        subtask_name: str,
        attempt_index: int,
        *,
        visualize: bool = False,
    ):
        self.client = Sandbox(base_url=base_url)
        self.model_slug = model_slug
        self.subtask_name = subtask_name
        self.attempt_index = attempt_index
        self.remote_workspace: Optional[str] = None
        self.visualize = visualize

    def prepare_remote_workspace(self) -> str:
        context = self.client.sandbox.get_context()
        home_dir = context.home_dir.rstrip("/")
        remote_dir = (
            f"{home_dir}/agency_eval/{self.model_slug}/"
            f"{self.subtask_name}/attempt_{self.attempt_index:02d}"
        )
        mkdir_result = self.client.shell.exec_command(command=f"mkdir -p '{remote_dir}'")
        if not mkdir_result.success:
            raise RuntimeError(f"Failed to create remote workspace: {mkdir_result.message}")
        self.remote_workspace = remote_dir
        return remote_dir

    def sync_local_workspace(self, local_path: Path) -> None:
        if not self.remote_workspace:
            self.prepare_remote_workspace()
        assert self.remote_workspace

        files: List[Path] = [p for p in local_path.rglob("*") if p.is_file()]
        for file_path in files:
            rel = file_path.relative_to(local_path)
            remote_path = f"{self.remote_workspace}/{rel.as_posix()}"
            remote_dir = os.path.dirname(remote_path)
            self.client.shell.exec_command(command=f"mkdir -p '{remote_dir}'")
            data = file_path.read_bytes()
            result = self.client.file.upload_file(file=data, path=remote_path)
            if not result.success:
                raise RuntimeError(f"Failed to upload {file_path} to {remote_path}: {result.message}")

    def cleanup_remote_workspace(self) -> None:
        if not self.remote_workspace:
            return
        self.client.shell.exec_command(command=f"rm -rf '{self.remote_workspace}'")
        self.remote_workspace = None

    def _start_http_server(self) -> Tuple[int, str]:
        if not self.remote_workspace:
            raise RuntimeError("Remote workspace is not prepared")

        port = random.randint(28080, 29999)
        command = (
            f"cd '{self.remote_workspace}' && "
            f"python -u -m http.server {port} --bind 127.0.0.1"
        )
        result = self.client.shell.exec_command(command=command, async_mode=True)
        if not result.success or not result.data or not result.data.session_id:
            message = result.message if result else "unknown failure"
            raise RuntimeError(f"Failed to launch HTTP server: {message}")
        return port, result.data.session_id

    def _stop_http_server(self, session_id: str) -> None:
        try:
            self.client.shell.kill_process(id=session_id)
        except Exception as exc:  # pragma: no cover - best effort cleanup
            print(f"[SANDBOX][WARN] Failed to stop HTTP server: {exc}")

    async def _wait_for_http_server(self, port: int) -> None:
        assert self.remote_workspace
        probe = textwrap.dedent(
            f"""
            python - <<'PY'
import socket
import sys
s = socket.socket()
try:
    s.settimeout(0.5)
    s.connect(('127.0.0.1', {port}))
except OSError:
    sys.exit(1)
else:
    s.close()
PY
            """
        ).strip()

        for _ in range(12):
            check = self.client.shell.exec_command(command=probe)
            if check.success and check.data and check.data.exit_code == 0:
                return
            await asyncio.sleep(0.5)
        raise TimeoutError(f"HTTP server on port {port} did not become ready")

    async def capture_screenshots(
        self,
        html_entry: str,
        output_dir: Path,
        names: Iterable[str],
        interaction_plan: Optional[List[Dict[str, Any]]] = None,
        record_video: bool = False,
        video_name: str = "capture.webm",
    ) -> Tuple[List[Path], Optional[Path], Dict[str, Any]]:
        if not self.remote_workspace:
            self.prepare_remote_workspace()
        assert self.remote_workspace

        browser_info = self.client.browser.get_info().data
        screenshots: List[Path] = []
        video_path: Optional[Path] = None
        diagnostics: Dict[str, Any] = {}
        server_session_id: Optional[str] = None
        remote_url = f"file://{self.remote_workspace}/{html_entry}"

        try:
            port, server_session_id = self._start_http_server()
            await self._wait_for_http_server(port)
            remote_url = f"http://127.0.0.1:{port}/{html_entry}"
            diagnostics["http_server_port"] = port
            diagnostics["http_server_mode"] = "python"
        except Exception as exc:
            diagnostics.setdefault("warnings", []).append(
                f"Falling back to file:// load: {exc}"
            )
            if server_session_id:
                self._stop_http_server(server_session_id)
            server_session_id = None

        try:
            async with async_playwright() as playwright:
                visual_browser = None
                visual_page = None
                browser = None
                try:
                    if self.visualize:
                        try:
                            visual_browser = await playwright.chromium.launch(headless=False)
                            visual_page = await visual_browser.new_page(
                                viewport={
                                    "width": browser_info.viewport.width,
                                    "height": browser_info.viewport.height,
                                }
                            )
                            await visual_page.goto("http://localhost:8080", wait_until="networkidle")
                            await visual_page.evaluate(
                                """
(() => {
  if (window.workspace) {
    workspace.panels.browser.active = true;
    workspace.panels.browser.visible = true;
    workspace.renderPanels();
  }
})();
                                """
                            )
                            await visual_page.wait_for_timeout(1000)
                            print("[SANDBOX] Visualization window opened at http://localhost:8080")
                        except Exception as exc:
                            print(f"[SANDBOX][WARN] Failed to open visualization window: {exc}")
                            if visual_browser is not None:
                                await visual_browser.close()
                            visual_browser = None
                            visual_page = None

                    browser = await playwright.chromium.connect_over_cdp(browser_info.cdp_url)
                    context = None
                    try:
                        context_kwargs = {
                            "viewport": {
                                "width": browser_info.viewport.width,
                                "height": browser_info.viewport.height,
                            }
                        }
                        if record_video:
                            context = await browser.new_context(
                                viewport=context_kwargs["viewport"],
                                record_video_dir=str(output_dir),
                                record_video_size=context_kwargs["viewport"],
                            )
                        else:
                            context = await browser.new_context(**context_kwargs)

                        page = await context.new_page()
                        diagnostics["serving_url"] = remote_url
                        load_attempts = 3
                        last_error: Optional[Exception] = None
                        for attempt in range(load_attempts):
                            try:
                                await page.goto(remote_url, wait_until="domcontentloaded")
                                try:
                                    await page.wait_for_load_state("networkidle", timeout=3000)
                                except Exception:
                                    diagnostics.setdefault("warnings", []).append(
                                        "Network idle wait timed out"
                                    )
                                last_error = None
                                break
                            except Exception as exc:
                                last_error = exc
                                if attempt + 1 == load_attempts:
                                    raise
                                await page.wait_for_timeout(500)
                        if last_error:
                            diagnostics.setdefault("errors", []).append(f"Goto warning: {last_error}")
                        found_selector = None
                        for selector in (".cell", ".tile", ".intersection"):
                            try:
                                await page.wait_for_selector(selector, timeout=5000)
                                found_selector = selector
                                break
                            except Exception:
                                continue
                        if found_selector:
                            diagnostics["grid_selector"] = found_selector
                        else:
                            diagnostics.setdefault(
                                "warnings",
                                [],
                            ).append("Failed to detect grid cells (.cell/.tile/.intersection)")
                        diagnostics["page_title"] = await page.title()
                        diagnostics["page_url"] = page.url
                        try:
                            diagnostics["body_preview"] = (await page.content())[:2000]
                        except Exception:
                            diagnostics.setdefault("warnings", []).append("Failed to capture page content")

                        if interaction_plan:
                            await self._execute_interaction_plan(page, interaction_plan, diagnostics)

                        for name in names:
                            path = output_dir / name
                            img_bytes = await page.screenshot(type="png", full_page=True)
                            path.write_bytes(img_bytes)
                            screenshots.append(path)

                        if record_video:
                            await page.close()
                            await context.close()
                            context = None
                            if page.video:
                                raw_video_path = await page.video.path()
                                video_path = output_dir / video_name
                                shutil.copy2(raw_video_path, video_path)

                    finally:
                        if context is not None:
                            await context.close()
                        if browser is not None:
                            await browser.close()

                finally:
                    if visual_browser is not None:
                        await visual_browser.close()
        finally:
            if server_session_id:
                self._stop_http_server(server_session_id)

        return screenshots, video_path, diagnostics

    async def _execute_interaction_plan(self, page, plan: List[Dict[str, Any]], diagnostics: Dict[str, Any]) -> None:
        for step in plan:
            action = step.get("action")
            if action == "click-board":
                selector = step.get("selector", "#gameBoard,#board")
                board = None
                for candidate in selector.split(","):
                    try:
                        board = await page.wait_for_selector(candidate.strip(), timeout=2000)
                        if board:
                            break
                    except Exception:
                        continue
                if not board:
                    diagnostics.setdefault("errors", []).append("Board selector not found")
                    continue
                box = await board.bounding_box()
                if not box:
                    diagnostics.setdefault("errors", []).append("Board bounding box missing")
                    continue
                clicks = step.get("clicks") or []
                for offset in clicks:
                    x = box["x"] + box["width"] * offset.get("x_ratio", 0.5)
                    y = box["y"] + box["height"] * offset.get("y_ratio", 0.5)
                    await page.mouse.click(x, y)
                    await page.wait_for_timeout(offset.get("wait_ms", 500))
            elif action == "eval-js":
                expression = step.get("expression")
                label = step.get("label", "result")
                try:
                    result = await page.evaluate(expression)
                    diagnostics[label] = result
                except Exception as exc:
                    diagnostics.setdefault("errors", []).append(f"JS eval failed for {label}: {exc}")
            elif action == "click-element":
                selector = step.get("selector")
                wait_ms = int(step.get("wait_ms", 300))
                timeout = int(step.get("timeout", 2000))
                if not selector:
                    diagnostics.setdefault("errors", []).append("click-element missing selector")
                    continue
                try:
                    element = await page.wait_for_selector(selector, timeout=timeout)
                    await element.click()
                    await page.wait_for_timeout(wait_ms)
                except Exception as exc:
                    diagnostics.setdefault("errors", []).append(f"Failed to click selector {selector}: {exc}")
            elif action == "click-cells":
                coords = step.get("coords") or []
                if not coords:
                    diagnostics.setdefault("warnings", []).append("click-cells requested with empty coords")
                    continue
                for entry in coords:
                    if isinstance(entry, dict):
                        coord = entry.get("coord")
                        wait_ms = int(entry.get("wait_ms", step.get("wait_ms", 400)))
                    else:
                        coord = str(entry)
                        wait_ms = int(step.get("wait_ms", 400))
                    if not coord:
                        diagnostics.setdefault("warnings", []).append("click-cells encountered empty coord")
                        continue
                    try:
                        layout = await page.evaluate(
                            "window.game && window.game.describeLayout ? window.game.describeLayout() : null"
                        )
                    except Exception as exc:
                        diagnostics.setdefault("errors", []).append(f"describeLayout failed: {exc}")
                        break
                    cells = layout.get("cells") if isinstance(layout, dict) else None
                    if not isinstance(cells, dict):
                        diagnostics.setdefault("errors", []).append("describeLayout().cells missing or invalid")
                        break
                    cell_info = cells.get(coord)
                    if not isinstance(cell_info, dict):
                        diagnostics.setdefault("errors", []).append(f"Cell {coord} missing from describeLayout()")
                        continue
                    x = cell_info.get("x")
                    y = cell_info.get("y")
                    if x is None or y is None:
                        diagnostics.setdefault("errors", []).append(f"Cell {coord} missing x/y coordinates")
                        continue
                    try:
                        await page.mouse.click(float(x), float(y))
                        await page.wait_for_timeout(wait_ms)
                    except Exception as exc:
                        diagnostics.setdefault("errors", []).append(f"Failed to click cell {coord}: {exc}")
            elif action == "press-keys":
                keys = step.get("keys") or []
                if isinstance(keys, str):
                    keys = [keys]
                delay = int(step.get("wait_ms", 200))
                if not keys:
                    diagnostics.setdefault("warnings", []).append("press-keys requested with empty key list")
                    continue
                for key in keys:
                    if not key:
                        continue
                    try:
                        await page.keyboard.press(str(key))
                        await page.wait_for_timeout(delay)
                    except Exception as exc:
                        diagnostics.setdefault("errors", []).append(f"Failed to press key {key}: {exc}")
            elif action == "wait":
                await page.wait_for_timeout(int(step.get("duration", 500)))
            else:
                diagnostics.setdefault("errors", []).append(f"Unknown action: {action}")


###############################################################################
# Evaluators (text + vision)
###############################################################################


class OpenRouterTextEvaluator:
    def __init__(self, env_config: EnvConfig):
        self.env = env_config
        if self.env.gzy_openai_mode:
            endpoint = self.env.text_eval_api_base.rstrip("/")
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=self.env.text_eval_api_key,
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
            )
        else:
            self.client = OpenAI(base_url=self.env.text_eval_api_base, api_key=self.env.text_eval_api_key)

    def evaluate(
        self,
        subtask_name: str,
        rubric: str,
        bundle: ArtifactBundle,
        prompt_history: List[Dict[str, Any]],
        *,
        log_io: bool = False,
    ) -> EvaluatorResult:
        workspace = bundle.workspace_path
        artifact_summaries = self._build_artifact_summary(bundle)
        conversation_digest = self._summarize_transcript(prompt_history)

        user_prompt = textwrap.dedent(
            f"""
            You are the lead reviewer for AgencyBench task8. Assign an integer score between 0 and 10 inclusive for {subtask_name}, where 10 = flawless & fully compliant, 8 = strong with only minor polish gaps, 6 = acceptable but missing important requirements, 4 = significant issues with partial progress, 2 = minimal progress with critical failures, and 0 = unusable or absent work. Scores must be whole numbers—no decimals.
            Rubric:
            {rubric}

            Workspace root: {workspace}
            Conversation summary:
            {conversation_digest}

            Deliverable excerpts:
            {artifact_summaries}

            Respond in JSON with keys: score, confidence, reason. The score must be an integer from 0 to 10 with no decimals. Confidence must be a number between 0.0 (not confident) and 1.0 (extremely confident) using exactly one decimal place. Reason should concisely justify the score. Do not include any additional fields. You MUST output in JSON format.
            """
        ).strip()

        if log_io:
            print(f"[EVAL][Text] Prompt for {subtask_name}:\n{'-'*40}\n{user_prompt}\n{'-'*40}")

        messages: List[Dict[str, str]] = [{"role": "user", "content": user_prompt}]
        if self.env.gzy_openai_mode:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": "Assistant is a large language model trained by OpenAI.",
                },
            )

        response = self.client.chat.completions.create(
            model=self.env.eval_text_model,
            messages=messages,
        )
        content = response.choices[0].message.content  # type: ignore[index]
        parsed = self._parse_json_from_text(content)
        score_value = parsed.get("score", 0)
        score_int = enforce_int_score(score_value)
        confidence_value = parsed.get("confidence", 0.0)
        try:
            confidence_float = float(confidence_value)
        except (TypeError, ValueError):
            confidence_float = 0.0
        confidence_float = max(0.0, min(1.0, round(confidence_float, 1)))
        passed_flag = score_int >= 6
        result = EvaluatorResult(
            model=self.env.eval_text_model,
            score=float(score_int),
            passed=passed_flag,
            reason=str(parsed.get("reason", "")),
            confidence=confidence_float,
            raw_response={"content": content},
            input_prompt=user_prompt,
        )
        if log_io:
            print(f"[EVAL][Text] Response for {subtask_name}:\n{'-'*40}\n{content}\n{'-'*40}")
        print(f"[EVAL][Text] score={result.score:.0f} passed={result.passed} confidence={result.confidence:.1f}")
        return result

    @staticmethod
    def _parse_json_from_text(text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
        # attempt to extract JSON block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Failed to parse JSON from evaluator output: {text}")

    @staticmethod
    def _build_artifact_summary(bundle: ArtifactBundle) -> str:
        sections: List[str] = [
            f"WORKSPACE ROOT: {bundle.workspace_path}",
            f"EVALSPACE ROOT: {bundle.evalspace_path}",
        ]

        workspace_sections: List[str] = []
        for path in sorted(bundle.workspace_path.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(bundle.workspace_path)
            if _contains_hidden(rel.parts):
                continue
            try:
                content = path.read_text(encoding="utf-8")
                workspace_sections.append(
                    f"[[workspace]] {rel.as_posix()}\n{textwrap.indent(content, '    ')}\n"
                )
            except UnicodeDecodeError:
                size = path.stat().st_size
                workspace_sections.append(
                    f"[[workspace]] {rel.as_posix()} (binary asset, {size} bytes)"
                )

        if bundle.dom_snapshot:
            workspace_sections.append(
                f"[[workspace]] dom_snapshot\n{textwrap.indent(bundle.dom_snapshot, '    ')}\n"
            )

        for log_name, log_content in bundle.debug_logs.items():
            workspace_sections.append(
                f"[[workspace-log]] {log_name}\n{textwrap.indent(log_content, '    ')}\n"
            )

        evalspace_sections: List[str] = []
        if bundle.evalspace_path.exists():
            for path in sorted(bundle.evalspace_path.rglob("*")):
                if not path.is_file():
                    continue
                rel = path.relative_to(bundle.evalspace_path)
                if _contains_hidden(rel.parts):
                    continue
                suffix = path.suffix.lower()
                try:
                    if suffix in {".json", ".log", ".txt", ".html", ".css", ".js"}:
                        content = path.read_text(encoding="utf-8")
                        evalspace_sections.append(
                            f"[[evalspace]] {rel.as_posix()}\n{textwrap.indent(content, '    ')}\n"
                        )
                    else:
                        size = path.stat().st_size
                        evalspace_sections.append(
                            f"[[evalspace]] {rel.as_posix()} (asset, {size} bytes)"
                        )
                except UnicodeDecodeError:
                    size = path.stat().st_size
                    evalspace_sections.append(
                        f"[[evalspace]] {rel.as_posix()} (asset, {size} bytes)"
                    )

        if bundle.metadata:
            metadata_json = json.dumps(bundle.metadata, ensure_ascii=False, indent=2)
            evalspace_sections.append(
                f"[[metadata]]\n{textwrap.indent(metadata_json, '    ')}\n"
            )

        if bundle.screenshot_paths:
            evalspace_sections.append(
                "[[media]] Screenshots: "
                + ", ".join(f"{p.name} ({p.stat().st_size} bytes)" for p in bundle.screenshot_paths)
            )
        if bundle.video_paths:
            evalspace_sections.append(
                "[[media]] Videos: "
                + ", ".join(f"{p.name} ({p.stat().st_size} bytes)" for p in bundle.video_paths)
            )

        return "\n".join(sections + workspace_sections + evalspace_sections)

    @staticmethod
    def _summarize_transcript(transcript: List[Dict[str, Any]]) -> str:
        summaries = []
        for item in transcript[-10:]:
            summaries.append(f"[{item.get('type')}] {item.get('text','')[:120]}")
        return "\n".join(summaries)


class BaseVisionEvaluator:
    def __init__(self, env_config: EnvConfig):
        self.env = env_config

    def evaluate(
        self,
        subtask_name: str,
        rubric: str,
        bundle: ArtifactBundle,
        diagnostics: Dict[str, Any],
        *,
        log_io: bool = False,
    ) -> EvaluatorResult:
        raise NotImplementedError

    @staticmethod
    def _prep_diagnostics(diagnostics: Dict[str, Any]) -> str:
        try:
            return json.dumps(diagnostics, ensure_ascii=False)[:4000]
        except TypeError:
            safe = {}
            for key, value in diagnostics.items():
                try:
                    safe[key] = json.loads(json.dumps(value))
                except (TypeError, ValueError):
                    safe[key] = repr(value)
            return json.dumps(safe, ensure_ascii=False)[:4000]

class GeminiVisionEvaluator(BaseVisionEvaluator):
    def __init__(self, env_config: EnvConfig):
        super().__init__(env_config)
        if genai is None:
            raise ImportError(
                "google-genai package is required when EVAL_VISION_PROVIDER=google"
            )
        if not self.env.google_api_key:
            raise ValueError("GOOGLE_API_KEY must be configured for Google vision evaluation.")
        self.client = genai.Client(api_key=self.env.google_api_key)

    def evaluate(
        self,
        subtask_name: str,
        rubric: str,
        bundle: ArtifactBundle,
        diagnostics: Dict[str, Any],
        *,
        log_io: bool = False,
    ) -> EvaluatorResult:
        files: List[Any] = []
        file_summaries: List[Dict[str, Any]] = []
        for path in bundle.screenshot_paths + bundle.video_paths:
            if path.exists():
                upload = self._upload_and_wait(path)
                files.append(upload)
                file_summaries.append({"path": str(path), "upload_name": getattr(upload, "name", None)})

        prompt = textwrap.dedent(
            f"""
            You are the visual QA lead for AgencyBench task8. Assign an integer score between 0 and 10 inclusive for {subtask_name}'s visual deliverables: 10 = flawless match to rubric, 8 = strong with minor visual deviations, 6 = acceptable but misses notable criteria, 4 = major flaws with partial compliance, 2 = minimal progress, 0 = unusable visuals. Scores must be whole numbers—no decimals.
            Rubric:
            {rubric}

            Beyond the rubric, also judge the front-end's intuitive polish and overall aesthetics. Penalize confusing layouts, off-center or skewed boards, clashing or low-contrast color palettes, awkward naming, inconsistent spacing, or any visual choices that make the interface feel crude. These penalties stack—the more visual issues present, the harsher the deduction, even if each issue alone seems minor. Do not award high scores to submissions that look sloppy or unpleasant to use, even when the rubric does not explicitly call out those flaws.

            Supplementary diagnostics: {self._prep_diagnostics(diagnostics)}

            Respond in JSON with keys: score, confidence, reason. Keep score as an integer 0-10 with no decimals. Confidence must be between 0.0 (not confident) and 1.0 (fully confident) using exactly one decimal place. Reason should briefly explain the score. No additional fields. You MUST output in JSON format.
            """
        ).strip()

        contents: List[Any] = files + [prompt]
        if log_io:
            print(f"[EVAL][Vision] Prompt for {subtask_name}:\n{'-'*40}\n{prompt}\n{'-'*40}")
            if file_summaries:
                print(f"[EVAL][Vision] Files: {json.dumps(file_summaries, ensure_ascii=False)}")
        max_attempts = 20
        response = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.client.models.generate_content(model=self.env.vision_model, contents=contents)
                break
            except genai_errors.ServerError as exc:
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                if status == 503 and attempt < max_attempts:
                    delay = 1
                    print(
                        f"[EVAL][Vision] {self.env.vision_model} overloaded (503). "
                        f"Retrying in {delay}s (attempt {attempt}/{max_attempts})."
                    )
                    time.sleep(delay)
                    continue
                raise
            except httpx.RemoteProtocolError:
                if attempt < max_attempts:
                    delay = 1
                    print(
                        f"[EVAL][Vision] {self.env.vision_model} disconnected mid-request. "
                        f"Retrying in {delay}s (attempt {attempt}/{max_attempts})."
                    )
                    time.sleep(delay)
                    continue
                raise
        else:
            raise RuntimeError(f"{self.env.vision_model} vision scoring failed after retries")
        text = response.text.strip()
        parsed = OpenRouterTextEvaluator._parse_json_from_text(text)
        score_value = parsed.get("score", 0)
        score_int = enforce_int_score(score_value)
        confidence_value = parsed.get("confidence", 0.0)
        try:
            confidence_float = float(confidence_value)
        except (TypeError, ValueError):
            confidence_float = 0.0
        confidence_float = max(0.0, min(1.0, round(confidence_float, 1)))
        passed_flag = score_int >= 6
        result = EvaluatorResult(
            model=self.env.vision_model,
            score=float(score_int),
            passed=passed_flag,
            reason=str(parsed.get("reason", "")),
            confidence=confidence_float,
            raw_response={"text": text},
            input_prompt=prompt,
            input_files=file_summaries,
        )
        if log_io:
            print(f"[EVAL][Vision] Response for {subtask_name}:\n{'-'*40}\n{text}\n{'-'*40}")
        print(f"[EVAL][Vision] score={result.score:.0f} passed={result.passed} confidence={result.confidence:.1f}")
        return result

    def _upload_and_wait(self, path: Path):
        upload = self.client.files.upload(file=str(path))
        name = upload.name
        if not name:
            return upload
        print(f"[EVAL][Vision] Uploaded {path.name}, waiting for ACTIVE state")
        status = upload.state
        while status not in {"ACTIVE", "FAILED"}:
            time.sleep(2)
            upload = self.client.files.get(name=name)
            status = upload.state
            print(f"[EVAL][Vision]   state={status}")
        if status != "ACTIVE":
            raise RuntimeError(f"Uploaded file {path} failed to become ACTIVE (state={status})")
        return upload


class QwenVisionEvaluator(BaseVisionEvaluator):
    def __init__(self, env_config: EnvConfig):
        super().__init__(env_config)
        if not self.env.qwen_api_base or not self.env.qwen_api_key:
            raise ValueError("Qwen vision evaluator requires base URL and API key.")
        self.client = OpenAI(
            api_key=self.env.qwen_api_key,
            base_url=self.env.qwen_api_base.rstrip("/"),
            timeout=120,
        )

    def evaluate(
        self,
        subtask_name: str,
        rubric: str,
        bundle: ArtifactBundle,
        diagnostics: Dict[str, Any],
        *,
        log_io: bool = False,
    ) -> EvaluatorResult:
        prompt = textwrap.dedent(
            f"""
            You are the visual QA lead for AgencyBench task8. Assign an integer score between 0 and 10 inclusive for {subtask_name}'s visual deliverables: 10 = flawless match to rubric, 8 = strong with minor visual deviations, 6 = acceptable but misses notable criteria, 4 = major flaws with partial compliance, 2 = minimal progress, 0 = unusable visuals. Scores must be whole numbers—no decimals.
            Rubric:
            {rubric}

            Beyond the rubric, also judge the front-end's intuitive polish and overall aesthetics. Penalize confusing layouts, off-center or skewed boards, clashing or low-contrast color palettes, awkward naming, inconsistent spacing, or any visual choices that make the interface feel crude. These penalties stack—the more visual issues present, the harsher the deduction, even if each issue alone seems minor. Do not award high scores to submissions that look sloppy or unpleasant to use, even when the rubric does not explicitly call out those flaws.

            Supplementary diagnostics: {self._prep_diagnostics(diagnostics)}

            Respond in JSON with keys: score, confidence, reason. Keep score as an integer 0-10 with no decimals. Confidence must be between 0.0 (not confident) and 1.0 (fully confident) using exactly one decimal place. Reason should briefly explain the score. No additional fields. You MUST output in JSON format.
            """
        ).strip()

        media_blocks: List[Dict[str, Any]] = []
        file_summaries: List[Dict[str, Any]] = []
        for path in bundle.screenshot_paths + bundle.video_paths:
            if not path.exists():
                continue
            try:
                raw = path.read_bytes()
            except OSError as exc:
                print(f"[WARN] Failed to read media {path}: {exc}")
                continue
            encoded = base64.b64encode(raw).decode("utf-8")
            mime_type, _ = mimetypes.guess_type(path.name)
            if not mime_type:
                mime_type = "application/octet-stream"
            data_url = f"data:{mime_type};base64,{encoded}"
            block_type = "image_url" if mime_type.startswith("image/") else "video_url"
            block_entry = {"type": block_type, block_type: {"url": data_url}}
            media_blocks.append(block_entry)
            file_summaries.append({"path": str(path), "mime_type": mime_type, "size": len(raw)})

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a meticulous vision QA assistant. Always reply with JSON containing score, confidence, reason.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}] + media_blocks,
            },
        ]

        payload = {"model": self.env.vision_model, "messages": messages, "temperature": 0}
        if log_io:
            print(f"[EVAL][Vision] Prompt for {subtask_name} via Qwen:\n{'-'*40}\n{prompt}\n{'-'*40}")
            if file_summaries:
                print(f"[EVAL][Vision] Media payload: {json.dumps(file_summaries, ensure_ascii=False)}")

        response = self.client.chat.completions.create(**payload)
        try:
            choice = response.choices[0].message.content  # type: ignore[index]
        except (AttributeError, IndexError) as exc:
            raise RuntimeError(f"Unexpected Qwen vision response format: {response}") from exc

        if isinstance(choice, list):
            parts = []
            for item in choice:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            content = "\n".join(parts).strip()
        else:
            content = str(choice).strip()

        parsed = OpenRouterTextEvaluator._parse_json_from_text(content)
        score_value = parsed.get("score", 0)
        score_int = enforce_int_score(score_value)
        confidence_value = parsed.get("confidence", 0.0)
        try:
            confidence_float = float(confidence_value)
        except (TypeError, ValueError):
            confidence_float = 0.0
        confidence_float = max(0.0, min(1.0, round(confidence_float, 1)))
        passed_flag = score_int >= 6

        if log_io:
            print(f"[EVAL][Vision] Response for {subtask_name} via Qwen:\n{'-'*40}\n{content}\n{'-'*40}")

        print(f"[EVAL][Vision] score={score_int:.0f} passed={passed_flag} confidence={confidence_float:.1f}")
        return EvaluatorResult(
            model=self.env.vision_model,
            score=float(score_int),
            passed=passed_flag,
            reason=str(parsed.get("reason", "")),
            confidence=confidence_float,
            raw_response={"response": response.model_dump()},
            input_prompt=prompt,
            input_files=file_summaries,
        )


def build_vision_evaluator(env_config: EnvConfig) -> BaseVisionEvaluator:
    provider = env_config.vision_provider
    if provider == "google":
        return GeminiVisionEvaluator(env_config)
    if provider == "qwen":
        return QwenVisionEvaluator(env_config)
    raise ValueError(f"Unsupported vision evaluator provider: {provider}")


###############################################################################
# Artifact collection logic per subtask
###############################################################################


class ArtifactCollector:
    def __init__(
        self,
        sandbox_base_url: str,
        model_slug: str,
        *,
        visualize: bool = False,
        timeout_sec: int = 20,
    ):
        self.sandbox_base_url = sandbox_base_url
        self.model_slug = model_slug
        self.visualize = visualize
        self.capture_timeout_sec = max(0, timeout_sec)

    async def collect(
        self,
        subtask_name: str,
        attempt_index: int,
        workspace_path: Path,
        evalspace_path: Path,
        rubric: str,
    ) -> ArtifactBundle:
        bundle = ArtifactBundle(attempt_index=attempt_index, workspace_path=workspace_path, evalspace_path=evalspace_path)
        sandbox = SandboxWorkspace(
            self.sandbox_base_url,
            self.model_slug,
            subtask_name,
            attempt_index,
            visualize=self.visualize,
        )
        sandbox.prepare_remote_workspace()
        sandbox.sync_local_workspace(workspace_path)

        try:
            html_entry = "index.html"
            interaction_plan = self._plan_for_subtask(subtask_name)
            record_video = interaction_plan.get("record_video", False) if interaction_plan else False
            screenshot_names = interaction_plan.get("screenshots", ["capture.png"]) if interaction_plan else ["capture.png"]

            capture_coro = sandbox.capture_screenshots(
                html_entry=html_entry,
                output_dir=evalspace_path,
                names=screenshot_names,
                interaction_plan=interaction_plan.get("steps") if interaction_plan else None,
                record_video=record_video,
                video_name=interaction_plan.get("video_name", "capture.webm"),
            )
            try:
                if self.capture_timeout_sec:
                    screenshots, video_path, diagnostics = await asyncio.wait_for(
                        capture_coro, timeout=self.capture_timeout_sec
                    )
                else:
                    screenshots, video_path, diagnostics = await capture_coro
            except asyncio.TimeoutError:
                diagnostics = {
                    "errors": [
                        f"Sandbox capture timed out after {self.capture_timeout_sec}s; proceeding without media",
                    ]
                }
                screenshots = []
                video_path = None
                print(
                    f"[SANDBOX][WARN] capture_screenshots exceeded {self.capture_timeout_sec}s; continuing"
                )
        finally:
            sandbox.cleanup_remote_workspace()

        bundle.screenshot_paths.extend(screenshots)
        if video_path:
            bundle.video_paths.append(video_path)
        bundle.metadata.update(diagnostics)

        # DOM snapshot via local HTML
        html_path = workspace_path / "index.html"
        if html_path.exists():
            bundle.dom_snapshot = html_path.read_text(encoding="utf-8", errors="ignore")

        # Session log if present
        log_path = workspace_path / "session.log"
        if log_path.exists():
            bundle.debug_logs["session.log"] = log_path.read_text(encoding="utf-8", errors="ignore")

        # Diagnostics JSONs
        for extra in ("setup.json", "hard_cases.json"):
            extra_path = workspace_path / extra
            if extra_path.exists():
                bundle.debug_logs[extra] = extra_path.read_text(encoding="utf-8", errors="ignore")

        if bundle.debug_logs:
            bundle.metadata.setdefault("debug_logs", bundle.debug_logs)

        return bundle

    def _plan_for_subtask(self, subtask_name: str) -> Dict[str, Any]:
        plans: Dict[str, Dict[str, Any]] = {
            "subtask1": {
                "steps": [
                    {"action": "wait", "duration": 800},
                    {
                        "action": "eval-js",
                        "expression": "(() => { if (window.game && window.game.initializeGrid) { window.game.initializeGrid(); window.game.initializeGrid(); } return window.game && window.game.describeLayout ? window.game.describeLayout() : null; })()",
                        "label": "layout",
                    },
                ],
                "screenshots": ["layout.png"],
                "record_video": False,
            },
            "subtask2": {
                "steps": [
                    {"action": "wait", "duration": 600},
                    {
                        "action": "eval-js",
                        "expression": "(() => { if (!window.game || !window.game.setGridState) return 'missing-setGridState'; window.game.setGridState([[2,2,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]); if (window.game.setSpawnQueue) { window.game.setSpawnQueue([{ value: 2, row: 3, col: 3 }]); } return window.game.debugState ? window.game.debugState() : 'seeded'; })()",
                        "label": "seedState",
                    },
                    {
                        "action": "click-board",
                        "selector": "#grid",
                        "clicks": [
                            {"x_ratio": 0.5, "y_ratio": 0.5, "wait_ms": 200},
                        ],
                    },
                    {"action": "press-keys", "keys": ["ArrowRight"], "wait_ms": 280},
                    {"action": "wait", "duration": 350},
                    {
                        "action": "eval-js",
                        "expression": "window.game && window.game.debugState ? window.game.debugState() : null",
                        "label": "debugState",
                    },
                    {
                        "action": "eval-js",
                        "expression": "document.querySelector('#status-bar') ? document.querySelector('#status-bar').textContent : null",
                        "label": "statusText",
                    },
                ],
                "screenshots": ["after_move.png"],
                "record_video": True,
                "video_name": "merge_turns.webm",
            },
            "subtask3": {
                "steps": [
                    {"action": "wait", "duration": 600},
                    {
                        "action": "eval-js",
                        "expression": "(() => { if (!window.game || !window.game.setGridState) return 'missing-setGridState'; window.game.setGridState([[1024,1024,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]); return window.game.debugState ? window.game.debugState() : null; })()",
                        "label": "milestoneSeed",
                    },
                    {
                        "action": "eval-js",
                        "expression": "(() => { if (!window.game || !window.game.applyScript) return 'missing-applyScript'; return window.game.applyScript(['left']); })()",
                        "label": "scriptResult",
                    },
                    {
                        "action": "eval-js",
                        "expression": "window.game && window.game.checkMilestone ? window.game.checkMilestone() : null",
                        "label": "milestone",
                    },
                    {
                        "action": "eval-js",
                        "expression": "(() => { const banner = document.querySelector('#milestone-banner'); return banner ? banner.textContent : null; })()",
                        "label": "milestoneBanner",
                    },
                    {"action": "click-element", "selector": "button[data-action=\"undo\"]", "wait_ms": 240},
                    {"action": "click-element", "selector": "button[data-action=\"redo\"]", "wait_ms": 240},
                    {"action": "click-element", "selector": "button[data-action=\"replay\"]", "wait_ms": 240},
                    {"action": "wait", "duration": 2600},
                    {
                        "action": "eval-js",
                        "expression": "window.game && window.game.exportLog ? window.game.exportLog() : null",
                        "label": "exportedLog",
                    },
                ],
                "screenshots": ["milestone.png"],
                "record_video": True,
                "video_name": "replay2048.webm",
            },
            "subtask4": {
                "steps": [
                    {"action": "wait", "duration": 600},
                    {
                        "action": "eval-js",
                        "expression": "(() => { if (!window.game || !window.game.setGridState) return 'missing-setGridState'; window.game.setGridState([[4,2,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]); return 'seeded'; })()",
                        "label": "persistenceSeed",
                    },
                    {"action": "click-element", "selector": "#save-run", "wait_ms": 260},
                    {
                        "action": "eval-js",
                        "expression": "(() => { if (!window.game || !window.game.serializeState) return 'missing-serializeState'; const serialized = window.game.serializeState(); const payload = typeof serialized === 'string' ? serialized : JSON.stringify(serialized); window.__persistedState = payload; const textarea = document.querySelector('#state-json'); const storage = window.localStorage ? window.localStorage.getItem('2048-state') : null; return { payload, textarea: textarea ? textarea.value : null, storage }; })()",
                        "label": "savedState",
                    },
                    {"action": "click-element", "selector": "#reset-board", "wait_ms": 260},
                    {
                        "action": "eval-js",
                        "expression": "window.game && window.game.debugState ? window.game.debugState() : null",
                        "label": "stateAfterReset",
                    },
                    {"action": "click-element", "selector": "#load-run", "wait_ms": 260},
                    {"action": "wait", "duration": 400},
                    {
                        "action": "eval-js",
                        "expression": "window.game && window.game.debugState ? window.game.debugState() : null",
                        "label": "stateAfterLoad",
                    },
                    {
                        "action": "eval-js",
                        "expression": "window.game && window.game.getStats ? window.game.getStats() : null",
                        "label": "statsSnapshot",
                    },
                ],
                "screenshots": ["persistence.png"],
                "record_video": False,
                "video_name": "persistence_capture.webm",
            },
            "subtask5": {
                "steps": [
                    {"action": "wait", "duration": 600},
                    {
                        "action": "eval-js",
                        "expression": "(() => { if (!window.game || !window.game.loadScenario) return 'missing-loadScenario'; return window.game.loadScenario(0); })()",
                        "label": "scenario0",
                    },
                    {"action": "wait", "duration": 400},
                    {
                        "action": "eval-js",
                        "expression": "window.game && window.game.getDiagnostics ? window.game.getDiagnostics() : null",
                        "label": "diagnosticsAfterScenario0",
                    },
                    {
                        "action": "eval-js",
                        "expression": "(() => { if (!window.game || !window.game.summarizeScenario) return 'missing-summarizeScenario'; return window.game.summarizeScenario(0); })()",
                        "label": "summaryScenario0",
                    },
                    {"action": "click-element", "selector": "#toggle-diagnostics", "wait_ms": 260},
                    {
                        "action": "eval-js",
                        "expression": "(async () => { if (!window.game || !window.game.playScenario) return 'missing-playScenario'; await window.game.playScenario(1, { intervalMs: 220 }); return 'played'; })()",
                        "label": "scenario1Playback",
                    },
                    {"action": "wait", "duration": 2600},
                    {
                        "action": "eval-js",
                        "expression": "window.game && window.game.getDiagnostics ? window.game.getDiagnostics() : null",
                        "label": "diagnosticsAfterScenario1",
                    },
                    {
                        "action": "eval-js",
                        "expression": "(() => { const table = document.querySelector('#diagnostics table'); if (!table) return null; return { headers: Array.from(table.querySelectorAll('th')).map(th => th.textContent), rows: Array.from(table.querySelectorAll('tbody tr')).map(tr => Array.from(tr.querySelectorAll('td')).map(td => td.textContent)) }; })()",
                        "label": "diagnosticsTable",
                    },
                    {
                        "action": "eval-js",
                        "expression": "(() => { if (!window.game || !window.game.estimateLoad) return 'missing-estimateLoad'; const readings = []; for (let i = 0; i < 5; i += 1) { readings.push(window.game.estimateLoad()); } return readings; })()",
                        "label": "loadSamples",
                    },
                ],
                "screenshots": ["diagnostics.png"],
                "record_video": True,
                "video_name": "scenario_preview.webm",
            },
        }
        return plans.get(
            subtask_name,
            {"steps": [], "screenshots": ["capture.png"], "record_video": False, "video_name": "capture.webm"},
        )


###############################################################################
# Meta logging
###############################################################################


class MetaLogger:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.data: Dict[str, Any] = {"subtasks": {}}
        if self.output_path.exists():
            try:
                self.data = json.loads(self.output_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

    def append_attempt(self, subtask: str, record: AttemptRecord) -> None:
        entry = self.data.setdefault("subtasks", {}).setdefault(subtask, {"attempts": []})
        entry["attempts"].append(
            {
                "attempt_index": record.attempt_index,
                "prompt": record.prompt,
                "agent_messages": record.agent_messages,
                "assistant_text": record.assistant_text,
                "artifacts": {
                    "screenshots": [str(p) for p in record.artifacts.screenshot_paths],
                    "videos": [str(p) for p in record.artifacts.video_paths],
                    "debug_logs": record.artifacts.debug_logs,
                    "metadata": record.artifacts.metadata,
                },
                "text_result": record.text_result.__dict__,
                "vision_result": record.vision_result.__dict__,
                "status": record.status,
            }
        )
        self.output_path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")


###############################################################################
# Coordinator
###############################################################################


class EvaluationCoordinator:
    def __init__(self, env_config: EnvConfig, description_path: Path, *, visualize: bool = False):
        self.env = env_config
        self.description = read_json(description_path)
        self.subtasks = [f"subtask{i}" for i in range(1, self.description.get("subtask_count", 5) + 1)]
        base_task_dir = Path(__file__).resolve().parent
        self.repo_root = base_task_dir.parent
        self.run_root = ensure_directory(base_task_dir / self.env.target_model_name)
        for subtask_name in self.subtasks:
            subtask_root = ensure_directory(self.run_root / subtask_name)
            ensure_directory(subtask_root / "workspace")
            ensure_directory(subtask_root / "evalspace")

        self.collector = ArtifactCollector(
            env_config.sandbox_base_url,
            env_config.target_model_name,
            visualize=visualize,
            timeout_sec=env_config.sandbox_timeout_sec,
        )
        self.text_evaluator = OpenRouterTextEvaluator(env_config)
        self.vision_evaluator = build_vision_evaluator(env_config)
        self.meta_logger = MetaLogger(self.run_root / "meta_eval.json")
        self.visualize = visualize

    async def run(self) -> None:
        ensure_conda_env(self.env.conda_env)
        session = SiiAgentSession(self.env)
        try:
            session.set_workspace(self.run_root, reset_history=True)
            previous_workspace_seed: Optional[Path] = None

            banner = "=" * 110
            print(banner)
            print(f"[EVAL][MODEL ] ▶ {self.env.sii_target_model}  (run folder: {self.run_root})")
            print(f"[EVAL][TEXT  ] → {self.env.eval_text_model}")
            print(f"[EVAL][VISION] → {self.env.vision_model} [{self.env.vision_provider}]")
            print(f"[EVAL][ATTEMPTS] → max {self.env.max_attempts} per subtask")
            print(banner)

            total_subtasks = len(self.subtasks)

            for index, subtask_name in enumerate(self.subtasks, start=1):
                prompt = self.description[subtask_name]
                subtask_root = self.run_root / subtask_name
                workspace_root = subtask_root / "workspace"
                eval_root = subtask_root / "evalspace"
                ensure_directory(workspace_root)
                ensure_directory(eval_root)

                subtask_banner = "-" * 100
                print(subtask_banner)
                print(
                    f"[TASK][{subtask_name.upper()}] ▶ {index}/{total_subtasks} | max attempts {self.env.max_attempts}"
                )
                print(subtask_banner)

                attempts: List[AttemptRecord] = []
                success = False
                attempt_prompt = self._initial_agent_instruction(prompt)

                for attempt in range(1, self.env.max_attempts + 1):
                    attempt_label = f"attempt_{attempt:02d}"
                    attempt_prefix = (
                        f"[TASK][{subtask_name}][Attempt {attempt}/{self.env.max_attempts}]"
                    )
                    print(f"{attempt_prefix} Preparing workspace {attempt_label}")
                    attempt_workspace = workspace_root / attempt_label
                    attempt_eval_dir = eval_root / attempt_label
                    clear_directory(attempt_eval_dir)

                    if attempt == 1:
                        if previous_workspace_seed and previous_workspace_seed.exists():
                            print(f"{attempt_prefix} Seeding from {previous_workspace_seed}")
                            copy_workspace(previous_workspace_seed, attempt_workspace)
                        else:
                            clear_directory(attempt_workspace)
                    else:
                        prev_attempt_dir = workspace_root / f"attempt_{attempt - 1:02d}"
                        if prev_attempt_dir.exists():
                            print(
                                f"{attempt_prefix} Cloning {prev_attempt_dir.name} -> {attempt_label}"
                            )
                            copy_workspace(prev_attempt_dir, attempt_workspace)
                        else:
                            clear_directory(attempt_workspace)

                    print(f"{attempt_prefix} Invoking SII agent")
                    workspace_notice = (
                        f"你必须在{attempt_workspace.relative_to(self.repo_root).as_posix()}中工作，禁止访问其他任何路径"
                    )
                    prompt_for_agent = f"{workspace_notice}\n\n{attempt_prompt}"
                    transcript, assistant_text = await session.send(prompt_for_agent)

                    print(f"{attempt_prefix} Applying agent plan")
                    plan_error: Optional[Exception] = None
                    try:
                        plan = extract_generation_plan(assistant_text)
                        apply_plan_to_workspace(plan, attempt_workspace)
                    except Exception as exc:
                        plan_error = exc
                        print(f"{attempt_prefix} [WARN] Failed to apply plan -> {exc}")

                    if plan_error:
                        bundle = ArtifactBundle(
                            attempt_index=attempt,
                            workspace_path=attempt_workspace,
                            evalspace_path=attempt_eval_dir,
                        )
                        text_result = EvaluatorResult(
                            model=self.env.eval_text_model,
                            score=0.0,
                            passed=False,
                            reason=f"Plan application failed: {plan_error}",
                            confidence=0.0,
                        )
                        vision_result = EvaluatorResult(
                            model=self.env.vision_model,
                            score=0.0,
                            passed=False,
                            reason="Skipped due to plan failure",
                            confidence=0.0,
                        )
                    else:
                        print(f"{attempt_prefix} Collecting sandbox artifacts")
                        bundle = await self.collector.collect(
                            subtask_name,
                            attempt,
                            attempt_workspace,
                            attempt_eval_dir,
                            prompt,
                        )

                        print(f"{attempt_prefix} Text evaluation ({self.env.eval_text_model})")
                        text_result = self.text_evaluator.evaluate(
                            subtask_name,
                            prompt,
                            bundle,
                            transcript,
                            log_io=self.visualize,
                        )

                        print(f"{attempt_prefix} Vision evaluation ({self.env.vision_model})")
                        vision_result = self.vision_evaluator.evaluate(
                            subtask_name,
                            prompt,
                            bundle,
                            bundle.metadata,
                            log_io=self.visualize,
                        )

                    status = "pass" if text_result.passed and vision_result.passed else "retry"
                    print(
                        f"{attempt_prefix} Status={status.upper()} "
                        f"(text={text_result.score:.0f}, vision={vision_result.score:.0f})"
                    )

                    record = AttemptRecord(
                        attempt_index=attempt,
                        prompt=prompt_for_agent,
                        agent_messages=transcript,
                        assistant_text=assistant_text,
                        artifacts=bundle,
                        text_result=text_result,
                        vision_result=vision_result,
                        status=status,
                    )
                    self.meta_logger.append_attempt(subtask_name, record)
                    attempts.append(record)

                    if status == "pass":
                        success = True
                        print(f"{attempt_prefix} PASS")
                        break

                    feedback_prompt = self._build_feedback_prompt(
                        prompt, text_result, vision_result, assistant_text
                    )
                    attempt_prompt = feedback_prompt
                    print(f"{attempt_prefix} Prepared feedback for next attempt")

                if not success:
                    print(
                        f"[TASK][{subtask_name}] Exhausted {self.env.max_attempts} attempts without success"
                    )
                else:
                    print(
                        f"[TASK][{subtask_name}] Completed in {len(attempts)} attempt(s)"
                    )

                previous_workspace_seed = attempts[-1].artifacts.workspace_path if attempts else None
                if previous_workspace_seed:
                    print(
                        f"[TASK][{subtask_name}] Cached workspace for next subtask -> {previous_workspace_seed}"
                    )
        finally:
            session.restore_cwd()

    def _initial_agent_instruction(self, base_prompt: str) -> str:
        format_notice = textwrap.dedent(
            """
            You must produce the deliverables by returning ONLY strict JSON.
            Do not execute shell commands or use tools. Instead, synthesize the
            file contents directly. JSON schema:
            {
              "files": [
                {"path": "relative/path", "encoding": "utf-8"|"base64", "content": "..."}
              ],
              "notes": "optional guidance"
            }
            Every file listed will overwrite the file at that path inside the
            current workspace. Include all required deliverables each time.
            For binary assets, set encoding="base64" and provide base64 data.
            Required deliverables for every response:
              - index.html
              - styles.css
              - app.js
            Include session.log and JSON fixtures (setup.json, hard_cases.json)
            when the subtask requires them. Do not create unrelated files.
            Example valid response:
            {
              "files": [
                {"path": "index.html", "encoding": "utf-8", "content": "<!DOCTYPE html>..."}
              ],
              "notes": ""
            }
            Never reply with plain text like "Task completed". If no files need changes,
            still return the JSON object with an empty "files" array. Respond with JSON
            only, no extra commentary or code fences.
            """
        ).strip()
        return f"{base_prompt}\n\n{format_notice}"

    def _build_feedback_prompt(
        self,
        base_prompt: str,
        text_result: EvaluatorResult,
        vision_result: EvaluatorResult,
        last_response: str,
    ) -> str:
        response_excerpt = last_response.strip()
        if len(response_excerpt) > 800:
            response_excerpt = response_excerpt[:800] + " ..."
        feedback = textwrap.dedent(
            f"""
            Previous attempt was rejected. Update the existing files accordingly
            using the same JSON-only response format as before. Feedback:
            - Text evaluator ({text_result.model}) score {text_result.score:.1f}, passed={text_result.passed}, reason: {text_result.reason}
            - Vision evaluator ({vision_result.model}) score {vision_result.score:.1f}, passed={vision_result.passed}, reason: {vision_result.reason}

            Return updated files covering all deliverables. You must reply with a JSON
            object matching the required schema—no prose or code fences. If you leave out
            files, the evaluator will fail again. Last response snippet:
            {response_excerpt or "<empty>"}
            """
        ).strip()
        return f"{self._initial_agent_instruction(base_prompt)}\n\n{feedback}"


###############################################################################
# Entry point
###############################################################################


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run automated evaluation for AgencyBench task8")
    parser.add_argument("--env", default=".env", help="Path to environment configuration file")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open local sandbox visualization windows during automation",
    )
    args = parser.parse_args(argv)

    env_path = Path(args.env)
    env_data = load_env_file(env_path)
    env_config = EnvConfig.from_env(env_data)
    env_config.inject_defaults()

    description_path = Path(__file__).resolve().parent / "description.json"
    coordinator = EvaluationCoordinator(env_config, description_path, visualize=args.visualize)

    asyncio.run(coordinator.run())
    print("[DONE] Evaluation pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
