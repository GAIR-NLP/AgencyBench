#!/usr/bin/env python3
"""Automated multi-subtask evaluator for AgencyBench task26.

This script mirrors the orchestration patterns used by task21/task29 and adapts
them to the filesystem refactoring scenario described in description.json.
Behavior highlights:

* Loads credentials/environment variables from task26/.env.
* Creates a model-specific run directory (task26/<model_slug>/), copies the
  original ``desktop/`` tree into that directory, and prepares an empty
  ``desktop_2/`` folder where the agent must build ``workspace_v2`` without
  touching the template files.
* Opens a single sii-agent-sdk session that drives all five subtasks in order.
  Each subtask receives at most ``SUBTASK_ATTEMPT_LIMIT`` attempts (default: 2).
* After every attempt the evaluator inspects the workspace to determine whether
  the subtask-specific acceptance criteria are met. Failures produce feedback
  that is appended to the next prompt so the agent can fix the issues.
* Once a subtask ultimately fails, the remaining ones are skipped. The final
  run summary (including attempt-level metadata) is persisted to
  ``task26/<model_slug>/meta_eval.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from sii_agent_sdk import AssistantMessage, CompletedMessage, Message, SiiAgentOptions, TextBlock
from sii_agent_sdk._internal.event_logger import log_message
from sii_agent_sdk._internal.message_parser import parse_message
from sii_agent_sdk.bridge import BridgeProcess
from sii_agent_sdk.errors import BridgeConnectionError, ProcessError
from sii_agent_sdk.query import _message_to_session_turns, _raise_appropriate_error, validate_auth_config
from sii_agent_sdk.session_state import ConversationTurn, SessionState


###############################################################################
# Asyncio compatibility helpers (copied from task29 / task21)
###############################################################################


def _suppress_event_loop_closed_errors() -> None:
    """Work around noisy warnings triggered by the SII Agent SDK on exit."""

    try:
        import asyncio.base_subprocess as _base_subprocess
    except ImportError:
        return

    original_del = getattr(_base_subprocess.BaseSubprocessTransport, "__del__", None)
    if original_del is None or getattr(original_del, "_sii_patched", False):
        return

    def _patched_del(self, *args, **kwargs):
        try:
            original_del(self, *args, **kwargs)
        except RuntimeError as exc:  # pragma: no cover - defensive patch
            if "Event loop is closed" in str(exc):
                return
            raise

    setattr(_patched_del, "_sii_patched", True)
    _base_subprocess.BaseSubprocessTransport.__del__ = _patched_del


_suppress_event_loop_closed_errors()


def increase_asyncio_stream_limit(min_limit: int = 8 * 1024 * 1024) -> None:
    """Ensure asyncio streams can handle large JSON lines emitted by the bridge."""

    try:
        import asyncio.streams as streams
    except ImportError:
        return

    current = getattr(streams, "_DEFAULT_LIMIT", None)
    if isinstance(current, int) and current < min_limit:
        streams._DEFAULT_LIMIT = min_limit


increase_asyncio_stream_limit()


###############################################################################
# Utility helpers
###############################################################################


Validator = Callable[[Path], Tuple[bool, str]]


def load_env_file(env_path: Path) -> Dict[str, str]:
    if not env_path.exists():
        raise FileNotFoundError(f"Missing env file: {env_path}")

    parsed: Dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        parsed[key] = value
        os.environ[key] = value
    return parsed


def derive_model_name(identifier: str) -> str:
    raw = identifier.strip()
    if "/" in raw:
        raw = raw.split("/")[-1]
    sanitized = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in raw)
    sanitized = sanitized.strip("._-")
    return sanitized or "model"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stringify_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


###############################################################################
# Environment configuration
###############################################################################


@dataclass
class EnvConfig:
    username: str
    password: str
    agent_api_key: str
    agent_api_base: str
    target_model: str
    auth_type: str
    system_prompt: str
    max_turns: int
    attempt_limit: int
    model_slug: str = field(init=False)

    def __post_init__(self) -> None:
        self.model_slug = derive_model_name(self.target_model)

    @classmethod
    def from_env(cls, data: Dict[str, str]) -> "EnvConfig":
        def require(key: str) -> str:
            value = (data.get(key) or os.environ.get(key) or "").strip()
            if not value:
                raise ValueError(f"Environment variable '{key}' must be set in task26/.env")
            return value

        attempt_limit = int(data.get("SUBTASK_ATTEMPT_LIMIT", "2"))
        return cls(
            username=require("SII_USERNAME"),
            password=require("SII_PASSWORD"),
            agent_api_key=require("SII_AGENT_API_KEY"),
            agent_api_base=data.get("SII_AGENT_API_BASE_URL", "https://openrouter.ai/api/v1").strip(),
            target_model=require("SII_TARGET_MODEL"),
            auth_type=data.get("SII_AUTH_TYPE", "USE_OPENAI_WITH_SII_TOOLS").strip(),
            system_prompt=data.get("SII_SYSTEM_PROMPT", "").strip(),
            max_turns=int(data.get("SII_MAX_TURNS", "60")),
            attempt_limit=attempt_limit,
        )

    def inject_defaults(self) -> None:
        os.environ.setdefault("SII_AGENT_API_KEY", self.agent_api_key)
        os.environ.setdefault("SII_AGENT_API_BASE_URL", self.agent_api_base)
        os.environ.setdefault("SII_TARGET_MODEL", self.target_model)
        os.environ.setdefault("SII_AUTH_TYPE", self.auth_type)
        os.environ.setdefault("SII_SYSTEM_PROMPT", self.system_prompt)
        os.environ.setdefault("SII_MAX_TURNS", str(self.max_turns))
        os.environ.setdefault("SII_USERNAME", self.username)
        os.environ.setdefault("SII_PASSWORD", self.password)
        os.environ.setdefault("MODEL_API_KEY", os.environ.get("MODEL_API_KEY", self.agent_api_key))
        os.environ.setdefault("MODEL_API_BASE_URL", os.environ.get("MODEL_API_BASE_URL", self.agent_api_base))
        os.environ.setdefault("MODEL_API_MODEL", os.environ.get("MODEL_API_MODEL", self.target_model))


###############################################################################
# Attempt metadata
###############################################################################


@dataclass
class AttemptOutcome:
    attempt_index: int
    success: bool
    message: str
    assistant_response: str


###############################################################################
# SII Agent orchestration
###############################################################################


class SiiAgentRunner:
    """Persistent bridge runner that keeps a single SII Agent session alive."""

    def __init__(self, env_config: EnvConfig, agent_root: Path):
        self.env = env_config
        self.agent_root = ensure_directory(agent_root).resolve()
        self._options: Optional[SiiAgentOptions] = None
        self._bridge: Optional[BridgeProcess] = None
        self._session_state = SessionState()
        self._last_completion_metadata: Optional[Dict[str, Any]] = None
        self._reset_notice: Optional[str] = None

    def _build_options(self) -> SiiAgentOptions:
        env_vars = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", self.env.agent_api_key),
            "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL", self.env.agent_api_base),
            "SII_OPENAI_API_KEY": self.env.agent_api_key,
            "SII_OPENAI_BASE_URL": self.env.agent_api_base,
            "SII_OPENAI_MODEL": self.env.target_model,
            "SII_USERNAME": self.env.username,
            "SII_PASSWORD": self.env.password,
        }
        options = SiiAgentOptions(
            system_prompt=self.env.system_prompt,
            max_turns=self.env.max_turns,
            auth_type=self.env.auth_type,
            cwd=str(self.agent_root),
            yolo=True,
            allowed_tools=[],
            model=self.env.target_model,
            env=env_vars,
            log_events=False,
        )
        validate_auth_config(options.auth_type or "USE_SII", options.env)
        return options

    async def _ensure_bridge(self) -> None:
        if self._bridge and self._bridge.is_ready():
            return

        if self._options is None:
            self._options = self._build_options()

        self._bridge = BridgeProcess(self._options)
        os.environ["SII_SDK_ENTRYPOINT"] = "python-sdk"
        prev_cwd = Path.cwd()
        try:
            os.chdir(self.agent_root)
            await self._bridge.start()
        finally:
            os.chdir(prev_cwd)

    async def send(self, prompt: str) -> Tuple[List[Dict[str, Any]], str]:
        try:
            return await self._send_once(prompt)
        except (ProcessError, BridgeConnectionError) as exc:
            await self._handle_bridge_crash(exc)
            try:
                return await self._send_once(prompt)
            except (ProcessError, BridgeConnectionError) as exc2:
                await self._handle_bridge_crash(exc2)
                raise

    async def _send_once(self, prompt: str) -> Tuple[List[Dict[str, Any]], str]:
        await self._ensure_bridge()
        assert self._bridge is not None and self._options is not None

        request_payload: Dict[str, Any] = {
            "prompt": prompt,
            "options": self._options.to_dict(),
        }
        history_payload = self._session_state.to_bridge_history()
        if history_payload:
            request_payload["history"] = history_payload
        if self._session_state.session_id:
            request_payload["session_id"] = self._session_state.session_id
        if self._session_state.environment_context:
            request_payload["environment_context"] = self._session_state.environment_context

        prev_cwd = Path.cwd()
        transcript: List[Dict[str, Any]] = []
        assistant_chunks: List[str] = []
        message_index = 0
        try:
            os.chdir(self.agent_root)
            await self._bridge.send_request("query", request_payload)
            self._session_state.append(ConversationTurn(role="user", content=prompt))

            async for event in self._bridge.receive_events():
                event_type = event.get("type")
                if event_type == "error":
                    _raise_appropriate_error(event.get("error", {}) or {})

                message = parse_message(event)
                if message:
                    message_index += 1
                    payload = self._normalize_message(message, message_index)
                    transcript.append(payload)
                    if not isinstance(message, CompletedMessage):
                        log_message(message)
                    for turn in _message_to_session_turns(message, self._session_state):
                        self._session_state.append(turn)
                    if payload.get("type") == "assistant" and payload.get("text"):
                        assistant_chunks.append(payload["text"])
                    elif payload.get("type") == "tool_result":
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

                if event_type == "completed":
                    metadata = event.get("metadata", {}) or {}
                    session_id = metadata.get("session_id")
                    if session_id:
                        self._session_state.session_id = session_id
                    environment_context = metadata.get("environment_context")
                    if environment_context:
                        self._session_state.environment_context = environment_context
                    self._last_completion_metadata = metadata
                    break

        finally:
            os.chdir(prev_cwd)

        assistant_text = "\n".join(part.strip() for part in assistant_chunks if part).strip()
        return transcript, assistant_text

    async def _handle_bridge_crash(self, exc: Exception) -> None:
        reason = str(exc).strip() or "bridge exited unexpectedly"
        print(f"[TASK26] SII bridge exited unexpectedly: {reason}. Restarting session...")
        self._reset_notice = reason
        if self._bridge:
            try:
                await self._bridge.close()
            except Exception:
                pass
        self._bridge = None
        self._options = None
        self._session_state.clear()

    def consume_reset_notice(self) -> Optional[str]:
        note = self._reset_notice
        self._reset_notice = None
        return note

    async def shutdown(self, emit_summary: bool = True) -> None:
        if emit_summary and self._last_completion_metadata:
            log_message(CompletedMessage(type="completed", metadata=self._last_completion_metadata))
        if self._bridge:
            await self._bridge.close()
            self._bridge = None

    @staticmethod
    def _normalize_message(message: Message, index: int) -> Dict[str, Any]:
        if isinstance(message, AssistantMessage):
            texts = [block.text for block in message.content if isinstance(block, TextBlock)]
            return {"index": index, "type": "assistant", "text": "\n".join(texts)}
        data = message.__dict__.copy()
        data["index"] = index
        data.setdefault("type", data.get("role", "unknown"))
        return data


###############################################################################
# Subtask validation helpers
###############################################################################


def _workspace_v2_root(workspace: Path) -> Path:
    return workspace / "desktop_2" / "workspace_v2"


def verify_subtask1(workspace: Path) -> Tuple[bool, str]:
    base = _workspace_v2_root(workspace)
    required = [
        base,
        base / "dev_bundle",
        base / "dev_bundle" / "tests",
        base / "dev_bundle" / "source",
        base / "data_warehouse",
        base / "data_warehouse" / "legacy_archives",
        base / "data_warehouse" / "active_datasets",
        base / "knowledge_base",
    ]

    missing = [str(path.relative_to(workspace)) for path in required if not path.is_dir()]
    if missing:
        return False, f"Missing directories: {', '.join(missing)}"
    return True, "workspace_v2 directory skeleton exists."


def verify_subtask2(workspace: Path) -> Tuple[bool, str]:
    base = _workspace_v2_root(workspace) / "dev_bundle"
    tests_dir = base / "tests"
    source_dir = base / "source"
    if not tests_dir.is_dir() or not source_dir.is_dir():
        return False, "dev_bundle/tests or dev_bundle/source missing."

    expected_tests = {"debug_utils.py", "test_runner.py"}
    expected_source = {"train_model.py", "settings.py", "budget_calculator.py"}

    found_tests = {f.name for f in tests_dir.glob("*.py")}
    found_source = {f.name for f in source_dir.glob("*.py")}

    missing_tests = expected_tests - found_tests
    missing_source = expected_source - found_source
    misplaced_tests = expected_source & found_tests
    misplaced_source = expected_tests & found_source

    problems: List[str] = []
    if missing_tests:
        problems.append(f"Missing in tests/: {sorted(missing_tests)}")
    if missing_source:
        problems.append(f"Missing in source/: {sorted(missing_source)}")
    if misplaced_tests:
        problems.append(f"Source-only files wrongly placed in tests/: {sorted(misplaced_tests)}")
    if misplaced_source:
        problems.append(f"Test/debug files wrongly placed in source/: {sorted(misplaced_source)}")

    if problems:
        return False, "; ".join(problems)
    return True, "Python files relocated into correct dev_bundle folders."


def verify_subtask3(workspace: Path) -> Tuple[bool, str]:
    base = _workspace_v2_root(workspace) / "data_warehouse"
    legacy_dir = base / "legacy_archives"
    active_dir = base / "active_datasets"
    if not legacy_dir.is_dir() or not active_dir.is_dir():
        return False, "data_warehouse subdirectories missing."

    expected_legacy = {"raw_data.csv", "old_results.csv", "inventory_list.csv"}
    expected_active = {"progress.csv", "favorite_songs.csv"}

    found_legacy = {f.name for f in legacy_dir.glob("*.csv")}
    found_active = {f.name for f in active_dir.glob("*.csv")}

    missing_legacy = expected_legacy - found_legacy
    missing_active = expected_active - found_active
    misplaced_to_legacy = expected_active & found_legacy
    misplaced_to_active = expected_legacy & found_active

    issues: List[str] = []
    if missing_legacy:
        issues.append(f"Legacy files missing: {sorted(missing_legacy)}")
    if missing_active:
        issues.append(f"Active files missing: {sorted(missing_active)}")
    if misplaced_to_legacy:
        issues.append(f"Active files wrongly placed in legacy: {sorted(misplaced_to_legacy)}")
    if misplaced_to_active:
        issues.append(f"Legacy files wrongly placed in active: {sorted(misplaced_to_active)}")

    if issues:
        return False, "; ".join(issues)
    return True, "CSV files split between legacy and active targets correctly."


def verify_subtask4(workspace: Path) -> Tuple[bool, str]:
    kb_dir = _workspace_v2_root(workspace) / "knowledge_base"
    if not kb_dir.is_dir():
        return False, "knowledge_base directory missing."

    expected_files = {
        "project_alpha_README.md",
        "docs_architecture.md",
        "learning_study_notes.md",
        "music_music_manifest.md",
        "travel_plan_draft_plan.md",
    }
    found = {f.name for f in kb_dir.glob("*.md")}

    missing = expected_files - found
    if missing:
        return False, f"Missing renamed markdown files: {sorted(missing)}"
    return True, "All markdown files moved and renamed with parent prefixes."


def verify_subtask5(workspace: Path) -> Tuple[bool, str]:
    source_dir = workspace / "desktop"
    if source_dir.exists():
        return False, f"Legacy desktop directory still present at {source_dir}."
    return True, "Legacy desktop/ tree successfully deleted."


SUBTASK_VALIDATORS: Dict[str, Validator] = {
    "subtask1": verify_subtask1,
    "subtask2": verify_subtask2,
    "subtask3": verify_subtask3,
    "subtask4": verify_subtask4,
    "subtask5": verify_subtask5,
}


SUBTASK_ACCEPTANCE_HINTS: Dict[str, str] = {
    "subtask1": "Evaluator checks that desktop_2/workspace_v2 contains dev_bundle/tests+source, data_warehouse/{legacy_archives,active_datasets}, and knowledge_base.",
    "subtask2": "Evaluator inspects workspace_v2/dev_bundle to ensure debug/test files live in tests/ while the rest are in source/.",
    "subtask3": "Evaluator inspects workspace_v2/data_warehouse to ensure CSV files are separated based on old/exp parent directories.",
    "subtask4": "Evaluator verifies every markdown file is renamed to {parent}_{filename} inside workspace_v2/knowledge_base.",
    "subtask5": "Evaluator expects the original workspace/desktop tree to be fully deleted while workspace_v2 remains intact.",
}


###############################################################################
# Core evaluator
###############################################################################


class Task26Evaluator:
    def __init__(self, env_config: EnvConfig, description_path: Path):
        self.env = env_config
        self.description_path = description_path
        self.description = self._read_description(description_path)
        self.task_root = description_path.parent.resolve()
        self.repo_root = self.task_root.parent
        self.template_source = self.task_root / "desktop"
        if not self.template_source.exists():
            raise FileNotFoundError(f"Template desktop directory missing at {self.template_source}")

        self.model_root = ensure_directory(self.task_root / self.env.model_slug)
        self.desktop_root = self.model_root / "desktop"
        self.desktop2_root = self.model_root / "desktop_2"
        self.meta_path = self.model_root / "meta_eval.json"

        self._reset_model_directory()
        self._relocate_legacy_sii_state()
        self.runner = SiiAgentRunner(self.env, self.model_root)

        self.meta: Dict[str, Any] = {
            "model": self.env.target_model,
            "attempt_limit": self.env.attempt_limit,
            "model_root": str(self._relative_to_repo(self.model_root)),
            "desktop_copy": str(self._relative_to_repo(self.desktop_root)),
            "template_source": str(self._relative_to_repo(self.template_source)),
            "description_file": str(self._relative_to_repo(self.description_path)),
            "subtasks": [],
        }

    async def run(self) -> None:
        try:
            total = int(self.description.get("subtask_count", 0))
            if total <= 0:
                raise ValueError("description.json must define a positive subtask_count.")

            skip_remaining = False
            for index in range(1, total + 1):
                name = f"subtask{index}"
                description = self.description.get(name)
                validator = SUBTASK_VALIDATORS.get(name)

                if skip_remaining:
                    record = self._record_skipped(name, description)
                elif not description:
                    record = self._record_missing_description(name)
                elif not validator:
                    record = self._record_missing_validator(name)
                else:
                    record, halt_next = await self._execute_subtask(
                        name=name,
                        instructions=description,
                        validator=validator,
                    )
                    skip_remaining = halt_next

                self.meta["subtasks"].append(record)

            self.meta_path.write_text(stringify_json(self.meta), encoding="utf-8")
            print(f"[TASK26] Evaluation complete. Metadata saved to {self.meta_path}")
        finally:
            await self.runner.shutdown()

    async def _execute_subtask(
        self,
        name: str,
        instructions: str,
        validator: Validator,
    ) -> Tuple[Dict[str, Any], bool]:
        attempt_records: List[AttemptOutcome] = []
        previous_feedback: Optional[str] = None
        success = False

        for attempt in range(1, self.env.attempt_limit + 1):
            reset_notice = self.runner.consume_reset_notice()
            prompt = self._compose_prompt(
                subtask=name,
                instructions=instructions,
                attempt_index=attempt,
                previous_feedback=previous_feedback,
                reset_notice=reset_notice,
            )
            print(f"[TASK26][{name}] Attempt {attempt}/{self.env.attempt_limit}")
            _, assistant_reply = await self.runner.send(prompt)

            ok, message = validator(self.model_root)
            attempt_records.append(
                AttemptOutcome(
                    attempt_index=attempt,
                    success=ok,
                    message=message,
                    assistant_response=assistant_reply,
                )
            )

            if ok:
                success = True
                break

            previous_feedback = message
            if attempt < self.env.attempt_limit:
                print(f"[TASK26][{name}] Attempt {attempt} failed: {message}")

        record = {
            "name": name,
            "success": success,
            "total_attempts": len(attempt_records),
            "instructions": instructions,
            "attempts": [self._serialize_attempt(a) for a in attempt_records],
        }
        if success:
            record["note"] = "Completed successfully."
        else:
            record["note"] = previous_feedback or "Did not satisfy validator."

        halt_remaining = not success
        return record, halt_remaining

    def _compose_prompt(
        self,
        subtask: str,
        instructions: str,
        attempt_index: int,
        previous_feedback: Optional[str],
        reset_notice: Optional[str],
    ) -> str:
        workspace_rel = self._relative_to_repo(self.model_root)
        desktop_rel = self._relative_to_repo(self.desktop_root)
        workspace_v2_rel = self._relative_to_repo(_workspace_v2_root(self.model_root))
        desktop2_rel = self._relative_to_repo(self.desktop2_root)

        banner = textwrap.dedent(
            f"""
            You are operating inside the dedicated task26 workspace.
            Do not access files outside '{workspace_rel}'. Always `cd` into this directory before running commands.
            The original source files live in '{desktop_rel}'. The destination hierarchy must be built inside '{workspace_v2_rel}' (under '{desktop2_rel}').
            """
        ).strip()

        lines = [
            banner,
            f"Current focus: {subtask} (attempt {attempt_index}/{self.env.attempt_limit}).",
            "Follow the exact requirements from description.json for this step.",
            instructions.strip(),
        ]

        if reset_notice:
            lines.append(
                f"Session note: the SII Agent SDK session was restarted because '{reset_notice}'. "
                f"Reopen shells/editors from within {workspace_rel} before continuing."
            )
        acceptance = SUBTASK_ACCEPTANCE_HINTS.get(subtask)
        if acceptance:
            lines.append(f"Evaluator check: {acceptance}")
        if previous_feedback:
            lines.append(f"Previous attempt feedback: {previous_feedback}")

        lines.append(
            "When you finish, summarize the actions you performed and wait for the evaluator's response. "
            "Do not exit the session unless explicitly instructed."
        )
        return "\n\n".join(lines)

    def _serialize_attempt(self, attempt: AttemptOutcome) -> Dict[str, Any]:
        reply_excerpt = attempt.assistant_response.strip()
        if len(reply_excerpt) > 800:
            reply_excerpt = reply_excerpt[:800] + " …"
        return {
            "attempt_index": attempt.attempt_index,
            "success": attempt.success,
            "validator_message": attempt.message,
            "assistant_response_excerpt": reply_excerpt,
        }

    def _reset_model_directory(self) -> None:
        ensure_directory(self.model_root)
        if self.desktop_root.exists():
            shutil.rmtree(self.desktop_root)
        shutil.copytree(self.template_source, self.desktop_root)
        if self.desktop2_root.exists():
            shutil.rmtree(self.desktop2_root)
        ensure_directory(self.desktop2_root)

    def _relocate_legacy_sii_state(self) -> None:
        legacy_state = self.task_root / ".sii"
        target_state = self.model_root / ".sii"
        if legacy_state.exists():
            if target_state.exists():
                shutil.rmtree(legacy_state)
            else:
                shutil.move(str(legacy_state), str(target_state))

    def _record_skipped(self, name: str, instructions: Optional[str]) -> Dict[str, Any]:
        return {
            "name": name,
            "success": False,
            "total_attempts": 0,
            "instructions": instructions,
            "note": "Skipped because a previous subtask failed.",
            "attempts": [],
        }

    def _record_missing_description(self, name: str) -> Dict[str, Any]:
        return {
            "name": name,
            "success": False,
            "total_attempts": 0,
            "instructions": None,
            "note": "description.json is missing this subtask.",
            "attempts": [],
        }

    def _record_missing_validator(self, name: str) -> Dict[str, Any]:
        return {
            "name": name,
            "success": False,
            "total_attempts": 0,
            "instructions": self.description.get(name),
            "note": f"No validator registered for {name}.",
            "attempts": [],
        }

    def _relative_to_repo(self, path: Path) -> Path:
        try:
            return path.resolve().relative_to(self.repo_root)
        except ValueError:
            return path.resolve()

    @staticmethod
    def _read_description(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))


###############################################################################
# Entrypoint helpers
###############################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated evaluator for task26.")
    parser.add_argument("--env", default=".env", help="Path to the env file relative to task26/")
    parser.add_argument(
        "--description",
        default="description.json",
        help="Path to description.json relative to task26/",
    )
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    task_root = Path(__file__).resolve().parent
    env_path = Path(args.env)
    if not env_path.is_absolute():
        env_path = task_root / env_path
    description_path = Path(args.description)
    if not description_path.is_absolute():
        description_path = task_root / description_path

    env_data = load_env_file(env_path)
    env_config = EnvConfig.from_env(env_data)
    env_config.inject_defaults()

    evaluator = Task26Evaluator(env_config, description_path)
    await evaluator.run()


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
