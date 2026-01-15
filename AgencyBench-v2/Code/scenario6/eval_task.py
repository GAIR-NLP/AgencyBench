"""Automated evaluation pipeline for AgencyBench task29.

This orchestrator uses the SII Agent SDK to carry out up to five
equation-discovery subtasks described in ``description.json``. Each subtask
receives at most ``SUBTASK_ATTEMPT_LIMIT`` attempts (default: 2). After each
attempt the evaluator inspects ``workspace/subtask*_result.json`` to verify
that the reported loss meets the target threshold (1e-3 … 1e-11). When an
attempt fails, its JSON output and failure feedback are appended to the next
prompt so the agent can revise its work. As soon as a subtask fails to meet the
required loss, the remaining subtasks are skipped. Each evaluation run stages a
fresh copy of the template ``workspace`` inside ``task29/<model_slug>/`` and
constrains the agent to that directory. The final evaluation summary for every
subtask (formula, loss, and per-attempt details) is saved to
``task29/<model_slug>/meta_eval.json``.

Scoring note (AgencyBench-v2): we map the best observed optimization loss to a
10-point score:
- loss <= 1e-7  -> 10
- 1e-7 < loss <= 1e-6  -> 8
- 1e-6 < loss <= 1e-5  -> 6
- 1e-5 < loss <= 1e-4  -> 4
- 1e-4 < loss <= 1e-3  -> 2
- loss > 1e-3  -> 0
The evaluator records `best_loss` and `final_score` in `meta_eval.json`.
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
from typing import Any, Dict, List, Optional, Tuple

from sii_agent_sdk import AssistantMessage, CompletedMessage, Message, SiiAgentOptions, TextBlock
from sii_agent_sdk._internal.event_logger import log_message
from sii_agent_sdk._internal.message_parser import parse_message
from sii_agent_sdk.bridge import BridgeProcess
from sii_agent_sdk.query import _message_to_session_turns, _raise_appropriate_error, validate_auth_config
from sii_agent_sdk.session_state import ConversationTurn, SessionState


###############################################################################
# Asyncio compatibility helpers (copied from task21 evaluator)
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
# Utilities
###############################################################################


DEFAULT_THRESHOLDS = [1e-7, 1e-5, 1e-7, 1e-9, 1e-11]


def score_from_loss(loss: Optional[float]) -> int:
    if loss is None:
        return 0
    if loss <= 1e-7:
        return 10
    if loss <= 1e-6:
        return 8
    if loss <= 1e-5:
        return 6
    if loss <= 1e-4:
        return 4
    if loss <= 1e-3:
        return 2
    return 0


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


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stringify_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


###############################################################################
# Environment configuration
###############################################################################


def derive_model_name(identifier: str) -> str:
    raw = identifier.strip()
    if "/" in raw:
        raw = raw.split("/")[-1]
    sanitized = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in raw)
    sanitized = sanitized.strip("._-")
    return sanitized or "model"


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
                raise ValueError(f"Environment variable '{key}' must be set in task29/.env")
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
# Attempt evaluation metadata
###############################################################################


@dataclass
class AttemptEvaluation:
    attempt_index: int
    file_exists: bool
    parsed: bool
    meets_threshold: bool
    loss: Optional[float]
    equation: Optional[str]
    comment: Optional[str]
    threshold: float
    reason: str
    raw_json: Optional[str]

    @property
    def success(self) -> bool:
        return self.file_exists and self.parsed and self.meets_threshold


###############################################################################
# SII Agent orchestration
###############################################################################


class SiiAgentRunner:
    """Persistent bridge runner that keeps a single SII Agent session alive across prompts."""

    def __init__(self, env_config: EnvConfig, agent_root: Path):
        self.env = env_config
        self.agent_root = ensure_directory(agent_root).resolve()
        self._options: Optional[SiiAgentOptions] = None
        self._bridge: Optional[BridgeProcess] = None
        self._session_state = SessionState()
        self._last_completion_metadata: Optional[Dict[str, Any]] = None

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
# Task orchestration
###############################################################################


class Task29Evaluator:
    def __init__(self, env_config: EnvConfig, description_path: Path, env_data: Dict[str, str]):
        self.env = env_config
        self.description_path = description_path
        self.description = self._read_description(description_path)
        self.task_root = description_path.parent.resolve()
        self.repo_root = self.task_root.parent
        self.base_workspace = self.task_root / "workspace"
        if not self.base_workspace.exists():
            raise FileNotFoundError(
                f"Base workspace directory is missing at {self.base_workspace}. Unable to prepare model workspace."
            )
        self.model_root = ensure_directory(self.task_root / self.env.model_slug)
        self.workspace_dir = self.model_root / "workspace"
        self._reset_model_workspace()
        self._relocate_legacy_sii_state()
        self.runner = SiiAgentRunner(env_config, self.model_root)
        self.meta_path = self.model_root / "meta_eval.json"
        self.thresholds = self._parse_thresholds(env_data.get("SUBTASK_LOSS_THRESHOLDS"))
        self.meta: Dict[str, Any] = {
            "attempt_limit": self.env.attempt_limit,
            "loss_thresholds": self.thresholds,
            "model_root": str(self._relative_to_repo(self.model_root)),
            "workspace_template": str(self._relative_to_repo(self.base_workspace)),
            "workspace": str(self._relative_to_repo(self.workspace_dir)),
            "subtasks": [],
        }

    async def run(self) -> None:
        try:
            total = int(self.description.get("subtask_count", 0))
            if total <= 0:
                raise ValueError("description.json must define a positive subtask_count.")

            halt_remaining = False
            for index in range(1, total + 1):
                key = f"subtask{index}"
                content = self.description.get(key)
                result_path = self.workspace_dir / f"{key}_result.json"
                threshold = self._loss_threshold_for(index)
                if halt_remaining:
                    record = self._record_skipped_subtask(key, result_path, threshold)
                elif not content:
                    record = self._record_missing_description(key, result_path, threshold)
                else:
                    record, halt_remaining = await self._run_subtask(
                        key, content, index, result_path, threshold
                    )
                self.meta["subtasks"].append(record)

            best_loss: Optional[float] = None
            for subtask in self.meta.get("subtasks", []):
                attempts = subtask.get("attempts")
                if not isinstance(attempts, list):
                    continue
                for attempt in attempts:
                    if not isinstance(attempt, dict):
                        continue
                    loss = attempt.get("loss")
                    if isinstance(loss, (int, float)):
                        loss_value = float(loss)
                        best_loss = loss_value if best_loss is None else min(best_loss, loss_value)

            self.meta["best_loss"] = best_loss
            self.meta["final_score"] = score_from_loss(best_loss)
            self.meta_path.write_text(stringify_json(self.meta), encoding="utf-8")
            print(f"[TASK29] Run complete. Metadata stored at {self.meta_path}")
        finally:
            await self.runner.shutdown()

    async def _run_subtask(
        self,
        subtask_name: str,
        subtask_content: str,
        subtask_index: int,
        result_path: Path,
        threshold: float,
    ) -> Tuple[Dict[str, Any], bool]:
        attempt_records: List[Dict[str, Any]] = []
        previous_output: Optional[str] = None
        previous_feedback: Optional[str] = None
        success = False
        final_result: Optional[Dict[str, Any]] = None

        for attempt in range(1, self.env.attempt_limit + 1):
            self._prepare_result_artifact(result_path)
            prompt = self._compose_prompt(
                workspace=self.workspace_dir,
                result_path=result_path,
                subtask_content=subtask_content,
                subtask_name=subtask_name,
                threshold=threshold,
                attempt_index=attempt,
                attempt_limit=self.env.attempt_limit,
                previous_output=previous_output,
                previous_feedback=previous_feedback,
            )
            print(f"[TASK29][{subtask_name}] Attempt {attempt}/{self.env.attempt_limit}")
            await self.runner.send(prompt)

            evaluation = self._evaluate_result_file(result_path, threshold, attempt)
            attempt_records.append(self._attempt_record(evaluation, result_path))

            if evaluation.success:
                success = True
                final_result = self._final_result_payload(evaluation, result_path)
                break

            previous_feedback = evaluation.reason
            previous_output = evaluation.raw_json

            if attempt < self.env.attempt_limit:
                print(
                    f"[TASK29][{subtask_name}] Attempt {attempt} failed ({evaluation.reason}). Retrying..."
                )

        record: Dict[str, Any] = {
            "subtask": subtask_name,
            "threshold": threshold,
            "success": success,
            "total_attempts": len(attempt_records),
            "result_file": str(self._relative_to_repo(result_path)),
            "attempts": attempt_records,
        }
        if final_result:
            record["final_result"] = final_result
        else:
            record["note"] = (
                previous_feedback
                or f"Did not meet loss threshold {threshold} within {self.env.attempt_limit} attempts."
            )

        halt_remaining = not success
        return record, halt_remaining

    def _record_skipped_subtask(self, subtask_name: str, path: Path, threshold: float) -> Dict[str, Any]:
        return {
            "subtask": subtask_name,
            "threshold": threshold,
            "success": False,
            "total_attempts": 0,
            "result_file": str(self._relative_to_repo(path)),
            "attempts": [
                {
                    "attempt_index": 0,
                    "success": False,
                    "loss": None,
                    "equation": None,
                    "comment": None,
                    "notes": "Skipped because a previous subtask did not reach its target loss.",
                }
            ],
            "note": "Skipped due to earlier failure.",
        }

    def _record_missing_description(
        self, subtask_name: str, path: Path, threshold: float
    ) -> Dict[str, Any]:
        return {
            "subtask": subtask_name,
            "threshold": threshold,
            "success": False,
            "total_attempts": 0,
            "result_file": str(self._relative_to_repo(path)),
            "attempts": [
                {
                    "attempt_index": 0,
                    "success": False,
                    "loss": None,
                    "equation": None,
                    "comment": None,
                    "notes": "Skipped because description.json does not contain instructions for this subtask.",
                }
            ],
            "note": "Missing subtask description.",
        }

    def _attempt_record(self, evaluation: AttemptEvaluation, result_path: Path) -> Dict[str, Any]:
        return {
            "attempt_index": evaluation.attempt_index,
            "success": evaluation.success,
            "loss": evaluation.loss,
            "equation": evaluation.equation,
            "comment": evaluation.comment,
            "threshold": evaluation.threshold,
            "file_detected": evaluation.file_exists,
            "parsed": evaluation.parsed,
            "notes": evaluation.reason,
            "result_path": str(self._relative_to_repo(result_path)),
        }

    def _final_result_payload(
        self, evaluation: AttemptEvaluation, result_path: Path
    ) -> Dict[str, Any]:
        return {
            "loss": evaluation.loss,
            "equation": evaluation.equation,
            "comment": evaluation.comment,
            "result_path": str(self._relative_to_repo(result_path)),
        }

    def _reset_model_workspace(self) -> None:
        if not self.base_workspace.exists() or not self.base_workspace.is_dir():
            raise FileNotFoundError(
                f"Workspace template missing at {self.base_workspace}. Cannot stage model workspace."
            )
        ensure_directory(self.model_root)
        if self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir)
        shutil.copytree(self.base_workspace, self.workspace_dir)

    def _relocate_legacy_sii_state(self) -> None:
        legacy_state = self.task_root / ".sii"
        target_state = self.model_root / ".sii"
        if legacy_state.exists():
            if target_state.exists():
                shutil.rmtree(legacy_state)
            else:
                shutil.move(str(legacy_state), str(target_state))

    def _prepare_result_artifact(self, path: Path) -> None:
        ensure_directory(path.parent)
        if path.exists():
            path.unlink()

    def _evaluate_result_file(self, path: Path, threshold: float, attempt_index: int) -> AttemptEvaluation:
        if not path.exists():
            return AttemptEvaluation(
                attempt_index=attempt_index,
                file_exists=False,
                parsed=False,
                meets_threshold=False,
                loss=None,
                equation=None,
                comment=None,
                threshold=threshold,
                reason=f"{path.name} missing after attempt.",
                raw_json=None,
            )

        raw_text = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(raw_text)
            parsed = True
        except json.JSONDecodeError:
            return AttemptEvaluation(
                attempt_index=attempt_index,
                file_exists=True,
                parsed=False,
                meets_threshold=False,
                loss=None,
                equation=None,
                comment=None,
                threshold=threshold,
                reason=f"{path.name} is not valid JSON.",
                raw_json=raw_text,
            )

        loss_value = self._coerce_float(payload.get("loss"))
        equation = self._coerce_str(payload.get("equation"))
        comment = self._coerce_str(payload.get("comment"))

        if loss_value is None:
            return AttemptEvaluation(
                attempt_index=attempt_index,
                file_exists=True,
                parsed=True,
                meets_threshold=False,
                loss=None,
                equation=equation,
                comment=comment,
                threshold=threshold,
                reason=f"{path.name} is missing a numeric 'loss' field.",
                raw_json=raw_text,
            )

        meets_threshold = loss_value < threshold
        reason = (
            f"Loss {loss_value:.4g} {'<' if meets_threshold else '>='} target {threshold}."
        )

        return AttemptEvaluation(
            attempt_index=attempt_index,
            file_exists=True,
            parsed=True,
            meets_threshold=meets_threshold,
            loss=loss_value,
            equation=equation,
            comment=comment,
            threshold=threshold,
            reason=reason,
            raw_json=raw_text,
        )

    def _compose_prompt(
        self,
        workspace: Path,
        result_path: Path,
        subtask_content: str,
        subtask_name: str,
        threshold: float,
        attempt_index: int,
        attempt_limit: int,
        previous_output: Optional[str],
        previous_feedback: Optional[str],
    ) -> str:
        workspace_rel = self._relative_to_repo(workspace)
        result_rel = self._relative_to_repo(result_path)
        lines = [
            f"You must work exclusively inside '{workspace_rel}'. Do not touch files outside this directory.",
            "Available tools and files: equation.py (define equation), analysis.py (inspect data), evaluate_equation.py (compute loss), equation.parquet (data).",
            f"Current subtask: {subtask_name} (attempt {attempt_index}/{attempt_limit}). Target loss: < {threshold}.",
            f"After completing your edits, run `python evaluate_equation.py` to estimate the loss and record the final values in {result_rel}.",
            "The JSON file must include exactly these keys: loss (float), equation (string with the formula), comment (brief summary).",
        ]
        if previous_output:
            lines.append(
                "Previous attempt output from {path}:\n```json\n{json}\n```".format(
                    path=result_rel, json=previous_output.strip()
                )
            )
        if previous_feedback:
            lines.append(f"Previous attempt feedback: {previous_feedback}")

        lines.append("Subtask description:\n" + subtask_content.strip())
        body = "\n\n".join(lines)
        banner = self._workspace_banner(workspace_rel)
        return f"{banner}\n\n{body}"

    def _workspace_banner(self, workspace_rel: Path) -> str:
        workspace_text = str(workspace_rel)
        return textwrap.dedent(
            f"""
            You are already inside the model-specific workspace prepared for this evaluation.
            Always change into {workspace_text} before running commands and keep every edit restricted to this folder.
            """
        ).strip()

    def _parse_thresholds(self, raw: Optional[str]) -> List[float]:
        if not raw:
            return DEFAULT_THRESHOLDS.copy()
        thresholds: List[float] = []
        for part in raw.split(","):
            piece = part.strip()
            if not piece:
                continue
            try:
                thresholds.append(float(piece))
            except ValueError:
                continue
        return thresholds or DEFAULT_THRESHOLDS.copy()

    def _loss_threshold_for(self, subtask_index: int) -> float:
        if 1 <= subtask_index <= len(self.thresholds):
            return self.thresholds[subtask_index - 1]
        fallback_index = min(subtask_index - 1, len(DEFAULT_THRESHOLDS) - 1)
        return DEFAULT_THRESHOLDS[fallback_index]

    def _relative_to_repo(self, path: Path) -> Path:
        try:
            return path.resolve().relative_to(self.repo_root)
        except ValueError:
            return path.resolve()

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        return str(value)

    def _read_description(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))


###############################################################################
# Entrypoint
###############################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated evaluator for task29.")
    parser.add_argument("--env", default=".env", help="Path to the env file relative to task29/")
    parser.add_argument(
        "--description",
        default="description.json",
        help="Path to description.json relative to task29/",
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

    evaluator = Task29Evaluator(env_config, description_path, env_data)
    await evaluator.run()


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
