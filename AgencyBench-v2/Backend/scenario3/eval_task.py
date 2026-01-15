#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import textwrap
import asyncio
import hashlib
import math
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


def copy_workspace(src: Path, dst: Path) -> None:
    src = src.resolve()
    dst = dst.resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source workspace missing: {src}")
    ensure_directory(dst)
    clear_directory(dst)
    for root, dirs, files in os.walk(src):
        rel_root = Path(root).relative_to(src)
        if any(part.startswith(".") for part in rel_root.parts):
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        target_root = dst / rel_root
        target_root.mkdir(parents=True, exist_ok=True)
        for filename in files:
            if filename.startswith("."):
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
    timeout: int = 60,
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


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def try_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


METRIC_HINTS: Dict[str, str] = {
    "determinism": "Determinism未达标：保证固定seed/timestamp并使用SHA256比较同参输出。",
    "validation_accuracy": "Validation accuracy偏低：校验器仍检测到非法记录，请对照JSON Schema修正字段。",
    "mode_coverage": "Mode coverage不足：需在gen.log中记录append与overwrite两种运行。",
    "metadata_completeness": "Metadata completeness缺失：请覆盖Rubric要求的所有seed+schema组合并写入日志。",
    "recovery_correctness": "Recovery correctness失败：--test-recovery应完整重放WAL与快照并输出RECOVERY SUCCESS。",
    "transaction_isolation": "Transaction isolation不达标：串行化测试仍失败，请检查begin/commit隔离逻辑。",
    "concurrency_safety": "Concurrency safety告警：socket多客户端仍产生冲突，需增加锁或冲突检测。",
    "latency_budget": "Latency budget超标：单次commit需压缩在50ms以内，可优化IO或批量刷盘。",
    "completion_ratio": "Completion ratio不足：编排器未让全部任务完成，请检查任务分派/worker生存周期。",
    "retry_resilience": "Retry resilience偏低：强制失败后未成功重试，需实现重试策略并持久化状态。",
    "scheduling_fairness": "Scheduling fairness超限：任务分布不均，需改进worker调度逻辑。",
    "persistence_fidelity": "Persistence fidelity失败：重启后任务状态丢失或重复，请检视KV持久化流程。",
    "feasible_planning": "Feasible planning失败：规划器未生成可执行计划，需确保目标/操作可分解。",
    "dag_validity": "DAG validity异常：生成的任务图存在循环或悬空依赖，需修正依赖构建。",
    "metric_completeness": "Metric completeness缺项：planner_report.json必须包含规定的5个指标。",
    "integration_latency": "Integration latency偏高：规划任务到完成的延迟超标，需减少 orchestrator 延迟。",
    "heuristic_efficiency": "Heuristic efficiency不佳：搜索展开节点过多，请优化启发式或剪枝策略。",
    "detection_accuracy": "Detection accuracy不足：自动修复流程未检测到注入的故障。",
    "rollback_success": "Rollback success失败：快照恢复后状态与预期不符，需验证快照/回放流程。",
    "patch_effectiveness": "Patch effectiveness偏低：生成补丁后仍多次失败，请检查补丁内容与重试策略。",
    "validation_rigor": "Validation rigor不足：自愈后的自检未全部通过，需扩充分级验证。",
    "reporting_completeness": "Reporting completeness缺失：repair_report.json字段不完整，请补齐要求的所有字段。",
}

METRIC_IGNORE_KEYS = {"valid_records", "total_records", "tasks_done", "total_tasks"}


class SimpleJsonSchemaValidator:
    """Lightweight JSON schema validator supporting basic constructs."""

    def __init__(self, schema: Optional[Dict[str, Any]]):
        self.schema = schema or {}

    def validate(self, data: Any) -> bool:
        if not self.schema:
            return True
        return self._validate(self.schema, data)

    def _validate(self, schema: Dict[str, Any], data: Any) -> bool:
        if not isinstance(schema, dict):
            return True
        schema_type = schema.get("type")
        if schema_type:
            if isinstance(schema_type, list):
                if not any(self._check_type(type_name, data) for type_name in schema_type):
                    return False
            else:
                if not self._check_type(str(schema_type), data):
                    return False
        enum_values = schema.get("enum")
        if enum_values is not None and data not in enum_values:
            return False
        if schema_type == "object" or "properties" in schema or "required" in schema:
            if not self._validate_object(schema, data):
                return False
        if schema_type == "array" or "items" in schema:
            if not self._validate_array(schema, data):
                return False
        if self._is_number(data) and any(key in schema for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")):
            if not self._validate_number(schema, data):
                return False
        if isinstance(data, str) and any(key in schema for key in ("minLength", "maxLength", "pattern")):
            if not self._validate_string(schema, data):
                return False
        return True

    def _check_type(self, schema_type: str, value: Any) -> bool:
        if schema_type == "object":
            return isinstance(value, dict)
        if schema_type == "array":
            return isinstance(value, list)
        if schema_type == "string":
            return isinstance(value, str)
        if schema_type == "integer":
            return self._is_integer(value)
        if schema_type == "number":
            return self._is_number(value)
        if schema_type == "boolean":
            return isinstance(value, bool)
        if schema_type == "null":
            return value is None
        return True

    def _validate_object(self, schema: Dict[str, Any], value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        required = schema.get("required") or []
        for key in required:
            if key not in value:
                return False
        properties = schema.get("properties") or {}
        for key, subschema in properties.items():
            if key in value and subschema:
                if not self._validate(subschema, value[key]):
                    return False
        return True

    def _validate_array(self, schema: Dict[str, Any], value: Any) -> bool:
        if not isinstance(value, list):
            return False
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items is not None and len(value) < int(min_items):
            return False
        if max_items is not None and len(value) > int(max_items):
            return False
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for item in value:
                if not self._validate(items_schema, item):
                    return False
        return True

    def _validate_number(self, schema: Dict[str, Any], value: Any) -> bool:
        if schema.get("type") == "integer" and not self._is_integer(value):
            return False
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        exclusive_min = schema.get("exclusiveMinimum")
        exclusive_max = schema.get("exclusiveMaximum")
        if minimum is not None and value < float(minimum):
            return False
        if maximum is not None and value > float(maximum):
            return False
        if exclusive_min is not None and value <= float(exclusive_min):
            return False
        if exclusive_max is not None and value >= float(exclusive_max):
            return False
        return True

    def _validate_string(self, schema: Dict[str, Any], value: str) -> bool:
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")
        if min_length is not None and len(value) < int(min_length):
            return False
        if max_length is not None and len(value) > int(max_length):
            return False
        pattern = schema.get("pattern")
        if pattern:
            if not re.search(pattern, value):
                return False
        return True

    @staticmethod
    def _is_integer(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)


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
    score: float
    metrics: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subtask": self.subtask,
            "score": self.score,
            "metrics": self.metrics,
            "notes": self.notes,
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
# Test plan runner (records executed commands)
# --------------------------------------------------------------------------- #


class TestPlanRunner:
    """Run subtask-specific commands and persist their outputs for inspection."""

    def __init__(self, root: Path, logs_dir: Path, conda_env: Optional[str] = None, conda_exe: Optional[str] = None):
        self.root = root.resolve()
        self.logs_dir = ensure_directory(logs_dir)
        self.conda_env = conda_env
        self.conda_exe = conda_exe
        self._warned_conda_missing = False
        self._records: Dict[str, CommandResult] = {}

    def run(
        self,
        name: str,
        cmd: Sequence[str],
        input_text: Optional[str] = None,
        timeout: int = 120,
    ) -> CommandResult:
        wrapped = self._wrap_cmd(cmd)
        rc, out, err = run_command(wrapped, cwd=self.root, input_text=input_text, timeout=timeout)
        result = CommandResult(name=name, command=wrapped, returncode=rc, stdout=out, stderr=err)
        self._records[name] = result
        self._persist(result)
        return result

    def summary(self) -> Dict[str, Any]:
        return {name: result.to_dict() for name, result in self._records.items()}

    def _persist(self, result: CommandResult) -> None:
        log_path = self.logs_dir / f"{result.name}.txt"
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
        return [self.conda_exe, "run", "-n", self.conda_env, *cmd]


# --------------------------------------------------------------------------- #
# Rubric evaluation (heuristic checks)
# --------------------------------------------------------------------------- #


class RubricEvaluator:
    def evaluate(self, subtask: str, evalspace: Path, runner: TestPlanRunner) -> RubricResult:
        dispatch = {
            "subtask1": self._eval_subtask1,
            "subtask2": self._eval_subtask2,
            "subtask3": self._eval_subtask3,
            "subtask4": self._eval_subtask4,
            "subtask5": self._eval_subtask5,
        }
        handler = dispatch.get(subtask)
        if not handler:
            return RubricResult(subtask=subtask, score=0.0, metrics={}, notes=["Unknown subtask"])
        return handler(evalspace, runner)

    def _eval_subtask1(self, evalspace: Path, runner: TestPlanRunner) -> RubricResult:
        generator = evalspace / "subtask1_generator.py"
        schema = self._find_primary_schema(evalspace / "domain")
        data_dir = ensure_directory(evalspace / "data")
        logs_dir = ensure_directory(evalspace / "logs")
        if not generator.exists():
            return self._missing("subtask1", "Missing subtask1_generator.py")
        if not schema:
            return self._missing("subtask1", "Missing schema under domain/")

        output_path = data_dir / "events.jsonl"
        determinism = self._run_subtask1_determinism(runner, generator, schema, output_path)
        validation_ratio, valid_records, total_records = self._run_subtask1_validation(schema, output_path)
        self._run_subtask1_append(runner, generator, schema, output_path)
        mode_coverage = self._subtask1_mode_coverage(logs_dir / "gen.log")
        metadata = self._subtask1_metadata_matrix(runner, generator, schema, data_dir)

        score = round(10 * (0.35 * determinism + 0.25 * validation_ratio + 0.20 * mode_coverage + 0.20 * metadata), 2)
        metrics = {
            "determinism": determinism,
            "validation_accuracy": validation_ratio,
            "mode_coverage": mode_coverage,
            "metadata_completeness": metadata,
            "valid_records": float(valid_records),
            "total_records": float(total_records),
        }
        notes = self._notes_from_metrics(metrics)
        if total_records == 0:
            notes.append("validator did not observe any events")
        return RubricResult(subtask="subtask1", score=score, metrics=metrics, notes=notes)

    def _run_subtask1_determinism(
        self,
        runner: TestPlanRunner,
        generator: Path,
        schema: Path,
        output: Path,
    ) -> float:
        base_args = [
            "python3",
            str(generator),
            "--seed",
            "42",
            "--count",
            "5",
            "--schema",
            str(schema),
            "--mode",
            "overwrite",
            "--out",
            str(output),
        ]
        result_a = runner.run("subtask1_determinism_a", base_args, timeout=240)
        digest_a = self._sha256(output) if result_a.returncode == 0 else None
        result_b = runner.run("subtask1_determinism_b", base_args, timeout=240)
        digest_b = self._sha256(output) if result_b.returncode == 0 else None
        if not digest_a or not digest_b:
            return 0.0
        return 1.0 if digest_a == digest_b else 0.0

    def _run_subtask1_validation(self, schema: Path, output: Path) -> Tuple[float, int, int]:
        if not schema.exists() or not output.exists():
            return 0.0, 0, 0
        schema_doc = self._load_json(schema)
        validator = SimpleJsonSchemaValidator(schema_doc if isinstance(schema_doc, dict) else {})
        valid = 0
        total = 0
        for line in output.read_text(encoding="utf-8", errors="ignore").splitlines():
            payload = line.strip()
            if not payload:
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                total += 1
                continue
            total += 1
            if validator.validate(obj):
                valid += 1
        ratio = clamp(valid / total, 0.0, 1.0) if total else 0.0
        return ratio, valid, total

    def _run_subtask1_append(self, runner: TestPlanRunner, generator: Path, schema: Path, output: Path) -> None:
        args = [
            "python3",
            str(generator),
            "--seed",
            "7",
            "--count",
            "2",
            "--schema",
            str(schema),
            "--mode",
            "append",
            "--out",
            str(output),
        ]
        runner.run("subtask1_append_mode", args, timeout=180)

    def _subtask1_mode_coverage(self, log_path: Path) -> float:
        entries = self._parse_generator_logs(log_path)
        modes = {str(entry.get("mode", "")).lower().strip(",.") for entry in entries if entry.get("mode")}
        modes.discard("")
        return clamp(len(modes) / 2.0, 0.0, 1.0)

    def _subtask1_metadata_matrix(
        self,
        runner: TestPlanRunner,
        generator: Path,
        default_schema: Path,
        data_dir: Path,
    ) -> float:
        seeds = [101, 202]
        combos: List[Tuple[int, Path]] = []
        for index, seed in enumerate(seeds, start=1):
            target = data_dir / f"metadata_seed{seed}.jsonl"
            args = [
                "python3",
                str(generator),
                "--seed",
                str(seed),
                "--count",
                "3",
                "--schema",
                str(default_schema),
                "--mode",
                "overwrite",
                "--out",
                str(target),
            ]
            runner.run(f"subtask1_metadata_{index}", args, timeout=240)
            combos.append((seed, target))
        if not combos:
            return 0.0
        covered = sum(1 for _, target in combos if target.exists() and target.stat().st_size > 0)
        return clamp(covered / len(combos), 0.0, 1.0)

    def _eval_subtask2(self, evalspace: Path, runner: TestPlanRunner) -> RubricResult:
        script = evalspace / "subtask2_kvstore.py"
        if not script.exists():
            return self._missing("subtask2", "Missing subtask2_kvstore.py")

        recovery = self._subtask2_recovery(runner, script)
        isolation = self._subtask2_ratio_metric(
            runner,
            script,
            "subtask2_serializable",
            ["--run-serializable-tests"],
            "serializable_tests_passed",
            "total_serializable_tests",
        )
        concurrency = self._subtask2_ratio_metric(
            runner,
            script,
            "subtask2_concurrency",
            ["--run-concurrency-sim"],
            "socket_client_trials_without_conflict",
            "total_client_trials",
        )
        latency = self._subtask2_latency_metric(runner, script)

        score = round(10 * (0.40 * recovery + 0.30 * isolation + 0.20 * concurrency + 0.10 * latency), 2)
        metrics = {
            "recovery_correctness": recovery,
            "transaction_isolation": isolation,
            "concurrency_safety": concurrency,
            "latency_budget": latency,
        }
        notes = self._notes_from_metrics(metrics)
        return RubricResult(subtask="subtask2", score=score, metrics=metrics, notes=notes)

    def _subtask2_recovery(self, runner: TestPlanRunner, script: Path) -> float:
        result = runner.run("subtask2_recovery", ["python3", str(script), "--test-recovery"], timeout=300)
        output = (result.stdout + "\n" + result.stderr).lower()
        if result.returncode != 0:
            return 0.0
        return 1.0 if "recovery success" in output else 0.0

    def _subtask2_ratio_metric(
        self,
        runner: TestPlanRunner,
        script: Path,
        name: str,
        args: List[str],
        numerator_key: str,
        denominator_key: str,
    ) -> float:
        result = runner.run(name, ["python3", str(script), *args], timeout=300)
        blob = result.stdout + "\n" + result.stderr
        payload = self._extract_metrics_dict(blob)
        if payload:
            numerator = try_float(payload.get(numerator_key))
            denominator = try_float(payload.get(denominator_key))
        else:
            numerator = self._extract_number(blob, numerator_key)
            denominator = self._extract_number(blob, denominator_key)
        if not denominator or denominator <= 0 or numerator is None:
            return 0.0
        return clamp(numerator / denominator, 0.0, 1.0)

    def _subtask2_latency_metric(self, runner: TestPlanRunner, script: Path) -> float:
        result = runner.run("subtask2_latency", ["python3", str(script), "--bench-latency"], timeout=300)
        blob = result.stdout + "\n" + result.stderr
        payload = self._extract_metrics_dict(blob)
        latency_seconds: Optional[float] = None
        if payload:
            latency_seconds = try_float(payload.get("commit_latency_seconds"))
            if latency_seconds is None:
                latency_ms = try_float(payload.get("commit_latency_ms"))
                latency_seconds = latency_ms / 1000 if latency_ms is not None else None
        if latency_seconds is None:
            latency_ms = self._extract_number(blob, "commit_latency_ms")
            if latency_ms is not None:
                latency_seconds = latency_ms / 1000
        if latency_seconds is None:
            latency_seconds = self._extract_number(blob, "commit_latency_s")
        if latency_seconds is None:
            return 0.0
        return clamp((0.05 - latency_seconds) / 0.05, 0.0, 1.0)

    def _eval_subtask3(self, evalspace: Path, runner: TestPlanRunner) -> RubricResult:
        test_script = evalspace / "subtask3_integration_test.py"
        reports_dir = ensure_directory(evalspace / "reports")
        report_path = reports_dir / "subtask3_metrics.json"
        if not test_script.exists():
            return self._missing("subtask3", "Missing subtask3_integration_test.py")

        cmd = [
            "python3",
            str(test_script),
            "--workers",
            "3",
            "--tasks",
            "9",
            "--force-crash",
            "1",
            "--report",
            str(report_path),
        ]
        runner.run("subtask3_integration", cmd, timeout=480)

        report = self._load_json(report_path)
        if not isinstance(report, dict):
            report = self._extract_metrics_dict_from_logs(runner, "subtask3_integration") or {}

        total_tasks = try_float(report.get("total_tasks")) or 0.0
        tasks_done = try_float(report.get("tasks_done")) or 0.0
        forced_failures = max(try_float(report.get("forced_failures")) or 0.0, 0.0)
        successful_retries = try_float(report.get("successful_retries")) or 0.0
        sigma = try_float(report.get("task_distribution_stddev"))
        persistence_ok = bool(report.get("persistence_ok"))

        completion = clamp(tasks_done / total_tasks, 0.0, 1.0) if total_tasks else 0.0
        retry = clamp(successful_retries / forced_failures, 0.0, 1.0) if forced_failures else 0.0
        fairness = clamp(1.0 - (sigma or math.inf) / 1.5, 0.0, 1.0) if sigma is not None else 0.0
        persistence = 1.0 if persistence_ok else 0.0

        score = round(10 * (0.40 * completion + 0.25 * retry + 0.20 * fairness + 0.15 * persistence), 2)
        metrics = {
            "completion_ratio": completion,
            "retry_resilience": retry,
            "scheduling_fairness": fairness,
            "persistence_fidelity": persistence,
            "tasks_done": tasks_done,
            "total_tasks": total_tasks,
        }
        notes = self._notes_from_metrics(metrics)
        return RubricResult(subtask="subtask3", score=score, metrics=metrics, notes=notes)

    def _eval_subtask4(self, evalspace: Path, runner: TestPlanRunner) -> RubricResult:
        planner = evalspace / "subtask4_planner.py"
        if not planner.exists():
            return self._missing("subtask4", "Missing subtask4_planner.py")
        domain, goal = self._find_domain_goal(evalspace / "domain")
        if not domain or not goal:
            return self._missing("subtask4", "Missing domain/goal fixtures")
        reports_dir = ensure_directory(evalspace / "reports")
        report_path = reports_dir / "planner_report.json"
        plan_export = ensure_directory(evalspace / "data") / "plan_tasks.json"

        cmd = [
            "python3",
            str(planner),
            "--domain",
            str(domain),
            "--goal",
            str(goal),
            "--report",
            str(report_path),
            "--plan-export",
            str(plan_export),
        ]
        runner.run("subtask4_planner", cmd, timeout=480)

        report = self._load_json(report_path)
        if not isinstance(report, dict):
            return self._missing("subtask4", "planner_report.json invalid")

        success_rate = try_float(report.get("success_rate")) or 0.0
        nodes_expanded = try_float(report.get("nodes_expanded")) or 0.0
        latency = try_float(report.get("schedule_latency"))

        feasible = 1.0 if success_rate >= 1.0 or report.get("plan_success") else 0.0
        dag_validity = self._verify_dag(plan_export)
        metric_keys = ("success_rate", "plan_length", "nodes_expanded", "heuristic_cost", "schedule_latency")
        metric_completeness = clamp(sum(1 for key in metric_keys if key in report) / 5.0, 0.0, 1.0)
        latency_metric = clamp((0.5 - (latency or 1.0)) / 0.5, 0.0, 1.0)
        heuristic_efficiency = clamp(500.0 / max(nodes_expanded, 500.0), 0.0, 1.0)

        score = round(
            10 * (0.30 * feasible + 0.25 * dag_validity + 0.20 * metric_completeness + 0.15 * latency_metric + 0.10 * heuristic_efficiency),
            2,
        )
        metrics = {
            "feasible_planning": feasible,
            "dag_validity": dag_validity,
            "metric_completeness": metric_completeness,
            "integration_latency": latency_metric,
            "heuristic_efficiency": heuristic_efficiency,
        }
        notes = self._notes_from_metrics(metrics)
        return RubricResult(subtask="subtask4", score=score, metrics=metrics, notes=notes)

    def _eval_subtask5(self, evalspace: Path, runner: TestPlanRunner) -> RubricResult:
        supervisor = evalspace / "subtask5_autorepair.py"
        reports_dir = ensure_directory(evalspace / "reports")
        report_path = reports_dir / "repair_report.json"
        if not supervisor.exists():
            return self._missing("subtask5", "Missing subtask5_autorepair.py")

        runner.run(
            "subtask5_simulate",
            ["python3", str(supervisor), "--simulate-failure", "--report", str(report_path)],
            timeout=480,
        )
        report = self._load_json(report_path)
        if not isinstance(report, dict):
            return self._missing("subtask5", "repair_report.json invalid")

        detected = try_float(report.get("detected_failures")) or 0.0
        injected = try_float(report.get("injected_failures")) or 0.0
        rollback_status = str(report.get("rollback_status") or "").lower()
        successful_retries = try_float(report.get("successful_retries")) or 0.0
        total_retries = try_float(report.get("total_retries")) or 0.0
        post_checks = try_float(report.get("post_repair_checks_passed")) or 0.0
        total_checks = try_float(report.get("total_checks")) or 0.0

        detection = clamp(detected / injected, 0.0, 1.0) if injected else 0.0
        rollback = 1.0 if rollback_status in {"success", "ok", "passed", "true"} else 0.0
        patch_effectiveness = clamp(successful_retries / total_retries, 0.0, 1.0) if total_retries else 0.0
        validation_rigor = clamp(post_checks / total_checks, 0.0, 1.0) if total_checks else 0.0

        required_fields = [
            "trigger",
            "snapshot_source",
            "patch_id",
            "retry_attempts",
            "final_status",
            "detected_failures",
            "injected_failures",
            "rollback_status",
            "successful_retries",
            "total_retries",
            "post_repair_checks_passed",
            "total_checks",
        ]
        filled = sum(1 for field in required_fields if field in report and report[field] not in (None, ""))
        reporting = clamp(filled / len(required_fields), 0.0, 1.0)

        score = round(
            10 * (0.30 * detection + 0.25 * rollback + 0.25 * patch_effectiveness + 0.10 * validation_rigor + 0.10 * reporting),
            2,
        )
        metrics = {
            "detection_accuracy": detection,
            "rollback_success": rollback,
            "patch_effectiveness": patch_effectiveness,
            "validation_rigor": validation_rigor,
            "reporting_completeness": reporting,
        }
        notes = self._notes_from_metrics(metrics)
        return RubricResult(subtask="subtask5", score=score, metrics=metrics, notes=notes)

    def _missing(self, subtask: str, message: str) -> RubricResult:
        return RubricResult(subtask=subtask, score=0.0, metrics={}, notes=[message])

    def _notes_from_metrics(self, metrics: Dict[str, float]) -> List[str]:
        notes: List[str] = []
        for key, value in metrics.items():
            if key in METRIC_IGNORE_KEYS:
                continue
            numeric = try_float(value)
            if numeric is None:
                continue
            if numeric >= 0.99:
                continue
            hint = METRIC_HINTS.get(key, f"{key}未达标：请按照Rubric补足该测试点。")
            notes.append(f"{hint} (当前:{numeric:.2f})")
        return notes

    def _sha256(self, path: Path) -> Optional[str]:
        if not path.exists() or not path.is_file():
            return None
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _load_json(self, path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _extract_metrics_dict(self, blob: str) -> Optional[Dict[str, Any]]:
        snippet = self._extract_json_snippet(blob)
        if not snippet:
            return None
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
        return None

    def _extract_json_snippet(self, blob: str) -> Optional[str]:
        blob = blob.strip()
        if not blob:
            return None
        start = blob.find("{")
        end = blob.rfind("}")
        if start != -1 and end != -1 and end > start:
            return blob[start : end + 1]
        return None

    def _extract_number(self, blob: str, key: str) -> Optional[float]:
        pattern = re.compile(rf"{re.escape(key)}[^0-9]*([0-9]+(?:\\.[0-9]+)?)", re.IGNORECASE)
        match = pattern.search(blob)
        if match:
            return try_float(match.group(1))
        return None

    def _extract_metrics_dict_from_logs(self, runner: TestPlanRunner, command_name: str) -> Optional[Dict[str, Any]]:
        record = runner.summary().get(command_name)
        if not record:
            return None
        payload = (record.get("stdout") or "") + "\n" + (record.get("stderr") or "")
        return self._extract_metrics_dict(payload)

    def _find_primary_schema(self, domain_dir: Path) -> Optional[Path]:
        schemas = self._list_schema_files(domain_dir)
        if not schemas:
            return None
        preferred = [path for path in schemas if "event" in path.name.lower()]
        return preferred[0] if preferred else schemas[0]

    def _list_schema_files(self, domain_dir: Path) -> List[Path]:
        if not domain_dir.exists():
            return []
        return sorted([path for path in domain_dir.glob("*.json") if path.is_file()])

    def _parse_generator_logs(self, log_path: Path) -> List[Dict[str, Any]]:
        if not log_path.exists():
            return []
        entries: List[Dict[str, Any]] = []
        for raw in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line:
                continue
            parsed: Dict[str, Any] = {}
            snippet = self._extract_json_snippet(line)
            if snippet:
                try:
                    obj = json.loads(snippet)
                    if isinstance(obj, dict):
                        parsed = obj
                except json.JSONDecodeError:
                    parsed = {}
            if not parsed:
                seed_match = re.search(r"seed[^0-9]*(\\d+)", line, re.IGNORECASE)
                schema_match = re.search(r"schema[^=]*=([^\\s]+)", line, re.IGNORECASE)
                mode_match = re.search(r"mode[^=]*=([^\\s]+)", line, re.IGNORECASE)
                if seed_match:
                    parsed["seed"] = seed_match.group(1)
                if schema_match:
                    parsed["schema"] = schema_match.group(1)
                if mode_match:
                    parsed["mode"] = mode_match.group(1)
            if parsed:
                entries.append(parsed)
        return entries

    def _find_domain_goal(self, domain_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
        if not domain_dir.exists():
            return None, None
        json_files = [path for path in domain_dir.glob("*.json") if path.is_file()]
        domain = None
        goal = None
        for path in json_files:
            lowered = path.name.lower()
            if "goal" in lowered and goal is None:
                goal = path
            elif "domain" in lowered and domain is None:
                domain = path
        if not domain and json_files:
            domain = json_files[0]
        if not goal and len(json_files) > 1:
            goal = json_files[1 if json_files[0] == domain else 0]
        return domain, goal

    def _verify_dag(self, plan_path: Path) -> float:
        if not plan_path.exists():
            return 0.0
        data = self._load_json(plan_path)
        if not data:
            return 0.0
        tasks = data.get("tasks") if isinstance(data, dict) else data
        if not isinstance(tasks, list):
            return 0.0
        ids = {task.get("id") for task in tasks if isinstance(task, dict) and task.get("id")}
        if not ids:
            return 0.0
        graph: Dict[str, List[str]] = {}
        total_edges = 0
        valid_edges = 0
        for task in tasks:
            if not isinstance(task, dict):
                continue
            node_id = task.get("id")
            deps = task.get("dependencies") or task.get("deps") or []
            deps_list = [dep for dep in deps if isinstance(dep, str)]
            if node_id:
                graph[node_id] = deps_list
            for dep in deps_list:
                total_edges += 1
                if dep in ids:
                    valid_edges += 1

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return False
            if node in visited:
                return True
            visiting.add(node)
            for dep in graph.get(node, []):
                if dep not in ids:
                    continue
                if not visit(dep):
                    return False
            visiting.remove(node)
            visited.add(node)
            return True

        if not all(visit(node) for node in ids):
            return 0.0
        if total_edges == 0:
            return 1.0
        return clamp(valid_edges / total_edges, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Coordinator
# --------------------------------------------------------------------------- #


class EvaluationCoordinator:
    def __init__(
        self,
        env_config: EnvConfig,
        description_path: Path,
        visualize: bool = False,
        start_subtask: int = 1,
    ):
        self.env_config = env_config
        description_path = resolve_script_path(description_path)
        self.description_path = description_path
        self.visualize = visualize
        self.description = read_json(description_path)
        self.base_dir = description_path.parent.resolve()
        self.repo_root = self.base_dir.parent.resolve()
        self.run_root = (self.base_dir / env_config.model_name).resolve()
        self.rubric = RubricEvaluator()
        self.start_subtask = max(1, start_subtask)
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
        agent = AgentRunner(self.env_config, visualize=self.visualize)
        previous_best: Optional[Path] = None
        subtask_count = int(self.description.get("subtask_count") or 0)

        start_index = min(self.start_subtask, subtask_count or 1)
        print(
            f"[BOOT] Model={self.env_config.model_name} scorer={self.env_config.scorer_name} "
            f"start_subtask={start_index}"
        )
        for index in range(start_index, subtask_count + 1):
            subtask = f"subtask{index}"
            prompt = self.description.get(subtask, "")
            attempt_summaries: List[AttemptSummary] = []
            cache_workspace: Optional[Path] = None
            feedback: str = ""
            print(f"[SUBTASK] Starting {subtask}")
            for attempt in range(1, self.env_config.max_attempts + 1):
                workspace, evalspace = self._prepare_attempt_dirs(subtask, attempt, previous_best)
                if attempt > 1:
                    self._clone_previous_attempt(subtask, attempt, workspace)
                elif previous_best:
                    copy_workspace(previous_best, workspace)

                agent_output = agent.send(
                    prompt + ("\n\n" + feedback if feedback else ""),
                    workspace_notice(workspace, self.repo_root),
                    workspace,
                )
                print(f"[COPY] Starting copy workspace")
                copy_workspace(workspace, evalspace)
                logs_dir = ensure_directory(evalspace / "logs")
                conda_env = self.env_config.conda_env_name or None
                print(f"[EVAL] Starting evaluation")
                runner = TestPlanRunner(
                    evalspace,
                    logs_dir,
                    conda_env=conda_env if conda_exe else None,
                    conda_exe=conda_exe,
                )
                rubric_result = self.rubric.evaluate(subtask, evalspace, runner)
                commands = runner.summary()
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
                metrics_preview = ", ".join(
                    f"{key}={value:.2f}"
                    for key, value in list(rubric_result.metrics.items())[:3]
                )
                metric_text = f" metrics: {metrics_preview}" if metrics_preview else ""
                print(f"[RESULT] {subtask} attempt {attempt}: score={rubric_result.score}{metric_text}")
                if rubric_result.notes:
                    print("         Notes: " + "; ".join(rubric_result.notes))
                if rubric_result.score >= 0:
                    cache_workspace = workspace
                    break
                cache_workspace = cache_workspace or workspace

            best = max(attempt_summaries, key=lambda item: item.score) if attempt_summaries else None
            if cache_workspace is None and best:
                cache_workspace = best.workspace
            if cache_workspace:
                previous_best = cache_workspace
            self.meta["subtasks"].append(
                {
                    "name": subtask,
                    "attempts": [item.to_dict() for item in attempt_summaries],
                    "best_score": best.score if best else 0,
                    "best_attempt": best.attempt_index if best else None,
                    "best_metrics": best.rubric.metrics if best else {},
                    "best_notes": best.rubric.notes if best else [],
                    "attempt_count": len(attempt_summaries),
                    "best_workspace": str(best.workspace) if best else "",
                }
            )

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
        if not rubric.notes:
            return ""
        bullets = "\n".join(f"- {item}" for item in rubric.notes)
        return f"Focus on improving these rubric metrics:\n{bullets}"

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

        cmd = [conda_exe, "run", "-n", env_name, "python", "-c", "import sys; print('conda-env', sys.executable)"]
        rc, out, err = run_command(cmd, timeout=45)
        print(f"[DEBUG] Conda check (python) rc={rc}")
        if out.strip():
            print(f"[DEBUG] stdout: {out.strip()}")
        if err.strip():
            print(f"[DEBUG] stderr: {err.strip()}")
        if rc != 0:
            print(
                f"[WARN] Conda env '{env_name}' failed python check; "
                "falling back to host PATH commands."
            )
            return None

        print(f"[DEBUG] Conda env '{env_name}' verified via {conda_exe}.")
        return conda_exe

    def _print_command_diagnostics(
        self, subtask: str, attempt: int, commands: Dict[str, Any]
    ) -> None:
        print(f"[DETAIL] {subtask} attempt {attempt} command diagnostics:")
        if not commands:
            print("         No commands recorded.")
            return
        for name in sorted(commands):
            data = commands.get(name)
            if not isinstance(data, dict):
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

    def _print_rubric_diagnostics(self, subtask: str, attempt: int, rubric: RubricResult) -> None:
        print(f"[DETAIL] {subtask} attempt {attempt} rubric diagnostics:")
        if not rubric.metrics:
            print("         No rubric metrics recorded.")
            return
        for key, value in rubric.metrics.items():
            numeric = try_float(value)
            if numeric is None:
                status = "FAIL"
                display = str(value)
            else:
                status = "PASS" if numeric >= 0.99 else "FAIL"
                display = f"{numeric:.2f}"
            print(f"         [{status}] {key} = {display}")
        if rubric.notes:
            print("         Notes: " + "; ".join(rubric.notes))

    @staticmethod
    def _truncate_output(text: str, limit: int = 200) -> str:
        if not text:
            return ""
        cleaned = " ".join(text.split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 3] + "..."


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task38 evaluator runner")
    parser.add_argument("--env", default=".env", help="Path to env file for agent credentials")
    parser.add_argument("--visualize", action="store_true", help="Enable SII SDK visualization")
    parser.add_argument("--description", default="description.json", help="Path to task description JSON")
    parser.add_argument("--start-subtask", type=int, default=1, help="1-based subtask index to start evaluation from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_path = resolve_script_path(Path(args.env))
    env_config = EnvConfig.load(env_path, visualize=args.visualize)
    description_path = resolve_script_path(Path(args.description))
    coordinator = EvaluationCoordinator(
        env_config,
        description_path,
        visualize=args.visualize,
        start_subtask=max(1, int(args.start_subtask or 1)),
    )
    coordinator.run()


if __name__ == "__main__":
    main()
