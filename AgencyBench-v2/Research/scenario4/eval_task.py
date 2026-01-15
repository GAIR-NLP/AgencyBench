"""Automated evaluation pipeline for AgencyBench task21.

This script coordinates two subtasks executed through a single SII Agent SDK
session:

1. Use web/deep-research tools to collect 5–10 trustworthy references that
   answer the Chat-vs-Agent question and persist them to `workspace/web_info.json`.
2. Reuse the same session to write a structured report saved as
   `workspace/ResearchSystem_ChatVsAgent.json`, then run the ResearcherBench
   rubric evaluator on that report.

Each subtask may be attempted twice (or `SUBTASK_ATTEMPT_LIMIT`), and scoring is
binary: producing the required artifact yields 10 points, otherwise 0. If the
first subtask still fails after the last attempt, later subtasks are skipped and
scored zero automatically. Metadata for all attempts plus rubric output goes to
`meta_eval.json` in the model run directory.

Scoring note (AgencyBench-v2): this scenario reports a ResearcherBench rubric
`recall` for subtask2. We normalize `recall` with 0.5 as full score and compute
`final_score = round(10 * clamp(recall / 0.5, 0, 1))`, saved into `meta_eval.json`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sii_agent_sdk import (
    AssistantMessage,
    Message,
    SiiAgentOptions,
    TextBlock,
    SiiAgentSession as SDKAgentSession,
)


###############################################################################
# Compatibility helpers
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
# Generic utilities
###############################################################################


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


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


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
                raise ValueError(f"Environment variable '{key}' must be set in task21/.env")
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
# Attempt summary for rubric eval
###############################################################################


@dataclass
class AttemptRubricSummary:
    recall: Optional[float] = None
    covered_points: List[str] = field(default_factory=list)
    uncovered_points: List[str] = field(default_factory=list)

    @property
    def covered_count(self) -> int:
        return len(self.covered_points)


###############################################################################
# SII Agent orchestration
###############################################################################


class SiiAgentRunner:
    def __init__(self, env_config: EnvConfig, agent_root: Path):
        self.env = env_config
        self.agent_root = ensure_directory(agent_root).resolve()
        self._session: Optional[SDKAgentSession] = None
        self._options: Optional[SiiAgentOptions] = None

    def _ensure_session(self) -> None:
        env_vars = {
            "OPENAI_API_KEY": self.env.agent_api_key,
            "OPENAI_BASE_URL": self.env.agent_api_base,
            "SII_OPENAI_API_KEY": self.env.agent_api_key,
            "SII_OPENAI_BASE_URL": self.env.agent_api_base,
            "SII_OPENAI_MODEL": self.env.target_model,
            "SII_USERNAME": self.env.username,
            "SII_PASSWORD": self.env.password,
        }

        if self._options is None:
            self._options = SiiAgentOptions(
                system_prompt=self.env.system_prompt,
                max_turns=self.env.max_turns,
                auth_type=self.env.auth_type,
                cwd=str(self.agent_root),
                yolo=True,
                allowed_tools=[],
                model=self.env.target_model,
                env=env_vars,
            )

        if self._session is None:
            self._session = SDKAgentSession(self._options)

    async def send(self, prompt: str) -> Tuple[List[Dict[str, Any]], str]:
        self._ensure_session()
        assert self._session is not None and self._options is not None

        transcript: List[Dict[str, Any]] = []
        assistant_chunks: List[str] = []
        prev_cwd = Path.cwd()
        try:
            os.chdir(self.agent_root)
            async for message in self._session.run(prompt, options=self._options):
                payload = self._normalize_message(message, len(transcript) + 1)
                transcript.append(payload)
                msg_type = payload.get("type")
                if msg_type == "assistant" and payload.get("text"):
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
        finally:
            os.chdir(prev_cwd)

        assistant_text = "\n".join(part.strip() for part in assistant_chunks if part).strip()
        return transcript, assistant_text

    @staticmethod
    def _normalize_message(message: Message, index: int) -> Dict[str, Any]:
        if isinstance(message, AssistantMessage):
            texts = [block.text for block in message.content if isinstance(block, TextBlock)]
            return {"index": index, "type": "assistant", "text": "\n".join(texts)}
        if isinstance(message, TextBlock):
            return {"index": index, "type": "text", "text": message.text}
        data = message.__dict__.copy()
        data["index"] = index
        data.setdefault("type", data.get("role", "unknown"))
        return data


###############################################################################
# Task orchestration
###############################################################################


class Task21Evaluator:
    RECALL_FULL_SCORE = 0.5
    REQUIRED_OUTPUTS = {
        1: "web_info.json",
        2: "ResearchSystem_ChatVsAgent.json",
    }

    def __init__(self, env_config: EnvConfig, description_path: Path):
        self.env = env_config
        self.description_path = description_path
        self.description = read_json(description_path)
        self.task_root = description_path.parent.resolve()
        self.repo_root = self.task_root.parent
        self.model_root = ensure_directory(self.task_root / self.env.model_slug)
        self.workspace_dir = ensure_directory(self.model_root / "workspace")
        self.runner = SiiAgentRunner(env_config, self.model_root)
        self.meta: Dict[str, Any] = {
            "attempt_limit": self.env.attempt_limit,
            "subtasks": [],
        }
        self.rubric_eval_script = (
            self.task_root / "ResearcherBench" / "code" / "rubric_eval" / "main.py"
        )
        self.rubrics_file = (
            self.task_root / "ResearcherBench" / "data" / "eval_data" / "rubric.json"
        )

    async def run(self) -> None:
        subtask_total = int(self.description.get("subtask_count", 0))
        if subtask_total <= 0:
            raise ValueError("description.json must define a positive subtask_count.")

        halt_remaining = False
        for idx in range(subtask_total):
            key = f"subtask{idx + 1}"
            content = self.description.get(key)
            if not content:
                print(f"[TASK21] Skipping {key}: no description provided.")
                continue

            required_file = self._required_output_path(idx + 1)
            require_rubric = self._should_run_rubric(idx + 1)

            if halt_remaining:
                record = self._record_skipped_subtask(key, required_file)
            else:
                record, halt_remaining = await self._run_subtask(
                    subtask_name=key,
                    subtask_index=idx + 1,
                    content=content,
                    required_file=required_file,
                    require_rubric=require_rubric,
                )

            self.meta["subtasks"].append(record)

        recall: Optional[float] = None
        for record in self.meta.get("subtasks", []):
            rubric = record.get("rubric_evaluation")
            if isinstance(rubric, dict) and rubric.get("recall") is not None:
                try:
                    recall = float(rubric["recall"])
                except (TypeError, ValueError):
                    recall = None
                break

        if recall is None:
            normalized = None
            final_score = 0
        else:
            normalized = min(max(recall / self.RECALL_FULL_SCORE, 0.0), 1.0)
            final_score = round(10 * normalized)

        self.meta["rubric_recall"] = recall
        self.meta["rubric_recall_full_score"] = self.RECALL_FULL_SCORE
        self.meta["final_score"] = final_score
        self.meta["final_score_normalized_recall"] = normalized
        meta_path = self.model_root / "meta_eval.json"
        meta_path.write_text(stringify_json(self.meta), encoding="utf-8")
        print(f"[TASK21] Wrote run metadata to {meta_path}")

    async def _run_subtask(
        self,
        subtask_name: str,
        subtask_index: int,
        content: str,
        required_file: Path,
        require_rubric: bool,
    ) -> Tuple[Dict[str, Any], bool]:
        attempt_feedback: Optional[str] = None
        attempt_records: List[Dict[str, Any]] = []
        success = False
        rubric_summary: Optional[AttemptRubricSummary] = None
        required_rel = str(self._relative_to_repo(required_file))

        for attempt in range(1, self.env.attempt_limit + 1):
            self._prepare_required_artifact(required_file)
            prompt = self._compose_prompt(self.workspace_dir, content, attempt_feedback)
            print(
                f"[TASK21][{subtask_name}] Attempt {attempt}/{self.env.attempt_limit} → workspace {self.workspace_dir}"
            )
            await self.runner.send(prompt)

            file_exists = required_file.exists()
            score = 10 if file_exists else 0
            attempt_records.append(
                {
                    "attempt_index": attempt,
                    "score": score,
                    "file_detected": file_exists,
                    "notes": f"Located {required_rel}" if file_exists else f"{required_rel} missing after attempt.",
                }
            )

            if file_exists:
                success = True
                if require_rubric:
                    rubric_summary = self._run_rubric_evaluation(
                        subtask_name=subtask_name,
                        attempt_index=attempt,
                        model_file=required_file,
                        result_dir=self._rubric_result_dir(subtask_index),
                    )
                break

            if attempt < self.env.attempt_limit:
                attempt_feedback = self._missing_file_feedback(subtask_index, required_file)

        record: Dict[str, Any] = {
            "subtask": subtask_name,
            "success": success,
            "total_attempts": len(attempt_records),
            "required_artifact": str(self._relative_to_repo(required_file)),
            "attempts": attempt_records,
            "final_score": 10 if success else 0,
        }
        if rubric_summary:
            record["rubric_evaluation"] = {
                "recall": rubric_summary.recall,
                "covered_points": rubric_summary.covered_points,
                "uncovered_points": rubric_summary.uncovered_points,
            }

        halt_remaining = not success and len(attempt_records) >= self.env.attempt_limit
        if halt_remaining:
            record[
                "note"
            ] = "Required artifact missing after maximum attempts; remaining subtasks will receive 0 points."
        return record, halt_remaining

    def _required_output_path(self, subtask_index: int) -> Path:
        filename = self.REQUIRED_OUTPUTS.get(subtask_index)
        if not filename:
            filename = f"subtask_{subtask_index}_output.json"
        return self.workspace_dir / filename

    def _should_run_rubric(self, subtask_index: int) -> bool:
        return subtask_index == 2

    def _record_skipped_subtask(self, subtask_name: str, required_file: Path) -> Dict[str, Any]:
        return {
            "subtask": subtask_name,
            "success": False,
            "total_attempts": 0,
            "required_artifact": str(self._relative_to_repo(required_file)),
            "final_score": 0,
            "attempts": [
                {
                    "attempt_index": 0,
                    "score": 0,
                    "file_detected": False,
                    "notes": "Skipped because a previous subtask failed to produce its required artifact.",
                }
            ],
            "note": "Skipped due to earlier failure; automatic score of 0.",
        }

    def _prepare_required_artifact(self, path: Path) -> None:
        ensure_directory(path.parent)
        if path.exists():
            path.unlink()

    def _missing_file_feedback(self, subtask_index: int, required_file: Path) -> str:
        rel_path = self._relative_to_repo(required_file)
        base = (
            f"The evaluator could not find the required artifact at '{rel_path}'. "
            "Please repeat the subtask instructions and ensure the file is saved before responding."
        )
        if subtask_index == 1:
            prefix = "prompt没有生成对应的json文件。"
            return textwrap.dedent(
                f"""
                {prefix}
                {base}
                """
            ).strip()
        return base

    def _rubric_result_dir(self, subtask_index: int) -> Path:
        return ensure_directory(self.workspace_dir / f"results_{subtask_index}")

    def _run_rubric_evaluation(
        self,
        subtask_name: str,
        attempt_index: int,
        model_file: Path,
        result_dir: Path,
    ) -> AttemptRubricSummary:
        if not model_file.exists():
            print(
                f"[TASK21][Eval][{subtask_name}] 第{attempt_index}次评测：{model_file.name} does not exist, skipping rubric evaluation."
            )
            return AttemptRubricSummary()
        if not self.rubric_eval_script.exists():
            print(
                f"[TASK21][Eval][{subtask_name}] 第{attempt_index}次评测：rubric evaluator is missing at {self.rubric_eval_script}."
            )
            return AttemptRubricSummary()
        if not self.rubrics_file.exists():
            print(
                f"[TASK21][Eval][{subtask_name}] 第{attempt_index}次评测：rubrics file is missing at {self.rubrics_file}."
            )
            return AttemptRubricSummary()

        print(
            f"[TASK21][Eval][{subtask_name}] >>> 第{attempt_index}次评测：启动rubric评估 (model_file={model_file.name})"
        )
        cmd = [
            sys.executable or "python",
            str(self.rubric_eval_script),
            "--model_file",
            str(model_file),
            "--rubrics_file",
            str(self.rubrics_file),
            "--result_dir",
            str(result_dir),
            "--judge_model",
            "gzy/claude-4-sonnet",
        ]
        try:
            subprocess.run(cmd, check=True, cwd=self.repo_root)
        except subprocess.CalledProcessError as exc:
            print(
                f"[TASK21][Eval][{subtask_name}] 第{attempt_index}次评测：rubric评估失败 ({exc})."
            )
            return AttemptRubricSummary()

        return self._parse_rubric_eval_output(model_file, result_dir)

    def _parse_rubric_eval_output(self, model_file: Path, result_dir: Path) -> AttemptRubricSummary:
        model_name = model_file.stem
        output_file = result_dir / "rubric_eval" / model_name / f"{model_name}_evaluation_results.json"
        if not output_file.exists():
            print(f"[TASK21][Eval] Missing rubric output file: {output_file}")
            return AttemptRubricSummary()
        try:
            payload = json.loads(output_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[TASK21][Eval] Unable to parse rubric results: {output_file}")
            return AttemptRubricSummary()
        if not isinstance(payload, list):
            return AttemptRubricSummary()

        covered: List[str] = []
        uncovered: List[str] = []
        recalls: List[float] = []

        for entry in payload:
            result = entry.get("result") or {}
            coverage = result.get("coverage_results") or []
            if isinstance(coverage, list):
                for item in coverage:
                    point = str(item.get("point", "")).strip()
                    if not point:
                        continue
                    if item.get("covered"):
                        covered.append(point)
                    else:
                        uncovered.append(point)
            recalls.append(self._coerce_float(result.get("recall")))

        deduped_covered = self._dedupe_points(covered)
        deduped_uncovered = self._dedupe_points(uncovered)
        recall_value = sum(recalls) / len(recalls) if recalls else None
        return AttemptRubricSummary(
            recall=recall_value,
            covered_points=deduped_covered,
            uncovered_points=deduped_uncovered,
        )

    @staticmethod
    def _dedupe_points(points: List[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for point in points:
            key = point.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            ordered.append(key)
        return ordered

    @staticmethod
    def _coerce_float(value: Any) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _compose_prompt(self, workspace: Path, subtask_content: str, feedback: Optional[str]) -> str:
        workspace_rel = self._relative_to_repo(workspace)
        lines = [
            f"You must work inside the folder '{workspace_rel}' inside the AgencyBench repository. Do not write outside this directory.",
        ]
        if feedback:
            lines.append(feedback.strip())
        lines.append("Task description:\n" + subtask_content.strip())
        body = "\n\n".join(lines)
        banner = self._workspace_banner(workspace_rel)
        return f"{banner}\n\n{body}"

    def _workspace_banner(self, workspace_rel: Path) -> str:
        workspace_text = str(workspace_rel)
        return textwrap.dedent(
            f"""
            You are already inside the model run directory.
            All commands and edits must happen inside {workspace_text}. Change into this folder before running any command and avoid touching files elsewhere.
            """
        ).strip()

    def _relative_to_repo(self, path: Path) -> Path:
        try:
            return path.resolve().relative_to(self.repo_root)
        except ValueError:
            return path.resolve()


###############################################################################
# Entrypoint
###############################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated evaluator for task21.")
    parser.add_argument("--env", default=".env", help="Path to the env file relative to task21/")
    parser.add_argument(
        "--description", default="description.json", help="Path to description.json relative to task21/"
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
    evaluator = Task21Evaluator(env_config, description_path)
    await evaluator.run()


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
