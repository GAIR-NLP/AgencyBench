"""Automated evaluation pipeline for AgencyBench task23.

This script reproduces the manual benchmarking workflow described in
`description.json` by orchestrating SII Agent SDK interactions and LLM-based
judging inside a single session. The microsoft/autogen repository is cloned once
under the model run directory (e.g., `claude-sonnet-4.5/autogen`) and reused
across every subtask/attempt. Before each attempt the repo is reset to the
requested commit, the SII agent is instructed to work inside that fixed path,
patches and judge outputs are stored inside the attempt folder, and metadata is
aggregated in `meta_eval.json`.

Key behaviors:
* Environment variables come from `.env` (see task21/.env for reference).
* One SII Agent SDK session services every attempt to preserve tool state.
* Repo is cloned once; later attempts only checkout different commits.
* After each attempt we write `patch_{subtask}.diff` and judge results inside
  the attempt directory, solicit an LLM judge score, and either proceed or
  surface feedback to the agent.
* Attempt metadata and judge outputs accumulate in `meta_eval.json`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from sii_agent_sdk import (
    AssistantMessage,
    Message,
    SiiAgentOptions,
    TextBlock,
    SiiAgentSession as SDKAgentSession,
)


###############################################################################
# Generic helpers
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


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def derive_model_name(identifier: str) -> str:
    raw = identifier.strip()
    if "/" in raw:
        raw = raw.split("/")[-1]
    sanitized = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in raw)
    sanitized = sanitized.strip("._-")
    return sanitized or "model"


def run_command(cmd: List[str], cwd: Optional[Path] = None) -> str:
    process = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"Command {' '.join(cmd)} failed with code {process.returncode}: {process.stderr.strip()}"
        )
    return process.stdout


def stringify_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    judge_api_key: str
    judge_api_base: str
    judge_model: str
    score_threshold: float
    repo_url: str
    repo_clone_retries: int
    model_slug: str = field(init=False)

    def __post_init__(self) -> None:
        self.model_slug = derive_model_name(self.target_model)

    @classmethod
    def from_env(cls, data: Dict[str, str]) -> "EnvConfig":
        def require(key: str) -> str:
            value = (data.get(key) or os.environ.get(key) or "").strip()
            if not value:
                raise ValueError(f"Environment variable '{key}' is required for eval_task23.")
            return value

        return cls(
            username=require("SII_USERNAME"),
            password=require("SII_PASSWORD"),
            agent_api_key=require("SII_AGENT_API_KEY"),
            agent_api_base=data.get("SII_AGENT_API_BASE_URL", "https://openrouter.ai/api/v1").strip(),
            target_model=require("SII_TARGET_MODEL"),
            auth_type=data.get("SII_AUTH_TYPE", "USE_OPENAI_WITH_SII_TOOLS").strip(),
            system_prompt=data.get("SII_SYSTEM_PROMPT", "").strip(),
            max_turns=int(data.get("SII_MAX_TURNS", "80")),
            attempt_limit=int(
                data.get("SUBTASK_ATTEMPT_LIMIT") or data.get("MAX_SUBTASK_ATTEMPTS") or "2"
            ),
            judge_api_key=require("JUDGE_API_KEY"),
            judge_api_base=data.get("JUDGE_API_BASE_URL", "https://openrouter.ai/api/v1").strip(),
            judge_model=require("JUDGE_MODEL"),
            score_threshold=float(data.get("JUDGE_SCORE_THRESHOLD", "0.8")),
            repo_url=data.get("TARGET_REPO_URL", "https://github.com/microsoft/autogen").strip(),
            repo_clone_retries=int(data.get("REPO_CLONE_RETRIES", "3")),
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
        os.environ.setdefault(
            "MODEL_API_BASE_URL", os.environ.get("MODEL_API_BASE_URL", self.agent_api_base)
        )
        os.environ.setdefault("MODEL_API_MODEL", os.environ.get("MODEL_API_MODEL", self.target_model))


###############################################################################
# SII Agent session management
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
# LLM Judge client
###############################################################################


class LLMJudgeClient:
    def __init__(self, env_config: EnvConfig, template: str, stage_patches: Dict[str, str]):
        self.env = env_config
        self.template = template
        self.stage_patches = stage_patches

    def evaluate(
        self,
        *,
        base_commit: str,
        diff_patch: str,
        test_query: str,
    ) -> Dict[str, Any]:
        ground_truth_patch = self.stage_patches.get(base_commit)
        if ground_truth_patch is None:
            raise KeyError(f"No ground-truth patch for commit {base_commit}")

        prompt = (
            self.template.replace("{{ground_truth_patch}}", ground_truth_patch)
            .replace("{{diff_patch}}", diff_patch or " ")
            .replace("{{test_query}}", test_query)
        )

        response = self._call_openrouter(prompt)
        parsed = self._parse_response(response)
        return {
            "prompt": prompt,
            "raw_response": response,
            "score": parsed.get("score", 0.0),
            "comment": parsed.get("comment", "").strip(),
        }

    def _call_openrouter(self, prompt: str) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.env.judge_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.env.judge_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 512,
        }
        url = self.env.judge_api_base.rstrip("/") + "/chat/completions"
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _parse_response(response: Dict[str, Any]) -> Dict[str, Any]:
        content = ""
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = (message.get("content") or "").strip()
        if not content:
            return {"score": 0.0, "comment": "Judge returned empty response."}

        parsed = try_parse_json(content)
        if isinstance(parsed, dict):
            try:
                score = float(parsed.get("score", 0))
            except (TypeError, ValueError):
                score = 0.0
            comment = str(parsed.get("comment", "")).strip()
            return {"score": score, "comment": comment}

        return {"score": 0.0, "comment": f"Unable to parse judge JSON: {content}"}


def try_parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


###############################################################################
# Task orchestration
###############################################################################


class Task23Evaluator:
    SUBTASK_BASE_COMMITS = {
        1: "a8da3854c00cf8a4517d2572668a8b45077c63bc",
        2: "80954e4b8d0752fd4772f339dee419fbf1debc6f",
        3: "50ac5476377c1b41330589a6cfc5c4e65b93079f",
    }

    def __init__(
        self,
        env_config: EnvConfig,
        description_path: Path,
        query_path: Path,
        evaluator_template_path: Path,
    ):
        self.env = env_config
        self.description_path = description_path
        self.description = read_json(description_path)
        self.query_path = query_path
        self.template_text = evaluator_template_path.read_text(encoding="utf-8")
        self.stage_ground_truth = self._load_stage_ground_truth(query_path)
        self.task_root = description_path.parent.resolve()
        self.model_root = ensure_directory(self.task_root / self.env.model_slug)
        self.workspace_root = ensure_directory(self.model_root / "workspace")
        self.repo_path = self.model_root / "autogen"
        self._repo_fetched = False
        self.meta: Dict[str, Any] = {
            "attempt_limit": self.env.attempt_limit,
            "score_threshold": self.env.score_threshold,
            "subtasks": [],
            "repo_path": str(self.repo_path.relative_to(self.task_root)),
        }
        self.runner = SiiAgentRunner(env_config, self.model_root)
        self.judge = LLMJudgeClient(env_config, self.template_text, self.stage_ground_truth)

    async def run(self) -> None:
        total = int(self.description.get("subtask_count", 0))
        if total <= 0:
            raise ValueError("description.json must specify subtask_count > 0")

        for idx in range(1, total + 1):
            key = f"subtask{idx}"
            query_text = self.description.get(key)
            if not query_text:
                self.meta["subtasks"].append(
                    {
                        "index": idx,
                        "name": key,
                        "status": "skipped",
                        "reason": "No query text found in description.json.",
                    }
                )
                continue

            base_commit = self.SUBTASK_BASE_COMMITS.get(idx)
            if not base_commit:
                self.meta["subtasks"].append(
                    {
                        "index": idx,
                        "name": key,
                        "status": "skipped",
                        "reason": "Automation is configured only for subtasks 1-3.",
                    }
                )
                continue

            record = await self._run_subtask(
                subtask_index=idx,
                subtask_name=key,
                base_commit=base_commit,
                query_text=query_text,
            )
            self.meta["subtasks"].append(record)

        meta_path = self.model_root / "meta_eval.json"
        meta_path.write_text(stringify_json(self.meta), encoding="utf-8")
        self._emit_meta_output(meta_path)

    async def _run_subtask(
        self,
        *,
        subtask_index: int,
        subtask_name: str,
        base_commit: str,
        query_text: str,
    ) -> Dict[str, Any]:
        attempts: List[Dict[str, Any]] = []
        success = False
        feedback: Optional[str] = None

        repo_rel = str(self.repo_path.relative_to(self.task_root))

        for attempt in range(1, self.env.attempt_limit + 1):
            attempt_dir = self._prepare_attempt_dir(subtask_index, attempt)
            self._prepare_repo_for_commit(base_commit)
            prompt = self._compose_prompt(
                attempt_dir=attempt_dir,
                repo_path=self.repo_path,
                subtask_name=subtask_name,
                query_text=query_text,
                base_commit=base_commit,
                attempt=attempt,
                feedback=feedback,
            )
            transcript, assistant_text = await self.runner.send(prompt)
            self._persist_transcript(attempt_dir, transcript, assistant_text)

            diff_text = self._generate_patch(self.repo_path)
            patch_path = self._write_patch(subtask_index, diff_text, attempt_dir)
            judge_result = self.judge.evaluate(
                base_commit=base_commit,
                diff_patch=diff_text,
                test_query=query_text,
            )
            score = float(judge_result.get("score") or 0.0)
            comment = judge_result.get("comment", "")
            status = "success" if score >= self.env.score_threshold else "retry"
            judge_output_path = attempt_dir / "judge_result.json"
            judge_output_path.write_text(stringify_json(judge_result), encoding="utf-8")
            self._emit_attempt_output(
                subtask_index=subtask_index,
                attempt=attempt,
                diff_text=diff_text,
                judge_result=judge_result,
            )
            attempts.append(
                {
                    "attempt_index": attempt,
                    "attempt_dir": str(attempt_dir.relative_to(self.task_root)),
                    "repo_path": repo_rel,
                    "patch_file": str(patch_path.relative_to(self.task_root)),
                    "diff_lines": len(diff_text.splitlines()),
                    "judge_score": score,
                    "judge_comment": comment,
                    "judge_raw": judge_result.get("raw_response"),
                    "judge_result_file": str(judge_output_path.relative_to(self.task_root)),
                    "status": status,
                }
            )

            if score >= self.env.score_threshold:
                success = True
                break

            feedback = self._build_feedback(comment, patch_path)

        return {
            "index": subtask_index,
            "name": subtask_name,
            "base_commit": base_commit,
            "attempts": attempts,
            "success": success,
            "final_score": attempts[-1]["judge_score"] if attempts else 0.0,
        }

    def _prepare_attempt_dir(self, subtask_index: int, attempt: int) -> Path:
        return ensure_directory(self.workspace_root / f"subtask{subtask_index}" / f"attempt{attempt}")

    def _ensure_repo_cloned(self) -> None:
        if (self.repo_path / ".git").exists():
            return
        if self.repo_path.exists():
            shutil.rmtree(self.repo_path)

        for attempt in range(1, self.env.repo_clone_retries + 1):
            try:
                run_command(["git", "clone", self.env.repo_url, str(self.repo_path)], cwd=self.model_root)
                return
            except RuntimeError as exc:
                if attempt >= self.env.repo_clone_retries:
                    raise RuntimeError(
                        f"git clone failed after {attempt} attempts. "
                        f"Please manually clone {self.env.repo_url} into {self.repo_path} and rerun.\n{exc}"
                    ) from exc
                wait = min(30, 5 * attempt)
                time.sleep(wait)

    def _prepare_repo_for_commit(self, commit: str) -> None:
        self._ensure_repo_cloned()
        run_command(["git", "reset", "--hard"], cwd=self.repo_path)
        run_command(["git", "clean", "-fd"], cwd=self.repo_path)
        run_command(["git", "checkout", "--detach", commit], cwd=self.repo_path)
        run_command(["git", "reset", "--hard", commit], cwd=self.repo_path)

    def _compose_prompt(
        self,
        *,
        attempt_dir: Path,
        repo_path: Path,
        subtask_name: str,
        query_text: str,
        base_commit: str,
        attempt: int,
        feedback: Optional[str],
    ) -> str:
        attempt_rel = attempt_dir.relative_to(self.task_root)
        repo_rel = repo_path.relative_to(self.task_root)
        sections = [
            "You are participating in AgencyBench task23 automated evaluation.",
            f"Work ONLY inside '{repo_rel}'. This folder is reused across attempts; it has been reset to commit {base_commit}.",
            "Before editing, change directories into that repo and keep all edits there.",
            "When inspecting large files, obey the benchmark rule: use 'head' to read no more than the first 10 lines or use 'grep'. Do not run scripts such as python/pip/docker.",
            "After implementing the query requirements, ensure files are saved. No git commit is required.",
            f"Attempt artifacts (patch/judge logs) belong in '{attempt_rel}'.",
            f"Current attempt: {subtask_name} attempt {attempt} (repo path: {repo_rel}).",
        ]
        if feedback:
            sections.append("Previous evaluator feedback:\n" + feedback.strip())
        sections.append("Task query (verbatim from description.json):\n" + query_text.strip())
        return "\n\n".join(sections)

    def _persist_transcript(self, workspace: Path, transcript: List[Dict[str, Any]], summary: str) -> None:
        payload = {"transcript": transcript, "assistant_summary": summary}
        (workspace / "transcript.json").write_text(stringify_json(payload), encoding="utf-8")

    def _generate_patch(self, repo_path: Path) -> str:
        try:
            output = run_command(["git", "status", "--short"], cwd=repo_path)
        except RuntimeError as exc:
            raise RuntimeError(f"git status failed in {repo_path}: {exc}") from exc
        if not output.strip():
            return ""
        diff = run_command(["git", "diff"], cwd=repo_path)
        return diff

    def _write_patch(self, subtask_index: int, diff_text: str, attempt_dir: Path) -> Path:
        patch_name = f"patch_{subtask_index}.diff"
        patch_path = attempt_dir / patch_name
        patch_path.write_text(diff_text, encoding="utf-8")
        return patch_path

    def _build_feedback(self, comment: str, patch_path: Path) -> str:
        note = comment or "Judge returned no feedback. Please revisit the requirements."
        return textwrap.dedent(
            f"""
            LLM judge score was below threshold ({self.env.score_threshold}).
            Patch file: {patch_path.name}
            Feedback from judge:
            {note}
            """
        ).strip()

    @staticmethod
    def _load_stage_ground_truth(query_path: Path) -> Dict[str, str]:
        payload = read_json(query_path)
        stage_map: Dict[str, str] = {}
        for stage in payload.get("stages", []):
            commit = stage.get("base_commit_sha")
            patch = stage.get("ground_truth_patch")
            if commit and patch:
                stage_map[commit] = patch
        return stage_map

    def _emit_attempt_output(
        self,
        *,
        subtask_index: int,
        attempt: int,
        diff_text: str,
        judge_result: Dict[str, Any],
    ) -> None:
        label = f"SUBTASK_{subtask_index}_ATTEMPT_{attempt}"
        diff_body = diff_text.strip() or "[EMPTY_DIFF]"
        print(f"=== PATCH_{label} ===")
        print(diff_body)
        print(f"=== JUDGE_{label} ===")
        print(stringify_json(judge_result))

    def _emit_meta_output(self, meta_path: Path) -> None:
        print("=== META_EVAL ===")
        print(meta_path.read_text(encoding="utf-8"))


###############################################################################
# Entrypoint
###############################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated evaluator for task23.")
    parser.add_argument("--env", default=".env", help="Path to env file (relative to task23/).")
    parser.add_argument("--description", default="description.json", help="Path to description.json.")
    parser.add_argument(
        "--query",
        default="query_microsoft_autogen_379_microsoft_autogen_autogenerated-379_416d61b8_2025-11-14t02_22_18.733791_00_00.json",
        help="Path to the query metadata JSON.",
    )
    parser.add_argument("--template", default="evaluator.j2", help="Path to evaluator prompt template.")
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    task_root = Path(__file__).resolve().parent

    env_path = Path(args.env)
    if not env_path.is_absolute():
        env_path = task_root / env_path
    env_data = load_env_file(env_path)
    env_config = EnvConfig.from_env(env_data)
    env_config.inject_defaults()

    description_path = Path(args.description)
    if not description_path.is_absolute():
        description_path = task_root / description_path
    query_path = Path(args.query)
    if not query_path.is_absolute():
        query_path = task_root / query_path
    template_path = Path(args.template)
    if not template_path.is_absolute():
        template_path = task_root / template_path

    evaluator = Task23Evaluator(env_config, description_path, query_path, template_path)
    await evaluator.run()


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
