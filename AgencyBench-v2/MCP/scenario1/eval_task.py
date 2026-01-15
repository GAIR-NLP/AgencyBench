#!/usr/bin/env python3
"""SII Agent SDK driven evaluator for AgencyBench task25.

This orchestrator keeps a single SII Agent SDK session alive while it guides the
candidate agent through the five GitHub workflow subtasks described in
``description.json``. After each turn the evaluator queries GitHub directly to
verify whether the required artifacts (issue, branch, template file, labels,
comments, and pull request) exist. Each subtask may be attempted at most
``SUBTASK_ATTEMPT_LIMIT`` times (default: 2). Every model run receives its own
``task25/<model>/workspace`` directory (no shared template), and the final run
metadata is persisted to ``<task25>/<model>/meta_eval.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from sii_agent_sdk import (
    AssistantMessage,
    CompletedMessage,
    Message,
    SiiAgentOptions,
    TextBlock,
)
from sii_agent_sdk._internal.event_logger import log_message
from sii_agent_sdk._internal.message_parser import parse_message
from sii_agent_sdk.bridge import BridgeProcess
from sii_agent_sdk.errors import ProcessError
from sii_agent_sdk.query import (
    _message_to_session_turns,
    _raise_appropriate_error,
    validate_auth_config,
)
from sii_agent_sdk.session_state import ConversationTurn, SessionState


###############################################################################
# Asyncio compatibility helpers (borrowed from other AgencyBench tasks)
###############################################################################


def _suppress_event_loop_closed_errors() -> None:
    """Mask noisy asyncio warnings triggered by the bridge shutdown."""

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
    """Ensure asyncio streams can handle lengthy JSON lines."""

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


def stringify_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def truncate_text(text: str, limit: int = 1800) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… (truncated, total {len(text)} chars)"


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
                raise ValueError(f"Environment variable '{key}' must be set in task25/.env")
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
            max_turns=int(data.get("SII_MAX_TURNS", "80")),
            attempt_limit=attempt_limit if attempt_limit > 0 else 2,
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
# GitHub verification helpers
###############################################################################


@dataclass
class CheckResult:
    success: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class GitHubVerifier:
    BRANCH_NAME = "config/issue-templates"
    TEMPLATE_PATH = ".github/ISSUE_TEMPLATE/bug_report.md"

    ISSUE_TITLE_KEYWORDS = ["setup standard bug report template", "process improvement"]
    ISSUE_BODY_KEYWORDS = ["issue template", "standardization", "bug tracking"]
    ISSUE_REQUIRED_SECTIONS = ["## Context", "## Goals", "## Expected Outcome"]

    REQUIRED_LABELS_PRESENT = ["meta", "in-progress"]
    REQUIRED_LABELS_ABSENT = ["triage"]

    TEMPLATE_REQUIRED_STRINGS = [
        "name: Bug Report",
        "about: Create a report to help us improve",
        "title: '[BUG] '",
        "labels: bug",
        "assignees: ''",
        "## Describe the Bug",
        "## Reproduction Steps",
        "## Expected Behavior",
        "## Environment",
    ]
    TEMPLATE_NOTE_KEYWORD = "search for existing issues"

    PR_TITLE_KEYWORDS = ["add structured bug report template", "configuration"]
    PR_REQUIRED_SECTIONS = ["## Description", "## Verification"]
    PR_REQUIRED_LABELS = ["configuration", "dependencies"]

    def __init__(self, token: str, org: str, repo: str):
        if not token:
            raise ValueError("MCP_GITHUB_TOKEN must be configured")
        if not org:
            raise ValueError("GITHUB_EVAL_ORG must be configured")
        if not repo:
            raise ValueError("GITHUB_REPO must be configured")

        self.org = org
        self.repo = repo
        self.base_url = f"https://api.github.com/repos/{org}/{repo}"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github+json",
            }
        )
        self._cached_issue: Optional[Dict[str, Any]] = None

    def evaluate(self, subtask_index: int) -> CheckResult:
        handlers = {
            1: self._check_issue_creation,
            2: self._check_branch_exists,
            3: self._check_template_file,
            4: self._check_issue_updates,
            5: self._check_pull_request,
        }
        handler = handlers.get(subtask_index)
        if handler is None:
            return CheckResult(False, f"No verification handler for subtask {subtask_index}.")
        try:
            return handler()
        except requests.RequestException as exc:
            return CheckResult(False, f"GitHub request failed: {exc}")
        except RuntimeError as exc:
            return CheckResult(False, str(exc))

    # ------------------------------------------------------------------ checks

    def _check_issue_creation(self) -> CheckResult:
        issue = self._locate_issue(refresh=True)
        if not issue:
            return CheckResult(False, "Issue with required title keywords not found.")

        issue_body = issue.get("body") or ""
        missing_sections = [section for section in self.ISSUE_REQUIRED_SECTIONS if section not in issue_body]
        missing_keywords = [
            keyword
            for keyword in self.ISSUE_BODY_KEYWORDS
            if keyword.lower() not in issue_body.lower()
        ]
        labels = self._labels(issue)
        if "triage" not in labels or "meta" not in labels:
            return CheckResult(
                False,
                "Issue labels must include both 'triage' and 'meta' at creation.",
                {"issue_number": issue.get("number"), "labels": labels},
            )

        if missing_sections:
            return CheckResult(
                False,
                f"Issue body missing required section(s): {', '.join(missing_sections)}.",
                {"issue_number": issue.get("number")},
            )
        if missing_keywords:
            return CheckResult(
                False,
                f"Issue body missing required keyword(s): {', '.join(missing_keywords)}.",
                {"issue_number": issue.get("number")},
            )

        self._cached_issue = issue
        return CheckResult(
            True,
            f"Issue #{issue['number']} satisfies the required title, sections, keywords, and labels.",
            {"issue_number": issue["number"], "issue_url": issue.get("html_url")},
        )

    def _check_branch_exists(self) -> CheckResult:
        if self._branch_exists(self.BRANCH_NAME):
            return CheckResult(True, f"Branch '{self.BRANCH_NAME}' exists and is reachable.")
        return CheckResult(False, f"Branch '{self.BRANCH_NAME}' not found on GitHub.")

    def _check_template_file(self) -> CheckResult:
        content = self._get_file_content(self.BRANCH_NAME, self.TEMPLATE_PATH)
        if not content:
            return CheckResult(
                False,
                f"Template file '{self.TEMPLATE_PATH}' not found on branch {self.BRANCH_NAME}.",
            )
        missing_bits = [token for token in self.TEMPLATE_REQUIRED_STRINGS if token not in content]
        if missing_bits:
            return CheckResult(
                False,
                f"Template file missing required content: {missing_bits}",
            )
        if self.TEMPLATE_NOTE_KEYWORD not in content.lower():
            return CheckResult(
                False,
                "Template file must remind users to search for existing issues.",
            )
        return CheckResult(
            True,
            f"Template file '{self.TEMPLATE_PATH}' includes all required sections.",
        )

    def _check_issue_updates(self) -> CheckResult:
        issue = self._locate_issue(refresh=True)
        if not issue:
            return CheckResult(False, "Cannot verify labels/comments because the issue is missing.")

        labels = self._labels(issue)
        missing_required = [label for label in self.REQUIRED_LABELS_PRESENT if label not in labels]
        labels_still_present = [label for label in self.REQUIRED_LABELS_ABSENT if label in labels]

        if missing_required or labels_still_present:
            problems = []
            if missing_required:
                problems.append(f"missing labels: {', '.join(missing_required)}")
            if labels_still_present:
                problems.append(f"should remove: {', '.join(labels_still_present)}")
            return CheckResult(
                False,
                f"Issue labels not updated correctly ({'; '.join(problems)}).",
                {"issue_number": issue.get("number"), "labels": labels},
            )

        comments = self._get_issue_comments(issue["number"])
        if not comments:
            return CheckResult(False, "Issue has no comments with the template content.")
        found_template = False
        for comment in comments:
            body = comment.get("body") or ""
            if "name: Bug Report" in body and "## Describe the Bug" in body:
                found_template = True
                break
        if not found_template:
            return CheckResult(
                False,
                "Could not find a comment containing the raw template markdown.",
                {"issue_number": issue["number"], "comment_count": len(comments)},
            )

        return CheckResult(
            True,
            f"Issue #{issue['number']} labels are correct and the template markdown comment is present.",
            {"issue_number": issue["number"]},
        )

    def _check_pull_request(self) -> CheckResult:
        issue = self._locate_issue(refresh=True)
        if not issue:
            return CheckResult(False, "Cannot evaluate PR because the tracking issue is missing.")

        pr = self._find_pr_by_title_keywords()
        if not pr:
            return CheckResult(False, "Pull request with required title keywords not found.")

        body = (pr.get("body") or "").strip()
        missing_sections = [section for section in self.PR_REQUIRED_SECTIONS if section not in body]
        if missing_sections:
            return CheckResult(
                False,
                f"PR body missing section(s): {', '.join(missing_sections)}.",
                {"pr_number": pr.get("number")},
            )

        pr_labels = self._labels(pr)
        missing_labels = [label for label in self.PR_REQUIRED_LABELS if label not in pr_labels]
        if missing_labels:
            return CheckResult(
                False,
                f"PR missing required label(s): {', '.join(missing_labels)}.",
                {"pr_number": pr.get("number"), "labels": pr_labels},
            )

        issue_number = issue.get("number")
        body_lower = body.lower()
        if not issue_number or f"resolves #{issue_number}" not in body_lower:
            return CheckResult(
                False,
                f"PR body must link back using 'Resolves #{issue_number}'.",
                {"pr_number": pr.get("number")},
            )

        if "raw markdown" not in body_lower or "issue comment" not in body_lower:
            return CheckResult(
                False,
                "PR verification section must mention that the raw markdown lives in the issue comments.",
                {"pr_number": pr.get("number")},
            )

        head_ref = pr.get("head", {}).get("ref")
        base_ref = pr.get("base", {}).get("ref")
        if head_ref != self.BRANCH_NAME or base_ref != "main":
            return CheckResult(
                False,
                "PR must target main from config/issue-templates.",
                {
                    "pr_number": pr.get("number"),
                    "head": head_ref,
                    "base": base_ref,
                },
            )

        return CheckResult(
            True,
            f"PR #{pr['number']} links to issue #{issue_number} with correct labels and content.",
            {"pr_number": pr["number"], "issue_number": issue_number},
        )

    # ----------------------------------------------------------------- helpers

    def _branch_exists(self, branch: str) -> bool:
        endpoint = f"branches/{branch}"
        response = self._request(endpoint, allow_missing=True)
        return response is not None

    def _get_file_content(self, branch: str, path: str) -> Optional[str]:
        endpoint = f"contents/{path}?ref={branch}"
        payload = self._request(endpoint, allow_missing=True)
        if not payload:
            return None
        content = payload.get("content")
        if not content:
            return None
        try:
            return base64.b64decode(content).decode("utf-8")
        except Exception:
            return None

    def _locate_issue(self, refresh: bool = False) -> Optional[Dict[str, Any]]:
        if self._cached_issue and not refresh:
            return self._cached_issue

        for state in ("open", "closed"):
            issues = self._request(f"issues?state={state}&per_page=100")
            if not isinstance(issues, list):
                continue
            for issue in issues:
                if "pull_request" in issue:
                    continue
                title = (issue.get("title") or "").lower()
                if all(keyword in title for keyword in self.ISSUE_TITLE_KEYWORDS):
                    self._cached_issue = issue
                    return issue
        return None

    def _get_issue_comments(self, issue_number: int) -> List[Dict[str, Any]]:
        payload = self._request(f"issues/{issue_number}/comments", allow_missing=True)
        if not isinstance(payload, list):
            return []
        return payload

    def _find_pr_by_title_keywords(self) -> Optional[Dict[str, Any]]:
        for state in ("open", "closed"):
            prs = self._request(f"pulls?state={state}&per_page=100")
            if not isinstance(prs, list):
                continue
            for pr in prs:
                title = (pr.get("title") or "").lower()
                if all(keyword in title for keyword in self.PR_TITLE_KEYWORDS):
                    return pr
        return None

    def _labels(self, payload: Dict[str, Any]) -> List[str]:
        labels_data = payload.get("labels") or []
        names: List[str] = []
        for label in labels_data:
            name = label.get("name")
            if isinstance(name, str):
                names.append(name)
        return names

    def _build_url(self, endpoint: Optional[str]) -> str:
        if not endpoint:
            return self.base_url
        endpoint = endpoint.lstrip("/")
        return f"{self.base_url}/{endpoint}"

    def _request(self, endpoint: Optional[str], allow_missing: bool = False) -> Any:
        url = self._build_url(endpoint)
        response = self.session.get(url, timeout=30)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 404 and allow_missing:
            return None
        if response.status_code == 404:
            raise RuntimeError(f"GitHub resource not found: {endpoint}")
        raise RuntimeError(
            f"GitHub API error {response.status_code} for {endpoint}: {response.text}"
        )



###############################################################################
# Persistent SII agent runner
###############################################################################


class SiiAgentRunner:
    """Keep a single bridge process/session alive across multiple prompts."""

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
        except ProcessError as exc:
            await self._handle_bridge_crash(exc)
            try:
                return await self._send_once(prompt)
            except ProcessError as exc2:
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

    async def _handle_bridge_crash(self, exc: ProcessError) -> None:
        reason = str(exc).strip() or "bridge exited unexpectedly"
        print(f"[TASK25] SII bridge exited unexpectedly: {reason}. Restarting session...")
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
# Task orchestration
###############################################################################


class Task25Evaluator:
    def __init__(self, env_config: EnvConfig, description_path: Path, env_data: Dict[str, str]):
        self.env = env_config
        self.task_root = description_path.parent.resolve()
        self.repo_root = self.task_root.parent
        self.description = self._read_description(description_path)
        github_token = (env_data.get("MCP_GITHUB_TOKEN") or os.environ.get("MCP_GITHUB_TOKEN") or "").strip()
        github_org = (env_data.get("GITHUB_EVAL_ORG") or os.environ.get("GITHUB_EVAL_ORG") or "").strip()
        github_repo = (env_data.get("GITHUB_REPO") or os.environ.get("GITHUB_REPO") or "").strip()
        self.github_repo_full = f"{github_org}/{github_repo}"
        self.github_url = f"https://github.com/{self.github_repo_full}"
        self.verifier = GitHubVerifier(github_token, github_org, github_repo)

        self.model_root = ensure_directory(self.task_root / self.env.model_slug)
        self.workspace_dir = self.model_root / "workspace"
        self._reset_model_workspace()
        self._relocate_legacy_sii_state()
        self.runner = SiiAgentRunner(env_config, self.model_root)
        self.meta_path = self.model_root / "meta_eval.json"

        self.meta: Dict[str, Any] = {
            "github_repo": self.github_repo_full,
            "attempt_limit": self.env.attempt_limit,
            "model_root": str(self._relative_to_repo(self.model_root)),
            "workspace": str(self._relative_to_repo(self.workspace_dir)),
            "subtasks": [],
        }
        self.progress_log: List[str] = []

    async def run(self) -> None:
        total = int(self.description.get("subtask_count", 0))
        if total <= 0:
            raise ValueError("description.json must define a positive subtask_count.")

        try:
            for index in range(1, total + 1):
                name = f"subtask{index}"
                instructions = self.description.get(name)
                record = await self._execute_subtask(index, name, instructions)
                self.meta["subtasks"].append(record)
        finally:
            self.meta["progress_summary"] = self.progress_log
            self.meta_path.write_text(stringify_json(self.meta), encoding="utf-8")
            print(f"[TASK25] Evaluation summary written to {self.meta_path}")
            await self.runner.shutdown()

    async def _execute_subtask(
        self,
        subtask_index: int,
        subtask_name: str,
        instructions: Optional[str],
    ) -> Dict[str, Any]:
        if not instructions:
            return {
                "subtask": subtask_name,
                "success": False,
                "total_attempts": 0,
                "note": "description.json does not contain instructions for this subtask.",
            }

        print("\n" + "=" * 78)
        print(f"[TASK25] {subtask_name} – Attempting to satisfy requirements")
        attempt_records: List[Dict[str, Any]] = []
        success = False
        feedback: Optional[str] = None

        for attempt in range(1, self.env.attempt_limit + 1):
            prompt = self._build_prompt(
                subtask_name=subtask_name,
                instructions=instructions,
                attempt=attempt,
                feedback=feedback,
            )
            print(f"[TASK25][{subtask_name}] Attempt {attempt}/{self.env.attempt_limit}")
            _, assistant_text = await self.runner.send(prompt)

            result = self.verifier.evaluate(subtask_index)
            attempt_records.append(
                {
                    "attempt_index": attempt,
                    "success": result.success,
                    "feedback": result.message,
                    "details": result.details,
                    "assistant_response": truncate_text(assistant_text),
                }
            )

            if result.success:
                print(f"[TASK25][{subtask_name}] ✓ {result.message}")
                success = True
                summary_note = result.message
                if subtask_name == "subtask1" and result.details.get("issue_number"):
                    issue_num = result.details["issue_number"]
                    summary_note += f" (issue #{issue_num})"
                self.progress_log.append(f"{subtask_name}: {summary_note}")
                break

            feedback = result.message
            print(f"[TASK25][{subtask_name}] ✗ {result.message}")

        record: Dict[str, Any] = {
            "subtask": subtask_name,
            "description": instructions,
            "success": success,
            "total_attempts": len(attempt_records),
            "attempts": attempt_records,
        }
        if not success:
            record["note"] = feedback or "Subtask requirements not satisfied within attempt limit."
        return record

    def _build_prompt(
        self,
        subtask_name: str,
        instructions: str,
        attempt: int,
        feedback: Optional[str],
    ) -> str:
        workspace_rel = self._relative_to_repo(self.workspace_dir)
        lines = [
            f"You are working inside the dedicated evaluation workspace: {workspace_rel}",
            "Keep every file you touch inside this directory. The evaluator will reuse the same session, so persist notes as needed.",
            f"Repository under evaluation: {self.github_url}",
            "Use MCP_GITHUB_TOKEN for authenticated GitHub API access when necessary.",
            f"Current goal: {subtask_name} (attempt {attempt}/{self.env.attempt_limit}).",
            "After completing the work, summarize what you changed and reference any issue/PR numbers explicitly.",
            "The evaluator will query GitHub directly after your response, so ensure all commits, issues, labels, and PRs are pushed.",
        ]
        if self.progress_log:
            progress = "\n".join(f"- {item}" for item in self.progress_log)
            lines.append(f"Progress so far:\n{progress}")
        reset_notice = self.runner.consume_reset_notice()
        if reset_notice:
            lines.append(
                f"Session note: the previous SDK session closed unexpectedly ({reset_notice}). "
                "A fresh session has started; reopen shells/editors inside the workspace."
            )
        if feedback:
            lines.append(f"Previous verification feedback: {feedback}")
        lines.append("Subtask requirements:\n" + instructions.strip())

        banner = textwrap.dedent(
            """
            This is a single persistent SII Agent SDK session. Continue where you left off and avoid spawning additional IDE windows.
            """
        ).strip()
        body = "\n\n".join(lines)
        return f"{banner}\n\n{body}"

    def _reset_model_workspace(self) -> None:
        ensure_directory(self.model_root)
        if self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir)
        ensure_directory(self.workspace_dir)

    def _relocate_legacy_sii_state(self) -> None:
        legacy_state = self.task_root / ".sii"
        target_state = self.model_root / ".sii"
        if legacy_state.exists():
            if target_state.exists():
                shutil.rmtree(legacy_state)
            else:
                shutil.move(str(legacy_state), str(target_state))

    def _read_description(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def _relative_to_repo(self, path: Path) -> Path:
        try:
            return path.resolve().relative_to(self.repo_root)
        except ValueError:
            return path.resolve()


###############################################################################
# Entrypoint
###############################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated evaluator for task25.")
    parser.add_argument("--env", default=".env", help="Path to the env file relative to task25/")
    parser.add_argument(
        "--description",
        default="description.json",
        help="Path to description.json relative to task25/",
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

    evaluator = Task25Evaluator(env_config, description_path, env_data)
    await evaluator.run()


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
