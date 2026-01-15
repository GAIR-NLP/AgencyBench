#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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
class RubricResult:
    subtask: str
    score: int
    notes: List[str]
    checks: Dict[str, bool]


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
# Rubric evaluation
# --------------------------------------------------------------------------- #


# class RubricEvaluator:
#     """Evaluate workspace outputs according to the Task19 NBA rubric."""

#     def evaluate(self, subtask: str, workspace: Path) -> RubricResult:
#         dispatch = {
#             "subtask1": self._eval_subtask1,
#             "subtask2": self._eval_subtask2,
#             "subtask3": self._eval_subtask3,
#             "subtask4": self._eval_subtask4,
#             "subtask5": self._eval_subtask5,
#         }
#         handler = dispatch.get(subtask)
#         if not handler:
#             return RubricResult(subtask=subtask, score=0, notes=["Unknown subtask"], checks={})
#         return handler(workspace)

#     def _eval_subtask1(self, workspace: Path) -> RubricResult:
#         path = self._answer_path(workspace, "subtask1.md")
#         text = self._read_text(path)
#         if text is None:
#             return self._missing("subtask1", f"Missing report: {path}")

#         lower = text.lower()
#         player = "paul george" in lower
#         all_star = "all-star" in lower or "all star" in lower
#         all_def = "all-defensive" in lower or "all defensive" in lower
#         honors = all_star and all_def
#         season = bool(re.search(r"2018[\u2013-]?19", text))
#         trade = "trade" in lower and ("pacers" in lower or "thunder" in lower or "okc" in lower)
#         george_hill = "george hill" in lower
#         all_rookie = "all-rookie" in lower or "all rookie" in lower
#         beatles = "beatles" in lower or "beatle" in lower
#         checks = {
#             "player": player,
#             "season": season,
#             "dual_honors": honors,
#             "trade_context": trade,
#             "george_hill": george_hill,
#             "all_rookie": all_rookie,
#             "beatles": beatles,
#         }

#         notes: List[str] = []
#         if not player or not honors or not george_hill:
#             if not player:
#                 notes.append("Answer does not center on Paul George.")
#             if not honors:
#                 notes.append("All-Star and All-Defensive First Team linkage missing.")
#             if not george_hill:
#                 notes.append("George Hill reference absent, so trade context is unclear.")
#             return RubricResult(subtask="subtask1", score=0, notes=notes, checks=checks)

#         missing_detail_count = sum(
#             0 if ok else 1 for ok in (season, trade, all_rookie, beatles)
#         )
#         if missing_detail_count == 0:
#             score = 10
#         elif missing_detail_count == 1:
#             score = 8
#             if not season:
#                 notes.append("2018–19 season not clearly cited.")
#             if not trade:
#                 notes.append("Trade narrative is incomplete.")
#             if not all_rookie:
#                 notes.append("George Hill's All-Rookie credential not mentioned.")
#             if not beatles:
#                 notes.append("Beatles name linkage omitted.")
#         else:
#             score = 5
#             if not season:
#                 notes.append("Missing explicit 2018–19 season reference.")
#             if not trade:
#                 notes.append("Trade details between Pacers and Thunder are absent.")
#             if not all_rookie:
#                 notes.append("All-Rookie status for George Hill not shown.")
#             if not beatles:
#                 notes.append("Beatles reference missing.")
#         return RubricResult(subtask="subtask1", score=score, notes=notes, checks=checks)

#     def _eval_subtask2(self, workspace: Path) -> RubricResult:
#         path = self._answer_path(workspace, "subtask2.md")
#         text = self._read_text(path)
#         if text is None:
#             return self._missing("subtask2", f"Missing report: {path}")
#         lower = text.lower()
#         player = "james harden" in lower
#         mvp = "mvp" in lower and bool(re.search(r"2017[\u2013-]?18", text))
#         scoring = ("scoring title" in lower) or ("scoring champion" in lower) or ("scoring leader" in lower)
#         assist = ("assist title" in lower) or ("assist leader" in lower) or ("led the league in assists" in lower)
#         assist_season = bool(re.search(r"2016[\u2013-]?17", text))
#         steven_adams = "steven adams" in lower
#         all_rookie = "all-rookie" in lower or "all rookie" in lower
#         scoring_count = bool(re.search(r"(three|3)\s+(?:time|times)?\s*(?:scoring champion|scoring title|scoring leader)", lower))
#         checks = {
#             "player": player,
#             "mvp": mvp,
#             "scoring_titles": scoring,
#             "assist_title": assist,
#             "assist_season": assist_season,
#             "steven_adams": steven_adams,
#             "all_rookie": all_rookie,
#             "scoring_count": scoring_count,
#         }
#         notes: List[str] = []
#         if not player or not (mvp and scoring and assist):
#             if not player:
#                 notes.append("Answer is not centered on James Harden.")
#             if not mvp or not scoring or not assist:
#                 notes.append("MVP, scoring titles, and assist leader combo not demonstrated.")
#             return RubricResult("subtask2", 0, notes, checks)

#         if not steven_adams or not all_rookie:
#             if not steven_adams:
#                 notes.append("Steven Adams involvement in the trade is missing.")
#             if not all_rookie:
#                 notes.append("Thunder All-Rookie confirmation absent.")
#             return RubricResult("subtask2", 5, notes, checks)

#         detail_missing = []
#         if not assist_season:
#             detail_missing.append("2016–17 assist title not cited.")
#         if not scoring_count:
#             detail_missing.append("Three scoring titles confirmation missing.")
#         if not mvp:
#             detail_missing.append("2017–18 MVP year not clearly stated.")

#         if not detail_missing:
#             score = 10
#         else:
#             score = 8
#             notes.extend(detail_missing)
#         return RubricResult("subtask2", score, notes, checks)

#     def _eval_subtask3(self, workspace: Path) -> RubricResult:
#         path = self._answer_path(workspace, "subtask3.md")
#         text = self._read_text(path)
#         if text is None:
#             return self._missing("subtask3", f"Missing report: {path}")
#         lower = text.lower()
#         player = "kevin durant" in lower
#         roy = ("rookie of the year" in lower) or "roy" in lower
#         scoring_titles = any(
#             phrase in lower
#             for phrase in (
#                 "four scoring titles",
#                 "four-time scoring champion",
#                 "four time scoring champion",
#                 "4 scoring titles",
#             )
#         )
#         finals_mvp = "finals mvp" in lower
#         mvp_year = bool(re.search(r"2013[\u2013-]?14", text)) or "2014 mvp" in lower
#         free_agency = "free agency" in lower or "free agent" in lower
#         oladipo = "victor oladipo" in lower
#         all_star_ref = "all-star" in lower or "all star" in lower
#         checks = {
#             "player": player,
#             "roy": roy,
#             "scoring_titles": scoring_titles,
#             "finals_mvp": finals_mvp,
#             "mvp_year": mvp_year,
#             "free_agency": free_agency,
#             "oladipo": oladipo,
#             "all_star_link": all_star_ref,
#         }
#         notes: List[str] = []
#         if not player or not (roy and scoring_titles and finals_mvp):
#             if not player:
#                 notes.append("Answer is not about Kevin Durant.")
#             if not roy:
#                 notes.append("Rookie of the Year achievement missing.")
#             if not scoring_titles:
#                 notes.append("Four scoring titles not demonstrated.")
#             if not finals_mvp:
#                 notes.append("Finals MVP reference missing.")
#             return RubricResult("subtask3", 0, notes, checks)

#         if not oladipo or not all_star_ref:
#             if not oladipo:
#                 notes.append("Victor Oladipo exchange not covered.")
#             if not all_star_ref:
#                 notes.append("Thunder All-Star status not mentioned.")
#             return RubricResult("subtask3", 5, notes, checks)

#         detail_gaps = []
#         if not mvp_year:
#             detail_gaps.append("2013–14 MVP timing not explicit.")
#         if not free_agency:
#             detail_gaps.append("Free-agency framing absent.")

#         if not detail_gaps:
#             score = 10
#         else:
#             score = 8
#             notes.extend(detail_gaps)
#         return RubricResult("subtask3", score, notes, checks)

#     def _eval_subtask4(self, workspace: Path) -> RubricResult:
#         path = self._answer_path(workspace, "subtask4.md")
#         text = self._read_text(path)
#         if text is None:
#             return self._missing("subtask4", f"Missing report: {path}")
#         lower = text.lower()
#         player = "klay thompson" in lower
#         finals = "finals" in lower
#         record = ("no free throws" in lower and "three" in lower) or "most threes without free throws" in lower
#         fifty = "50-point" in lower or "50 point" in lower
#         scoring_cap = "22 ppg" in lower or "22 points per game" in lower or "never averaged more than 22" in lower
#         poole = "jordan poole" in lower
#         lottery = "lottery pick" in lower or "lottery selection" in lower
#         all_conf = "all-conference first team" in lower or "first-team all-conference" in lower
#         checks = {
#             "player": player,
#             "finals": finals,
#             "record": record,
#             "fifty_point": fifty,
#             "scoring_cap": scoring_cap,
#             "jordan_poole": poole,
#             "lottery": lottery,
#             "all_conference": all_conf,
#         }
#         notes: List[str] = []
#         if not player or not (finals and record and fifty):
#             if not player:
#                 notes.append("Answer is not focused on Klay Thompson.")
#             if not finals:
#                 notes.append("Multiple Finals runs are not mentioned.")
#             if not record:
#                 notes.append("Playoff three-point record without free throws absent.")
#             if not fifty:
#                 notes.append("50-point playoff games missing.")
#             return RubricResult("subtask4", 0, notes, checks)

#         if not poole or not (lottery and all_conf):
#             if not poole:
#                 notes.append("Jordan Poole link absent.")
#             if not lottery:
#                 notes.append("Lottery-pick detail missing.")
#             if not all_conf:
#                 notes.append("All-Conference First Team credential missing.")
#             return RubricResult("subtask4", 5, notes, checks)

#         if not scoring_cap:
#             notes.append("Never-averaged-above-22 PPG fact not documented.")
#             return RubricResult("subtask4", 8, notes, checks)
#         return RubricResult("subtask4", 10, notes, checks)

#     def _eval_subtask5(self, workspace: Path) -> RubricResult:
#         path = self._answer_path(workspace, "subtask_summary.json")
#         if not path.exists():
#             return self._missing("subtask5", f"Missing summary JSON: {path}")
#         try:
#             data = json.loads(path.read_text(encoding="utf-8"))
#         except json.JSONDecodeError:
#             return self._missing("subtask5", "JSON parse error in summary report.")

#         entries: List[Dict[str, Any]]
#         if isinstance(data, list):
#             entries = data
#         elif isinstance(data, dict):
#             entries = data.get("subtasks") if isinstance(data.get("subtasks"), list) else []
#         else:
#             entries = []

#         expected = [
#             "Paul George",
#             "James Harden",
#             "Kevin Durant",
#             "Klay Thompson",
#         ]
#         checks = {"structure_ok": bool(entries)}
#         if not entries or len(entries) < 4:
#             return RubricResult("subtask5", 0, ["JSON list missing one or more subtasks."], checks)

#         ground_truth_ok = True
#         match_fields_ok = True
#         conditions_present = True
#         descriptive_conditions = True
#         gt_seen: List[str] = []

#         for entry in entries:
#             gt = entry.get("ground_truth")
#             gt_seen.append(gt or "")
#             if gt not in expected:
#                 ground_truth_ok = False
#             if "matches" not in entry or not isinstance(entry.get("matches"), bool):
#                 match_fields_ok = False
#             conds = entry.get("conditions")
#             if not isinstance(conds, list):
#                 conditions_present = False
#             else:
#                 if not conds:
#                     descriptive_conditions = False
#                 elif not all(isinstance(item, str) and item.strip() for item in conds):
#                     descriptive_conditions = False

#         checks.update(
#             {
#                 "ground_truths": ground_truth_ok and sorted(gt_seen) == sorted(expected),
#                 "matches_field": match_fields_ok,
#                 "conditions_field": conditions_present,
#                 "conditions_detail": descriptive_conditions,
#             }
#         )

#         if not checks["ground_truths"]:
#             return RubricResult(
#                 "subtask5",
#                 0,
#                 ["Ground truth list missing or mismatched official names."],
#                 checks,
#             )
#         if not match_fields_ok or not conditions_present:
#             return RubricResult(
#                 "subtask5",
#                 5,
#                 ["Match flags or condition arrays are incomplete."],
#                 checks,
#             )
#         if not descriptive_conditions:
#             return RubricResult(
#                 "subtask5",
#                 8,
#                 ["Conditions present but lack descriptive per-clause notes."],
#                 checks,
#             )
#         return RubricResult("subtask5", 10, [], checks)

class RubricEvaluator:
    """Evaluate workspace outputs according to the Task19 NBA rubric."""

    def evaluate(self, subtask: str, workspace: Path) -> RubricResult:
        dispatch = {
            "subtask1": self._eval_subtask1,
            "subtask2": self._eval_subtask2,
            "subtask3": self._eval_subtask3,
            "subtask4": self._eval_subtask4,
            "subtask5": self._eval_subtask5,
        }
        handler = dispatch.get(subtask)
        if not handler:
            return RubricResult(subtask=subtask, score=0, notes=["Unknown subtask"], checks={})
        return handler(workspace)

    def _read_text(self, path: Path) -> str | None:
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def _answer_path(self, workspace: Path, filename: str) -> Path:
        # Adapting to the deliverable paths mentioned in the prompt
        # Some deliverables are workspace/Subtask1.md, others workspace/answers/subtask2.md
        # The prompt implies loose structure, but let's check standard locations
        candidates = [
            workspace / filename,
            workspace / "answers" / filename,
        ]
        for c in candidates:
            if c.exists():
                return c
        return workspace / filename # Default return

    def _missing(self, subtask: str, msg: str) -> RubricResult:
        return RubricResult(subtask=subtask, score=0, notes=[msg], checks={})

    def _eval_subtask1(self, workspace: Path) -> RubricResult:
        path = self._answer_path(workspace, "subtask1.md")
        text = self._read_text(path) or self._read_text(self._answer_path(workspace, "Subtask1.md"))
        if text is None:
            return self._missing("subtask1", f"Missing report: {path.name}")

        lower = text.lower()
        player = "paul george" in lower
        
        # Criteria: 2018-19 season All-Star + All-Defensive First Team
        season_match = bool(re.search(r"2018[\u2013-]?19", text))
        honors = ("all-star" in lower or "all star" in lower) and \
                 ("all-defensive" in lower or "all defensive" in lower)
        
        # Trade Criteria: Trade to Clippers, Shai Gilgeous-Alexander, Alexander the Great
        trade = "trade" in lower and ("clippers" in lower or "la" in lower)
        sga = "shai" in lower or "gilgeous-alexander" in lower
        alexander_great = "alexander" in lower and "great" in lower
        all_rookie = "all-rookie" in lower or "all rookie" in lower

        checks = {
            "player": player,
            "season_2018_19": season_match,
            "dual_honors": honors,
            "trade_clippers": trade,
            "sga_mentioned": sga,
            "alexander_great_ref": alexander_great,
            "sga_all_rookie": all_rookie,
        }

        notes: List[str] = []
        if not player:
            notes.append("Answer does not center on Paul George.")
            return RubricResult(subtask="subtask1", score=0, notes=notes, checks=checks)
        
        # If player is correct, evaluate details
        missing_count = 0
        
        if not (season_match and honors):
            missing_count += 1
            notes.append("2018–19 All-Star/All-Defensive First Team connection unclear.")
            
        if not (sga and trade):
            missing_count += 1
            notes.append("Trade details involving Shai Gilgeous-Alexander (SGA) missing.")
            
        if not alexander_great:
            missing_count += 1
            notes.append("Missing 'Alexander the Great' surname reference.")
            
        if not all_rookie:
            # Minor deduction or included in trade detail, strictly rubric asks for it
            missing_count += 1
            notes.append("SGA's All-Rookie status not cited.")

        if missing_count == 0:
            score = 10
        elif missing_count == 1:
            score = 8
        elif missing_count >= 2:
            score = 5
        else:
            score = 0 # Should not reach here if player is correct

        return RubricResult(subtask="subtask1", score=score, notes=notes, checks=checks)

    def _eval_subtask2(self, workspace: Path) -> RubricResult:
        path = self._answer_path(workspace, "subtask2.md")
        text = self._read_text(path)
        if text is None:
            return self._missing("subtask2", f"Missing report: {path}")
        lower = text.lower()
        
        player = "james harden" in lower
        
        # MVP 2017-18
        mvp = "mvp" in lower and bool(re.search(r"2017[\u2013-]?18", text))
        
        # Scoring Titles (3+)
        scoring = ("scoring title" in lower or "scoring champion" in lower or "scoring leader" in lower)
        scoring_count = bool(re.search(r"(three|3|4|four)\s+(?:time|times)?", lower)) # Harden has 3
        
        # Assist Title 2016-17
        assist = ("assist title" in lower or "assist leader" in lower)
        assist_season = bool(re.search(r"2016[\u2013-]?17", text))
        
        # Trade: Ben Simmons, Rookie of the Year
        simmons = "ben simmons" in lower or "simmons" in lower
        roy = "rookie of the year" in lower or "roy" in lower
        
        checks = {
            "player": player,
            "mvp_17_18": mvp,
            "scoring_titles": scoring and scoring_count,
            "assist_title_16_17": assist and assist_season,
            "ben_simmons": simmons,
            "simmons_roy": roy,
        }
        
        notes: List[str] = []
        if not player:
            notes.append("Answer is not centered on James Harden.")
            return RubricResult("subtask2", 0, notes, checks)

        if not (mvp and scoring and assist):
            notes.append("Core MVP/Scoring/Assist milestones not fully demonstrated.")
            return RubricResult("subtask2", 5, notes, checks)

        if not simmons or not roy:
            if not simmons:
                notes.append("Ben Simmons trade component missing.")
            if not roy:
                notes.append("Ben Simmons' Rookie of the Year credential not cited.")
            return RubricResult("subtask2", 8, notes, checks)

        return RubricResult("subtask2", 10, notes, checks)

    def _eval_subtask3(self, workspace: Path) -> RubricResult:
        path = self._answer_path(workspace, "subtask3.md")
        text = self._read_text(path)
        if text is None:
            return self._missing("subtask3", f"Missing report: {path}")
        lower = text.lower()
        
        player = "kevin durant" in lower
        
        # ROY
        roy = "rookie of the year" in lower or "roy" in lower
        
        # 4 Scoring Titles
        scoring = "scoring" in lower and ("4" in lower or "four" in lower)
        
        # MVP (2014) before FMVPs (2017, 2018)
        # We check if these terms/years exist.
        mvp_year = "2014" in text and "mvp" in lower
        fmvp = "finals mvp" in lower
        
        # Sign-and-Trade with D'Angelo Russell (All-Star)
        dlo = "d'angelo" in lower or "russell" in lower
        all_star_dlo = "all-star" in lower or "all star" in lower
        
        checks = {
            "player": player,
            "roy": roy,
            "scoring_titles": scoring,
            "mvp_2014": mvp_year,
            "finals_mvp": fmvp,
            "dlo_russell": dlo,
            "dlo_all_star": all_star_dlo,
        }
        
        notes: List[str] = []
        if not player:
            notes.append("Answer is not about Kevin Durant.")
            return RubricResult("subtask3", 0, notes, checks)

        # Basic credential check
        if not (roy and scoring and mvp_year and fmvp):
            notes.append("Key career milestones (ROY, 4 Scoring Titles, MVP, FMVP) incomplete.")
            return RubricResult("subtask3", 5, notes, checks)

        # Trade specifics
        if not dlo or not all_star_dlo:
            if not dlo:
                notes.append("Exchange involving D'Angelo Russell not mentioned.")
            if not all_star_dlo:
                notes.append("D'Angelo Russell's All-Star status not cited.")
            return RubricResult("subtask3", 8, notes, checks)

        return RubricResult("subtask3", 10, notes, checks)

    def _eval_subtask4(self, workspace: Path) -> RubricResult:
        path = self._answer_path(workspace, "subtask4.md")
        text = self._read_text(path)
        if text is None:
            return self._missing("subtask4", f"Missing report: {path}")
        lower = text.lower()
        
        player = "klay thompson" in lower
        
        # Records: 14 Threes
        record_14 = "14" in text and ("three" in lower or "3-pointer" in lower)
        
        # Scoring < 23 PPG
        scoring_cap = "23" in text or "22" in text # Loose check for the concept
        
        # Multiple Finals
        finals = "finals" in lower
        
        # Teammate: Andrew Wiggins, #1 Pick, All-Conference
        wiggins = "andrew wiggins" in lower or "wiggins" in lower
        pick_one = "#1" in text or "number one" in lower or "1st overall" in lower or "first overall" in lower
        college_honor = "all-conference" in lower or "all-big 12" in lower or "kansas" in lower
        
        checks = {
            "player": player,
            "record_14_threes": record_14,
            "scoring_cap_23": scoring_cap,
            "multiple_finals": finals,
            "wiggins": wiggins,
            "wiggins_pick_1": pick_one,
            "wiggins_college": college_honor,
        }
        
        notes: List[str] = []
        if not player:
            notes.append("Answer is not focused on Klay Thompson.")
            return RubricResult("subtask4", 0, notes, checks)

        if not (record_14 and finals):
             notes.append("14-three record or Finals history missing.")
             return RubricResult("subtask4", 5, notes, checks)
             
        if not wiggins or not pick_one or not college_honor:
             notes.append("Details regarding Andrew Wiggins (#1 Pick, College Honors) are incomplete.")
             return RubricResult("subtask4", 8, notes, checks)
             
        if not scoring_cap:
             notes.append("Confirmation of never averaging > 23 PPG is missing.")
             # Strict rubric requires this
             return RubricResult("subtask4", 8, notes, checks)

        return RubricResult("subtask4", 10, notes, checks)

    def _eval_subtask5(self, workspace: Path) -> RubricResult:
        path = self._answer_path(workspace, "subtask_summary.json")
        if not path.exists():
            return self._missing("subtask5", f"Missing summary JSON: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return self._missing("subtask5", "JSON parse error in summary report.")

        # Normalize data structure
        entries: List[Dict[str, Any]]
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            # Check for keys like 'subtasks' or 'results'
            if "subtasks" in data and isinstance(data["subtasks"], list):
                entries = data["subtasks"]
            else:
                # If dict but not containing list, might be single object (unlikely) or malformed
                entries = [] 
        else:
            entries = []

        expected_players = [
            "Paul George",
            "James Harden",
            "Kevin Durant",
            "Klay Thompson",
        ]
        
        checks = {"structure_ok": bool(entries)}
        
        if not entries or len(entries) < 4:
            return RubricResult("subtask5", 0, ["JSON list missing one or more subtasks."], checks)

        ground_truth_ok = True
        match_fields_ok = True
        conditions_present = True
        descriptive_conditions = True
        
        found_names = []

        for entry in entries:
            # Check submitted player or ground truth field
            p_name = entry.get("submitted_player", "") or entry.get("ground_truth", "")
            found_names.append(p_name)
            
            if "matches_criteria" not in entry and "matches" not in entry:
                match_fields_ok = False
            
            conds = entry.get("verification_notes") or entry.get("conditions")
            if not isinstance(conds, list):
                conditions_present = False
            else:
                if not conds:
                    descriptive_conditions = False
                elif not all(isinstance(item, str) and item.strip() for item in conds):
                    descriptive_conditions = False

        # Verify all expected players are present in the JSON
        # normalizing check
        lower_found = [str(n).lower() for n in found_names]
        for exp in expected_players:
            if exp.lower() not in lower_found and exp.split()[-1].lower() not in lower_found:
                ground_truth_ok = False

        checks.update(
            {
                "ground_truths": ground_truth_ok,
                "matches_field": match_fields_ok,
                "conditions_field": conditions_present,
                "conditions_detail": descriptive_conditions,
            }
        )

        if not ground_truth_ok:
            return RubricResult(
                "subtask5",
                0,
                ["Ground truth list missing or mismatched official names."],
                checks,
            )
        if not match_fields_ok or not conditions_present:
            return RubricResult(
                "subtask5",
                5,
                ["Match flags or condition arrays are incomplete."],
                checks,
            )
        if not descriptive_conditions:
            return RubricResult(
                "subtask5",
                8,
                ["Conditions present but lack descriptive per-clause notes."],
                checks,
            )
        return RubricResult("subtask5", 10, [], checks)


    def _answers_dir(self, workspace: Path) -> Path:
        return workspace.resolve()

    def _answer_path(self, workspace: Path, filename: str) -> Path:
        return self._answers_dir(workspace) / filename

    @staticmethod
    def _read_text(path: Path) -> Optional[str]:
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _missing(subtask: str, message: str) -> RubricResult:
        return RubricResult(subtask=subtask, score=0, notes=[message], checks={})

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
        self.workspace_root = self.run_root / "workspace"
        self.evalspace_root = self.run_root / "evalspace"
        self.start_subtask = max(1, start_subtask)
        self.rubric = RubricEvaluator()
        self.meta: Dict[str, Any] = {
            "model": env_config.model_name,
            "scorer": env_config.scorer_name,
            "max_attempts": env_config.max_attempts,
            "subtasks": [],
        }

    def prepare_layout(self) -> None:
        ensure_directory(self.run_root)
        ensure_directory(self.workspace_root)
        ensure_directory(self.evalspace_root)

    def run(self) -> Dict[str, Any]:
        self.prepare_layout()
        self.env_config.inject_defaults()
        agent = AgentRunner(self.env_config, visualize=self.visualize)
        subtask_count = int(self.description.get("subtask_count") or 0)

        start_index = min(self.start_subtask, subtask_count or 1)
        print(
            f"[BOOT] Model={self.env_config.model_name} scorer={self.env_config.scorer_name} "
            f"start_subtask={start_index}"
        )
        for index in range(start_index, subtask_count + 1):
            subtask = f"subtask{index}"
            prompt = self._build_prompt(subtask)
            attempt_records: List[Dict[str, Any]] = []
            feedback = ""
            print(f"[SUBTASK] Starting {subtask}")
            for attempt in range(1, self.env_config.max_attempts + 1):
                workspace = self._prepare_workspace(subtask)

                prompt_body = prompt
                if feedback:
                    prompt_body = f"{prompt}\n\nPrevious feedback:\n{feedback}"
                agent_output = agent.send(
                    prompt_body,
                    workspace_notice(workspace, self.repo_root),
                    workspace,
                )
                self._sync_to_evalspace()
                rubric_result = self.rubric.evaluate(subtask, self.evalspace_root)
                attempt_records.append(
                    {
                        "attempt_index": attempt,
                        "workspace": str(workspace),
                        "agent_output": agent_output,
                        "score": rubric_result.score,
                        "notes": rubric_result.notes,
                        "checks": rubric_result.checks,
                    }
                )
                print(f"[RESULT] {subtask} attempt {attempt}: score={rubric_result.score}")
                if rubric_result.notes:
                    print("         Notes: " + "; ".join(rubric_result.notes))
                if rubric_result.score == 10:
                    break
                feedback = self._format_feedback(rubric_result)

            best_attempt = max(attempt_records, key=lambda item: item.get("score", 0)) if attempt_records else None
            self.meta["subtasks"].append(
                {
                    "name": subtask,
                    "attempts": attempt_records,
            "best_score": best_attempt.get("score") if best_attempt else 0,
            "best_attempt": best_attempt.get("attempt_index") if best_attempt else None,
        }
            )

        meta_path = self.run_root / "meta_eval.json"
        meta_path.write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        print(f"[DONE] Wrote results to {meta_path}")
        return self.meta

    def _prepare_workspace(self, subtask: str) -> Path:
        workspace = self.workspace_root.resolve()
        ensure_directory(workspace)
        target = self._target_answer_path(subtask)
        if target and target.exists():
            target.unlink()
        return workspace

    def _sync_to_evalspace(self) -> None:
        src = self.workspace_root
        dst = self.evalspace_root
        ensure_directory(self.evalspace_root)
        for item in list(dst.iterdir()):
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        for entry in src.iterdir():
            if entry.is_file():
                shutil.copy2(entry, dst / entry.name)

    def _target_answer_path(self, subtask: str) -> Optional[Path]:
        answers_dir = self.workspace_root
        mapping = {
            "subtask1": "subtask1.md",
            "subtask2": "subtask2.md",
            "subtask3": "subtask3.md",
            "subtask4": "subtask4.md",
            "subtask5": "subtask_summary.json",
        }
        filename = mapping.get(subtask)
        if not filename:
            return None
        return answers_dir / filename

    def _build_prompt(self, subtask: str) -> str:
        entry = self.description.get(subtask, "")
        if isinstance(entry, dict):
            query = str(entry.get("query", "")).strip()
            deliverables = str(entry.get("deliverables", "")).strip()
            if query and deliverables:
                return f"{query}\n\nDeliverables:\n{deliverables}"
            prompt_parts = [part for part in (query, deliverables) if part]
            return "\n\n".join(prompt_parts)
        return str(entry)

    @staticmethod
    def _format_feedback(rubric: RubricResult) -> str:
        if not rubric.notes:
            return ""
        bullets = "\n".join(f"- {note}" for note in rubric.notes)
        return f"Please address these rubric gaps:\n{bullets}"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task evaluator runner without scoring")
    parser.add_argument(
        "--env",
        default=".env",
        help="Path to env file for agent credentials",
    )
    parser.add_argument("--visualize", action="store_true", help="Enable SII SDK visualization")
    parser.add_argument(
        "--description",
        default="description.json",
        help="Path to task description JSON",
    )
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
