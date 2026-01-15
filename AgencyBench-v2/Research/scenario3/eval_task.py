#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import textwrap
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def try_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


METRIC_HINTS: Dict[str, str] = {
    "subtask1_answer_exists": "subtask1需要提供subtask1_answer.txt。",
    "subtask1_answer_company": "subtask1_answer.txt必须明确UnitedHealth Group。",
    "subtask1_findings_exists": "subtask1_findings.md缺失或无法读取。",
    "subtask1_mentions_optum": "subtask1_findings.md需提及Optum的服务收入。",
    "subtask1_mentions_disney": "subtask1_findings.md需与迪士尼2023年收入对比。",
    "subtask1_mentions_witty": "需概述Sir Andrew Witty的背景与头衔。",
    "subtask1_mentions_year": "需标注2023财年上下文或具体年份。",
    "subtask2_analysis_exists": "subtask2_analysis.md缺失。",
    "subtask2_mentions_broadcom": "分析中需明确Broadcom Inc.身份。",
    "subtask2_compares_salesforce": "需提供Broadcom与Salesforce营收对比。",
    "subtask2_mentions_locations": "需说明圣何塞总部与新加坡运营枢纽。",
    "subtask2_mentions_ceo": "需描述Hock Tan的职业经历。",
    "subtask2_mentions_revenue": "需包含FY2023营收数据。",
    "subtask2_evidence_exists": "subtask2_evidence.csv缺失。",
    "subtask2_evidence_broadcom": "证据表需包含Broadcom指标。",
    "subtask2_evidence_salesforce": "证据表需包含Salesforce对比。",
    "subtask3_memo_exists": "subtask3_memo.md缺失。",
    "subtask3_mentions_citigroup": "需明确花旗集团身份。",
    "subtask3_compares_goldman": "需与Goldman Sachs营收对比。",
    "subtask3_mentions_ceo": "需描述Jane Fraser的任命及履历。",
    "subtask3_mentions_education": "需提及哈佛商学院背景。",
    "subtask3_mentions_revenue": "需包含2023营收数据。",
    "subtask3_sources_exists": "subtask3_sources.txt缺失。",
    "subtask3_sources_urls": "来源列表需包含至少一个URL。",
    "subtask4_summary_exists": "subtask4_summary.md缺失。",
    "subtask4_mentions_xpeng": "摘要需明确XPeng Inc。",
    "subtask4_mentions_competitors": "需提到NIO与Li Auto交付比较。",
    "subtask4_mentions_founder": "需描述何小鹏的身份。",
    "subtask4_mentions_award": "需提及2014年Top 50 Entrepreneur荣誉。",
    "subtask4_mentions_startup": "需说明UCWeb联合创办经历。",
    "subtask4_metrics_exists": "subtask4_metrics.json缺失。",
    "subtask4_metrics_keys": "subtask4_metrics.json需包含指定字段。",
    "subtask4_metrics_company": "metrics.json中的company需为XPeng。",
    "subtask5_summary_exists": "subtask5_summary.csv缺失。",
    "subtask5_summary_columns": "CSV需包含要求的列名。",
    "subtask5_summary_companies": "CSV需列出四家公司。",
    "subtask5_summary_sources": "CSV的primary_source字段需填写。",
    "subtask5_overview_exists": "subtask5_overview.md缺失。",
    "subtask5_overview_mentions_companies": "概述需逐一确认四家公司。",
}


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
    workspace: Path
    evalspace: Path
    agent_output: str
    rubric: RubricResult
    feedback: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subtask": self.subtask,
            "attempt_index": self.attempt_index,
            "workspace": str(self.workspace),
            "evalspace": str(self.evalspace),
            "agent_output": self.agent_output,
            "rubric": self.rubric.to_dict(),
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
# Rubric evaluator aligned with description.json
# --------------------------------------------------------------------------- #


# class RubricEvaluator:
#     def evaluate(self, subtask: str, evalspace: Path) -> RubricResult:
#         handler = getattr(self, f"_eval_{subtask}", None)
#         if not handler:
#             return RubricResult(subtask=subtask, score=0.0, metrics={}, notes=[f"未知子任务: {subtask}"])
#         return handler(evalspace)

#     def _eval_subtask1(self, evalspace: Path) -> RubricResult:
#         findings = self._read_text(evalspace / "subtask1_findings.md")
#         answer = self._read_text(evalspace / "subtask1_answer.txt")
#         metrics = {
#             "subtask1_answer_exists": 1.0 if answer else 0.0,
#             "subtask1_answer_company": 1.0 if answer and "unitedhealth" in answer.lower() else 0.0,
#             "subtask1_findings_exists": 1.0 if findings else 0.0,
#             "subtask1_mentions_optum": 1.0 if findings and "optum" in findings.lower() else 0.0,
#             "subtask1_mentions_disney": 1.0 if findings and "disney" in findings.lower() else 0.0,
#             "subtask1_mentions_witty": 1.0 if findings and "witty" in findings.lower() else 0.0,
#             "subtask1_mentions_year": 1.0 if findings and "2023" in findings else 0.0,
#         }
#         return self._result_from_metrics("subtask1", metrics)

#     def _eval_subtask2(self, evalspace: Path) -> RubricResult:
#         analysis = self._read_text(evalspace / "subtask2_analysis.md")
#         evidence = self._read_text(evalspace / "subtask2_evidence.csv")
#         metrics = {
#             "subtask2_analysis_exists": 1.0 if analysis else 0.0,
#             "subtask2_mentions_broadcom": 1.0 if analysis and "broadcom" in analysis.lower() else 0.0,
#             "subtask2_compares_salesforce": 1.0 if analysis and "salesforce" in analysis.lower() else 0.0,
#             "subtask2_mentions_locations": 1.0
#             if analysis and ("san jose" in analysis.lower() or "california" in analysis.lower())
#             and "singapore" in analysis.lower()
#             else 0.0,
#             "subtask2_mentions_ceo": 1.0 if analysis and "hock tan" in analysis.lower() else 0.0,
#             "subtask2_mentions_revenue": 1.0 if analysis and ("2023" in analysis and "revenue" in analysis.lower()) else 0.0,
#             "subtask2_evidence_exists": 1.0 if evidence else 0.0,
#             "subtask2_evidence_broadcom": 1.0 if evidence and "broadcom" in evidence.lower() else 0.0,
#             "subtask2_evidence_salesforce": 1.0 if evidence and "salesforce" in evidence.lower() else 0.0,
#         }
#         return self._result_from_metrics("subtask2", metrics)

#     def _eval_subtask3(self, evalspace: Path) -> RubricResult:
#         memo = self._read_text(evalspace / "subtask3_memo.md")
#         sources = self._read_text(evalspace / "subtask3_sources.txt")
#         metrics = {
#             "subtask3_memo_exists": 1.0 if memo else 0.0,
#             "subtask3_mentions_citigroup": 1.0 if memo and "citigroup" in memo.lower() else 0.0,
#             "subtask3_compares_goldman": 1.0 if memo and "goldman" in memo.lower() else 0.0,
#             "subtask3_mentions_ceo": 1.0 if memo and "jane fraser" in memo.lower() else 0.0,
#             "subtask3_mentions_education": 1.0 if memo and "harvard" in memo.lower() else 0.0,
#             "subtask3_mentions_revenue": 1.0 if memo and ("2023" in memo and "revenue" in memo.lower()) else 0.0,
#             "subtask3_sources_exists": 1.0 if sources else 0.0,
#             "subtask3_sources_urls": 1.0 if sources and "http" in sources.lower() else 0.0,
#         }
#         return self._result_from_metrics("subtask3", metrics)

#     def _eval_subtask4(self, evalspace: Path) -> RubricResult:
#         summary = self._read_text(evalspace / "subtask4_summary.md")
#         metrics_path = evalspace / "subtask4_metrics.json"
#         metrics_json = self._load_json(metrics_path)
#         required_keys = {"company", "h1_2025_deliveries", "nio_li_auto_sum", "founder_award", "prior_startup"}
#         metrics = {
#             "subtask4_summary_exists": 1.0 if summary else 0.0,
#             "subtask4_mentions_xpeng": 1.0 if summary and "xpeng" in summary.lower() else 0.0,
#             "subtask4_mentions_competitors": 1.0
#             if summary and "nio" in summary.lower() and "li auto" in summary.lower()
#             else 0.0,
#             "subtask4_mentions_founder": 1.0 if summary and "he xiaopeng" in summary.lower() else 0.0,
#             "subtask4_mentions_award": 1.0 if summary and ("2014" in summary and "top 50" in summary.lower()) else 0.0,
#             "subtask4_mentions_startup": 1.0 if summary and "ucweb" in summary.lower() else 0.0,
#             "subtask4_metrics_exists": 1.0 if metrics_json else 0.0,
#             "subtask4_metrics_keys": 1.0 if metrics_json and required_keys.issubset(set(metrics_json.keys())) else 0.0,
#             "subtask4_metrics_company": 1.0
#             if metrics_json and isinstance(metrics_json.get("company"), str) and "xpeng" in metrics_json["company"].lower()
#             else 0.0,
#         }
#         return self._result_from_metrics("subtask4", metrics)

#     def _eval_subtask5(self, evalspace: Path) -> RubricResult:
#         summary_path = evalspace / "subtask5_summary.csv"
#         overview = self._read_text(evalspace / "subtask5_overview.md")
#         columns, rows = self._read_csv(summary_path)
#         required_columns = ["company", "core_business", "2023_or_latest_revenue", "key_executive", "appointment_year", "notable_fact", "primary_source"]
#         companies_required = {"unitedhealth", "broadcom", "citigroup", "xpeng"}
#         row_companies = {row.get("company", "").strip().lower() for row in rows if row.get("company")}
#         primary_sources_filled = all(row.get("primary_source", "").strip() for row in rows) if rows else False
#         overview_ok = overview and all(name in overview.lower() for name in companies_required)
#         metrics = {
#             "subtask5_summary_exists": 1.0 if columns else 0.0,
#             "subtask5_summary_columns": 1.0 if columns and all(col in columns for col in required_columns) else 0.0,
#             "subtask5_summary_companies": 1.0 if companies_required.issubset(row_companies) else 0.0,
#             "subtask5_summary_sources": 1.0 if primary_sources_filled else 0.0,
#             "subtask5_overview_exists": 1.0 if overview else 0.0,
#             "subtask5_overview_mentions_companies": 1.0 if overview_ok else 0.0,
#         }
#         return self._result_from_metrics("subtask5", metrics)


class RubricEvaluator:
    def evaluate(self, subtask: str, evalspace: Path) -> RubricResult:
        handler = getattr(self, f"_eval_{subtask}", None)
        if not handler:
            return RubricResult(subtask=subtask, score=0.0, metrics={}, notes=[f"未知子任务: {subtask}"])
        return handler(evalspace)

    def _eval_subtask1(self, evalspace: Path) -> RubricResult:
        findings = self._read_text(evalspace / "subtask1_findings.md")
        answer = self._read_text(evalspace / "subtask1_answer.txt")
        metrics = {
            "subtask1_answer_exists": 1.0 if answer else 0.0,
            "subtask1_answer_company": 1.0 if answer and "unitedhealth" in answer.lower() else 0.0,
            "subtask1_findings_exists": 1.0 if findings else 0.0,
            "subtask1_mentions_optum": 1.0 if findings and "optum" in findings.lower() else 0.0,
            "subtask1_mentions_disney": 1.0 if findings and "disney" in findings.lower() else 0.0,
            "subtask1_mentions_witty": 1.0 if findings and "witty" in findings.lower() else 0.0,
            "subtask1_mentions_gsk": 1.0 if findings and ("glaxosmithkline" in findings.lower() or "gsk" in findings.lower()) else 0.0,
        }
        return self._result_from_metrics("subtask1", metrics)

    def _eval_subtask2(self, evalspace: Path) -> RubricResult:
        analysis = self._read_text(evalspace / "subtask2_analysis.md")
        evidence = self._read_text(evalspace / "subtask2_evidence.csv")
        metrics = {
            "subtask2_analysis_exists": 1.0 if analysis else 0.0,
            "subtask2_mentions_broadcom": 1.0 if analysis and "broadcom" in analysis.lower() else 0.0,
            "subtask2_compares_salesforce": 1.0 if analysis and "salesforce" in analysis.lower() else 0.0,
            "subtask2_mentions_locations": 1.0
            if analysis and ("san jose" in analysis.lower() or "california" in analysis.lower())
            and "singapore" in analysis.lower()
            else 0.0,
            "subtask2_mentions_ceo": 1.0 if analysis and "hock tan" in analysis.lower() else 0.0,
            "subtask2_evidence_exists": 1.0 if evidence else 0.0,
            "subtask2_evidence_broadcom": 1.0 if evidence and "broadcom" in evidence.lower() else 0.0,
        }
        return self._result_from_metrics("subtask2", metrics)

    def _eval_subtask3(self, evalspace: Path) -> RubricResult:
        memo = self._read_text(evalspace / "subtask3_memo.md")
        sources = self._read_text(evalspace / "subtask3_sources.txt")
        metrics = {
            "subtask3_memo_exists": 1.0 if memo else 0.0,
            "subtask3_mentions_citigroup": 1.0 if memo and "citigroup" in memo.lower() else 0.0,
            "subtask3_compares_goldman": 1.0 if memo and "goldman" in memo.lower() else 0.0,
            "subtask3_mentions_ceo": 1.0 if memo and "jane fraser" in memo.lower() else 0.0,
            "subtask3_mentions_education": 1.0 if memo and "harvard" in memo.lower() else 0.0,
            "subtask3_sources_exists": 1.0 if sources else 0.0,
            "subtask3_sources_urls": 1.0 if sources and "http" in sources.lower() else 0.0,
        }
        return self._result_from_metrics("subtask3", metrics)

    def _eval_subtask4(self, evalspace: Path) -> RubricResult:
        summary = self._read_text(evalspace / "subtask4_summary.md")
        metrics_path = evalspace / "subtask4_metrics.json"
        metrics_json = self._load_json(metrics_path)
        
        # 更新为新 Benchmark 要求的字段
        required_keys = {"company", "strategic_partner", "investment_amount", "founder_prior_startup"}
        
        metrics = {
            "subtask4_summary_exists": 1.0 if summary else 0.0,
            "subtask4_mentions_xpeng": 1.0 if summary and "xpeng" in summary.lower() else 0.0,
            # 检查大众汽车 (Volkswagen) 合作伙伴
            "subtask4_mentions_partner": 1.0
            if summary and ("volkswagen" in summary.lower() or "vw" in summary.lower())
            else 0.0,
            # 检查创始人 UCWeb 背景
            "subtask4_mentions_startup": 1.0 if summary and "ucweb" in summary.lower() else 0.0,
            # 检查 700 million 投资金额
            "subtask4_mentions_investment": 1.0 if summary and "700" in summary else 0.0,
            
            "subtask4_metrics_exists": 1.0 if metrics_json else 0.0,
            "subtask4_metrics_keys": 1.0 if metrics_json and required_keys.issubset(set(metrics_json.keys())) else 0.0,
            "subtask4_metrics_company": 1.0
            if metrics_json and isinstance(metrics_json.get("company"), str) and "xpeng" in metrics_json["company"].lower()
            else 0.0,
        }
        return self._result_from_metrics("subtask4", metrics)

    def _eval_subtask5(self, evalspace: Path) -> RubricResult:
        summary_path = evalspace / "subtask5_summary.csv"
        overview = self._read_text(evalspace / "subtask5_overview.md")
        columns, rows = self._read_csv(summary_path)
        
        # 更新为新 Benchmark 要求的表头列名
        required_columns = ["company", "core_business", "2023_revenue_approx", "key_executive", "key_milestone_or_fact", "primary_source"]
        
        companies_required = {"unitedhealth", "broadcom", "citigroup", "xpeng"}
        row_companies = {row.get("company", "").strip().lower() for row in rows if row.get("company")}
        primary_sources_filled = all(row.get("primary_source", "").strip() for row in rows) if rows else False
        overview_ok = overview and all(name in overview.lower() for name in companies_required)
        
        metrics = {
            "subtask5_summary_exists": 1.0 if columns else 0.0,
            "subtask5_summary_columns": 1.0 if columns and all(col in columns for col in required_columns) else 0.0,
            "subtask5_summary_companies": 1.0 if companies_required.issubset(row_companies) else 0.0,
            "subtask5_summary_sources": 1.0 if primary_sources_filled else 0.0,
            "subtask5_overview_exists": 1.0 if overview else 0.0,
            "subtask5_overview_mentions_companies": 1.0 if overview_ok else 0.0,
        }
        return self._result_from_metrics("subtask5", metrics)

    # Helper utilities ----------------------------------------------------- #

    def _result_from_metrics(self, subtask: str, metrics: Dict[str, float]) -> RubricResult:
        if not metrics:
            return RubricResult(subtask=subtask, score=0.0, metrics={}, notes=["未检测到任何指标"])
        total = len(metrics)
        score = round(10 * (sum(clamp(value, 0.0, 1.0) for value in metrics.values()) / total), 2)
        notes = [
            METRIC_HINTS.get(key, f"{key}未满足要求。")
            for key, value in metrics.items()
            if clamp(value, 0.0, 1.0) < 0.99
        ]
        return RubricResult(subtask=subtask, score=score, metrics=metrics, notes=notes)

    def _read_text(self, path: Path) -> Optional[str]:
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return None

    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    def _read_csv(self, path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
        if not path.exists():
            return [], []
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = [row for row in reader]
                columns = reader.fieldnames or []
                return columns, rows
        except Exception:
            return [], []


# --------------------------------------------------------------------------- #
# Evaluation coordinator
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
        ensure_directory(self.run_root / "workspace")
        ensure_directory(self.run_root / "evalspace")

    def run(self) -> Dict[str, Any]:
        self.prepare_layout()
        self.env_config.inject_defaults()
        agent = AgentRunner(self.env_config, visualize=self.visualize)
        workspace = (self.run_root / "workspace").resolve()
        evalspace = (self.run_root / "evalspace").resolve()
        clear_directory(workspace)
        clear_directory(evalspace)
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
            feedback: str = ""
            print(f"[SUBTASK] Starting {subtask}")
            for attempt in range(1, self.env_config.max_attempts + 1):
                workspace, evalspace = self._prepare_attempt_dirs(subtask, attempt)

                prompt_with_feedback = f"{prompt}\n\n{feedback}" if feedback else prompt
                agent_output = agent.send(
                    prompt_with_feedback,
                    workspace_notice(workspace, self.repo_root),
                    workspace,
                )
                print(f"[COPY] Starting copy workspace to evalspace")
                clear_directory(evalspace)
                copy_workspace(workspace, evalspace)
                rubric_result = self.rubric.evaluate(subtask, evalspace)
                feedback = self._build_feedback(rubric_result)
                summary = AttemptSummary(
                    subtask=subtask,
                    attempt_index=attempt,
                    workspace=workspace,
                    evalspace=evalspace,
                    agent_output=agent_output,
                    rubric=rubric_result,
                    feedback=feedback,
                )
                attempt_summaries.append(summary)
                print(f"[RESULT] {subtask} attempt {attempt}: score={rubric_result.score}")
                if rubric_result.notes:
                    print("         Notes: " + "; ".join(rubric_result.notes))
                if rubric_result.score >= 9.99:
                    break

            best = max(attempt_summaries, key=lambda item: item.rubric.score) if attempt_summaries else None
            self.meta["subtasks"].append(
                {
                    "name": subtask,
                    "attempts": [item.to_dict() for item in attempt_summaries],
                    "best_score": best.rubric.score if best else 0.0,
                    "best_attempt": best.attempt_index if best else None,
                }
            )

        meta_path = self.run_root / "meta_eval.json"
        meta_path.write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        print(f"[DONE] Wrote results to {meta_path}")
        return self.meta

    def _prepare_attempt_dirs(
        self, subtask: str, attempt_index: int
    ) -> Tuple[Path, Path]:
        workspace = (self.run_root / "workspace").resolve()
        evalspace = (self.run_root / "evalspace").resolve()
        ensure_directory(workspace)
        ensure_directory(evalspace)
        return workspace, evalspace

    def _build_feedback(self, rubric: RubricResult) -> str:
        if not rubric.notes:
            return ""
        bullets = "\n".join(f"- {note}" for note in rubric.notes)
        return f"请重点完善以下Rubric要点：\n{bullets}"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluator runner")
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
