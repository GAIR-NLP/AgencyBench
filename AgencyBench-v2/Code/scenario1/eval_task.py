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
        # It's okay if env file doesn't exist, we might be relying on existing env vars
        return {}

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


EXCLUDED_COPY_DIRS = {".git", ".hg", ".svn", "__pycache__", ".sii", "logs", "evalspace"}
EXCLUDED_COPY_FILES = {"meta_eval.json"}
EXCLUDED_COPY_SUFFIXES = {".db", ".sqlite", ".pyc"}


def copy_workspace(src: Path, dst: Path) -> None:
    src = src.resolve()
    dst = dst.resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source workspace missing: {src}")
    ensure_directory(dst)
    clear_directory(dst)
    for root, dirs, files in os.walk(src):
        rel_root = Path(root).relative_to(src)
        # Filter dirs
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in EXCLUDED_COPY_DIRS]
        
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
        You must work inside {target}. Do not access other paths. Keep the current structure.
        Only modify `equation.py` as requested.
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

        # Allow running without explicit env file if vars are in OS env, 
        # but require basic Agent config if not eval_only
        sii_target = fetch("SII_TARGET_MODEL") or "local_model"
        sii_api_key = fetch("SII_AGENT_API_KEY") or "dummy"
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
    passed: bool
    loss: Optional[float]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subtask": self.subtask,
            "score": self.score,
            "passed": self.passed,
            "loss": self.loss,
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
# Command runner (Python script execution)
# --------------------------------------------------------------------------- #


class CommandRunner:
    """Execute Python evaluation scripts."""

    def __init__(self, logs_dir: Path, conda_env: Optional[str] = None, conda_exe: Optional[str] = None):
        self.logs_dir = ensure_directory(logs_dir)
        self.conda_env = conda_env
        self.conda_exe = conda_exe
        self._warned_conda_missing = False

    def capture(self, subtask: str, root: Path) -> Dict[str, Any]:
        """
        Runs the evaluation script 'evaluate_equation.py' located in the root.
        """
        root = root.resolve()
        results: Dict[str, CommandResult] = {}
        
        # We assume the main entry point is evaluate_equation.py per instructions
        script_name = "evaluate_equation.py"
        script_path = root / script_name
        
        if not script_path.exists():
             results["evaluate"] = CommandResult(
                name="evaluate",
                command=[script_name],
                returncode=1,
                stdout="",
                stderr=f"Error: {script_name} not found in {root}",
            )
        else:
            cmd = self._wrap_cmd(["python3", script_name])
            rc, out, err = run_command(cmd, cwd=root, timeout=300)
            results["evaluate"] = CommandResult(name="evaluate", command=cmd, returncode=rc, stdout=out, stderr=err)

        self._persist(results, subtask=subtask)
        summary = {name: res.to_dict() for name, res in results.items()}
        return summary

    def _persist(self, results: Dict[str, CommandResult], subtask: str = "") -> None:
        for name, result in results.items():
            targets = [self.logs_dir / f"{name}.txt"]
            # Also emit subtask-specific log to match description expectations
            if subtask and name == "evaluate":
                targets.append(self.logs_dir / f"{subtask}_evaluate.txt")
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
            for log_path in targets:
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
# Rubric evaluation (MSE based)
# --------------------------------------------------------------------------- #


class RubricEvaluator:
    # Thresholds: (zero_score_thresh, full_score_thresh)
    THRESHOLDS = {
        "subtask1": (5e-1, 1e-1),
        "subtask2": (2e-1, 5e-2),
        "subtask3": (5e-2, 1e-2),
        "subtask4": (2e-2, 5e-3),
        "subtask5": (1e-2, 1e-3),
    }

    def evaluate(
        self, subtask: str, evalspace: Path, command_results: Dict[str, Any]
    ) -> RubricResult:
        
        # 1. Check for required documentation
        doc_path = evalspace / "docs" / f"{subtask}_rationale.md"
        eq_file = evalspace / "equation.py"
        
        notes: List[str] = []
        penalty_notes: List[str] = []
        structural_notes: List[str] = []
        penalty_count = 0

        if not doc_path.exists():
            penalty_count += 1
            penalty_notes.append(f"Missing documentation: {doc_path.name} (-5)")
        if not eq_file.exists():
            penalty_count += 1
            penalty_notes.append("Missing equation.py (-5)")
        
        # 1b. Check for required log naming (allow fallback to evaluate.txt)
        if not self._has_expected_log(subtask, evalspace):
            penalty_count += 1
            penalty_notes.append(f"Missing evaluation log: logs/{subtask}_evaluate.txt (or equivalent) (-5)")
        
        eq_source = ""
        if eq_file.exists():
            try:
                eq_source = eq_file.read_text(encoding="utf-8")
            except Exception as exc:  # pragma: no cover - safety
                penalty_count += 1
                penalty_notes.append(f"Could not read equation.py: {exc} (-5)")
        if eq_source:
            structural_notes = self._apply_description_checks(subtask, eq_source)
        
        # 2. Extract Loss from evaluation output
        evaluate_res = command_results.get("evaluate", {})
        stdout = evaluate_res.get("stdout", "")
        stderr = evaluate_res.get("stderr", "")
        
        loss = self._parse_loss(stdout)
        if loss is None:
            # Try stderr just in case
            loss = self._parse_loss(stderr)
        
        if loss is None:
            notes = penalty_notes + structural_notes + ["Could not parse MSE/Loss from output"]
            return RubricResult(subtask=subtask, score=0, passed=False, loss=None, notes=notes)

        # 3. Calculate Score
        thresholds = self.THRESHOLDS.get(subtask, (1.0, 0.1)) # Default fallback
        zero_thresh, full_thresh = thresholds
        
        if loss >= zero_thresh:
            base_score = 0.0
        elif loss <= full_thresh:
            base_score = 10.0
        else:
            # Linear interpolation: Score = 10 * (zero - loss) / (zero - full)
            base_score = 10.0 * (zero_thresh - loss) / (zero_thresh - full_thresh)

        score = base_score
        if penalty_count:
            score = max(0.0, score - 5.0 * penalty_count)
            penalty_notes.append(f"Score reduced by {5 * penalty_count} due to missing deliverables")
        
        # Structural requirements remain blocking
        if structural_notes:
            score = 0.0
        
        score = round(score, 2)
        notes = penalty_notes + structural_notes
        passed = (loss <= full_thresh) and not structural_notes and not penalty_count

        return RubricResult(
            subtask=subtask, 
            score=score, 
            passed=passed, 
            loss=loss, 
            notes=notes
        )

    def _has_expected_log(self, subtask: str, evalspace: Path) -> bool:
        logs_dir = evalspace / "logs"
        expected = logs_dir / f"{subtask}_evaluate.txt"
        fallback = logs_dir / "evaluate.txt"
        return expected.exists() or fallback.exists()

    def _apply_description_checks(self, subtask: str, source: str) -> List[str]:
        """Apply heuristic checks that mirror description.json structural requirements."""
        cleaned = self._strip_docstrings(source)
        comment_count = self._comment_count(cleaned)
        issues: List[str] = []

        if subtask == "subtask1":
            if not self._is_first_order_linear(cleaned):
                issues.append("Subtask1: equation must remain first-order linear in t and A")
        elif subtask == "subtask2":
            if not self._has_nonlinear_interaction(cleaned):
                issues.append("Subtask2: missing explicit nonlinear t–A interaction term")
            if comment_count < 1:
                issues.append("Subtask2: add inline comments explaining nonlinear terms")
        elif subtask == "subtask3":
            if not self._has_coupled_term(cleaned):
                issues.append("Subtask3: missing coupled t–A term blending dynamics")
            if comment_count < 1:
                issues.append("Subtask3: add comments clarifying stiff vs slow pathways")
        elif subtask == "subtask4":
            if not self._has_coupled_term(cleaned):
                issues.append("Subtask4: missing coupled t–A term blending dynamics")
            if not self._has_stabilizing_term(cleaned):
                issues.append("Subtask4: missing damping/saturation safeguards (clip/epsilon/soft limiter)")
            if comment_count < 1:
                issues.append("Subtask4: add comments referencing how each term mitigates stiffness")
        elif subtask == "subtask5":
            pathways_ok, multi_path, has_comments = self._has_high_fidelity_pathways(cleaned)
            if not multi_path:
                issues.append("Subtask5: need multiple higher-order or competing pathways")
            if not pathways_ok:
                issues.append("Subtask5: ensure coupled nonlinear terms remain interpretable and stable")
            if not has_comments:
                issues.append("Subtask5: add pathway comments for interpretability")

        return issues

    def _strip_docstrings(self, source: str) -> str:
        return re.sub(r'"""[\\s\\S]*?"""|\'\'\'[\\s\\S]*?\'\'\'', "", source)

    def _comment_count(self, source: str) -> int:
        return sum(1 for line in source.splitlines() if line.strip().startswith("#"))

    def _is_first_order_linear(self, source: str) -> bool:
        lowered = source.lower()
        nonlinear_markers = [
            "**2",
            "**3",
            "np.exp",
            "np.log",
            "np.tanh",
            "np.sin",
            "np.cos",
            "np.power",
            "np.sqrt",
            "np.abs",
            "np.clip",
        ]
        cross_markers = [
            "t*A",
            "A*t",
            "t * A",
            "A * t",
            "A*A",
            "t*t",
            "A**",
            "t**",
        ]
        if any(marker in lowered for marker in nonlinear_markers):
            return False
        if any(marker in source for marker in cross_markers):
            return False
        return "t" in source and "A" in source

    def _has_nonlinear_interaction(self, source: str) -> bool:
        nonlinear_markers = [
            "**2",
            "**3",
            "np.exp",
            "np.log",
            "np.tanh",
            "np.power",
            "np.sqrt",
            "/ (1",
            "/(1",
            "t*A",
            "A*t",
            "t * A",
            "A * t",
        ]
        has_marker = any(marker in source for marker in nonlinear_markers)
        if not has_marker:
            return False
        return self._has_coupled_term(source) or "A/(1" in source or "A / (1" in source

    def _has_coupled_term(self, source: str) -> bool:
        pattern = re.compile(r"t[^\\n]{0,80}A|A[^\\n]{0,80}t")
        return bool(pattern.search(source))

    def _has_stabilizing_term(self, source: str) -> bool:
        stability_markers = [
            "np.clip",
            "np.maximum",
            "np.minimum",
            "1e-",
            "eps",
            "epsilon",
            "np.tanh",
            "/ (1",
            "/(1",
        ]
        if any(marker in source for marker in stability_markers):
            return True
        if re.search(r"/\s*\(?[^\n]*\+\s*1e-", source):
            return True
        return False

    def _has_high_fidelity_pathways(self, source: str) -> Tuple[bool, bool, bool]:
        comment_count = self._comment_count(source)
        has_comments = comment_count >= 2
        assignment_lines = [
            line for line in source.splitlines() if "=" in line and not line.strip().startswith("#")
        ]
        multi_path = len(assignment_lines) >= 2
        coupled_nonlin = self._has_coupled_term(source) and self._has_nonlinear_interaction(source)
        return coupled_nonlin and multi_path and has_comments, multi_path, has_comments

    def _parse_loss(self, text: str) -> Optional[float]:
        # Look for patterns like "MSE: 0.0123" or "Loss: 1.2e-4"
        # Case insensitive regex
        patterns = [
            r"(?:MSE|Loss|Error)\s*[:=]\s*([\d\.eE+-]+)",
            r"Mean Squared Error\s*[:=]\s*([\d\.eE+-]+)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    # Return the last reported loss (usually final)
                    return float(matches[-1])
                except ValueError:
                    continue
        # Fallback: bare number on its own line (handles legacy output)
        loose_numbers = re.findall(r"[-+]?\d+\.\d+(?:e[+-]?\d+)?", text, re.IGNORECASE)
        if loose_numbers:
            try:
                return float(loose_numbers[-1])
            except ValueError:
                return None
        return None


# --------------------------------------------------------------------------- #
# Coordinator
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
        
        previous_best_workspace: Optional[Path] = None
        
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
            
            # For the current subtask, we want to find the best workspace to pass to the next subtask
            current_subtask_best_workspace: Optional[Path] = None
            
            feedback: str = ""
            print(f"[SUBTASK] Starting {subtask}")
            attempt_limit = 1 if self.eval_only else self.env_config.max_attempts
            
            for attempt in range(1, attempt_limit + 1):
                workspace, evalspace = self._prepare_attempt_dirs(subtask, attempt)
                
                # --- WORKSPACE INITIALIZATION LOGIC ---
                if attempt == 1:
                    # Initialization
                    if index == 1:
                        # Subtask 1: Copy from 'source' folder in script directory
                        source_dir = SCRIPT_DIR / "source"
                        if source_dir.exists():
                            print(f"[INIT] Copying initial source from {source_dir} to {workspace}")
                            copy_workspace(source_dir, workspace)
                        else:
                            print(f"[WARN] No 'source' directory found at {source_dir}. Starting empty.")
                    else:
                        # Subsequent subtasks: Copy from previous best
                        if previous_best_workspace and previous_best_workspace.exists():
                             print(f"[INIT] Copying from previous best: {previous_best_workspace}")
                             copy_workspace(previous_best_workspace, workspace)
                else:
                    # Subsequent attempts: Clone from previous attempt of *this* subtask
                    self._clone_previous_attempt(subtask, attempt, workspace)
                # -------------------------------------

                if not self.eval_only and agent:
                    agent_output = agent.send(
                        prompt + ("\n\n" + feedback if feedback else ""),
                        workspace_notice(workspace, self.repo_root),
                        workspace,
                    )
                else:
                    agent_output = ""
                
                # Copy workspace to evalspace for isolation
                copy_workspace(workspace, evalspace) # Description says "copy fully"
                
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
                    f"[RESULT] {subtask} attempt {attempt}: score={rubric_result.score:.2f} "
                    f"loss={rubric_result.loss}"
                )
                
                # If perfect score, we can stop early for this subtask
                if not self.eval_only and rubric_result.score >= 9.9:
                    current_subtask_best_workspace = workspace
                    break
                
                current_subtask_best_workspace = workspace # keep the latest if not perfect, or logic below

            # Determine best attempt for this subtask
            best = max(attempt_summaries, key=lambda item: item.score) if attempt_summaries else None
            
            if best:
                current_subtask_best_workspace = best.workspace
                previous_best_workspace = current_subtask_best_workspace
            
            self.meta["subtasks"].append(
                {
                    "name": subtask,
                    "attempts": [item.to_dict() for item in attempt_summaries],
                    "best_score": best.score if best else 0,
                    "best_attempt": best.attempt_index if best else None,
                }
            )

            # Check termination condition (if score is too low, maybe stop? For now we just warn)
            if best and best.score <= 0.1 and not self.eval_only:
                 print(f"[WARN] Low score on {subtask}. Continuing anyway.")

        meta_path = self.run_root / "meta_eval.json"
        meta_path.write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        print(f"[DONE] Wrote results to {meta_path}")
        return self.meta

    def _prepare_attempt_dirs(
        self, subtask: str, attempt_index: int
    ) -> Tuple[Path, Path]:
        attempt_name = f"attempt_{attempt_index:02d}"
        workspace = (self.run_root / subtask / "workspace" / attempt_name).resolve()
        evalspace = (self.run_root / subtask / "evalspace" / attempt_name).resolve()
        
        if not self.eval_only:
             # Ensure clean slate for workspace if re-running
             if workspace.exists():
                 clear_directory(workspace)
        ensure_directory(workspace)
        
        if evalspace.exists():
            clear_directory(evalspace)
        ensure_directory(evalspace)
        
        return workspace, evalspace

    def _clone_previous_attempt(self, subtask: str, attempt_index: int, workspace: Path) -> None:
        prev_attempt = attempt_index - 1
        prev_path = (self.run_root / subtask / "workspace" / f"attempt_{prev_attempt:02d}").resolve()
        if prev_path.exists():
            print(f"[INIT] Cloning attempt {prev_attempt} to {attempt_index}")
            copy_workspace(prev_path, workspace)

    def _build_feedback(self, rubric: RubricResult) -> str:
        msg_parts = []
        if rubric.loss is not None:
             msg_parts.append(f"Current MSE: {rubric.loss}")
        else:
             msg_parts.append("Could not determine MSE from output.")
        
        if rubric.notes:
            msg_parts.append("Issues found: " + "; ".join(rubric.notes))
            
        return "\n".join(msg_parts)

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
        ]
        for args, label in checks:
            cmd = [conda_exe, "run", "-n", env_name, *args]
            rc, out, err = run_command(cmd, timeout=450)
            print(f"[DEBUG] Conda check ({label}) rc={rc}")
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
        for name, data in commands.items():
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

    @staticmethod
    def _truncate_output(text: str, limit: int = 200) -> str:
        if not text:
            return ""
        cleaned = " ".join(text.split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 3] + "..."

    def _print_rubric_diagnostics(self, subtask: str, attempt: int, rubric: RubricResult) -> None:
        print(f"[DETAIL] {subtask} attempt {attempt} rubric checks:")
        print(f"         MSE: {rubric.loss}")
        print(f"         Score: {rubric.score}")
        if rubric.notes:
            print(f"         Notes: {rubric.notes}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task Evaluator Runner")
    # Changed defaults to relative paths or empty to avoid hardcoded absolute paths
    parser.add_argument("--env", default=".env", help="Path to env file")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_path = resolve_script_path(Path(args.env))
    env_config = EnvConfig.load(env_path, visualize=args.visualize)
    env_config.start_subtask = args.start_subtask
    env_config.eval_only = args.eval_only
    description_path = resolve_script_path(Path(args.description))
    coordinator = EvaluationCoordinator(env_config, description_path, visualize=args.visualize)
    coordinator.run()


if __name__ == "__main__":
    main()
