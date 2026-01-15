#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import textwrap
import asyncio
import glob
import urllib.request
import urllib.error
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
# Helpers & LLM Client
# --------------------------------------------------------------------------- #


def resolve_script_path(path: Path) -> Path:
    """Resolve a path relative to this script directory if not absolute."""
    path = Path(path)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


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


def copy_workspace_filtered(src: Path, dst: Path, clear: bool = True) -> None:
    src = src.resolve()
    dst = dst.resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source workspace missing: {src}")
    ensure_directory(dst)
    if clear:
        clear_directory(dst)
    
    # Exclude basic hidden files and venv
    exclude_prefixes = {".", "__", "venv", "node_modules", "site-packages"}
    
    for root, dirs, files in os.walk(src):
        rel_root = Path(root).relative_to(src)
        if any(p.startswith(tuple(exclude_prefixes)) for p in rel_root.parts):
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


def call_llm_judge(
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    use_json_format: bool = True
) -> Dict[str, Any]:
    """Simple HTTP client to call the evaluation LLM."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,  # Low temp for deterministic grading
        "max_tokens": 1000
    }

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            parsed = _parse_json_content(content)
            if parsed is not None:
                return parsed
            return {
                "score": 0, 
                "pass_count": 0, 
                "total_points": 0, 
                "failed_points": ["Model output parse error", content[:200]]
            }
    except Exception as e:
        print(f"[EVAL ERROR] Failed to call LLM Judge: {e}")
        return {"score": 0, "pass_count": 0, "total_points": 0, "failed_points": [str(e)]}


def _parse_json_content(content: str) -> Optional[Dict[str, Any]]:
    """Parse JSON content, tolerating Markdown code fences."""
    candidates = [content]
    cleaned = content.strip()
    if "```" in cleaned:
        # Pull fenced block if present
        match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.S | re.I)
        if match:
            candidates.append(match.group(1).strip())
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            if lines[-1].startswith("```"):
                candidates.append("\n".join(lines[1:-1]).strip())
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


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
    
    # Agent Config
    sii_api_key: str
    sii_api_base: str
    sii_auth_type: str
    sii_target_model: str
    sii_system_prompt: str
    sii_max_turns: int
    bridge_timeout_ms: int
    
    # Evaluator Config
    eval_api_key: str
    eval_api_base: str
    eval_model: str
    
    eval_only: bool = False

    @classmethod
    def load(cls, env_path: Path, visualize: bool) -> "EnvConfig":
        env_path = resolve_script_path(env_path)
        values = load_env_file(env_path)
        
        def fetch(key: str, default: str = "") -> str:
            return values.get(key, os.environ.get(key, default))

        def require(key: str) -> str:
            val = fetch(key)
            if not val or val.startswith("<YOUR_"):
                raise ValueError(f"Environment variable '{key}' must be set in {env_path}")
            return val

        sii_target = require("SII_TARGET_MODEL")
        
        # Load Eval Config
        eval_key = fetch("EVAL_TEXT_API_KEY")
        eval_base = fetch("EVAL_TEXT_API_BASE_URL")
        eval_model = fetch("EVAL_TEXT_MODEL")
        
        # Fallback to Agent config if Eval not set
        if not eval_key:
            eval_key = require("SII_AGENT_API_KEY")
            eval_base = fetch("SII_AGENT_API_BASE_URL") or "https://api.openai.com/v1"
            eval_model = "gpt-4o" # Default judge

        return cls(
            env_path=env_path,
            visualize=visualize,
            env_values=values,
            model_name=derive_model_name(sii_target),
            scorer_name="llm-as-judge",
            max_attempts=int(fetch("MAX_SUBTASK_ATTEMPTS", "3")),
            conda_env_name=fetch("CONDA_ENV_NAME"),
            sii_api_key=require("SII_AGENT_API_KEY"),
            sii_api_base=fetch("SII_AGENT_API_BASE_URL", "https://openrouter.ai/api/v1"),
            sii_auth_type=fetch("SII_AUTH_TYPE", "USE_OPENAI"),
            sii_target_model=sii_target,
            sii_system_prompt=fetch("SII_SYSTEM_PROMPT", ""),
            sii_max_turns=int(fetch("SII_MAX_TURNS", "20")),
            bridge_timeout_ms=int(fetch("BRIDGE_TIMEOUT_MS", "180000")),
            eval_api_key=eval_key,
            eval_api_base=eval_base,
            eval_model=eval_model
        )

    def inject_defaults(self) -> None:
        defaults = {
            "SII_AGENT_API_KEY": self.sii_api_key,
            "SII_AGENT_API_BASE_URL": self.sii_api_base,
            "OPENAI_API_KEY": self.sii_api_key,
            "OPENAI_BASE_URL": self.sii_api_base,
            # Compat for SDK expecting these keys
            "SII_OPENAI_API_KEY": self.sii_api_key,
            "SII_OPENAI_BASE_URL": self.sii_api_base,
        }
        merged = {**defaults, **self.env_values}
        for key, value in merged.items():
            if value:
                os.environ.setdefault(key, value)
                self.env_values.setdefault(key, value)


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
    score: int
    pass_count: int
    total_points: int
    failed_points: List[str] = field(default_factory=list)
    feedback: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subtask": self.subtask,
            "score": self.score,
            "pass_count": self.pass_count,
            "total_points": self.total_points,
            "failed_points": self.failed_points,
            "feedback": self.feedback
        }


@dataclass
class AttemptSummary:
    subtask: str
    attempt_index: int
    score: int
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
# Agent runner
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
            self._session = SDKAgentSession(base_options)
        except Exception as exc:
            print(f"[AGENT] Failed to initialize SII session: {exc}")
            self._available = False

    def send(self, prompt: str, workspace_hint: str, workspace: Path) -> str:
        if not self._available or not self._session:
            return ""
        try:
            options = self._build_options(workspace)
            combined_prompt = f"{workspace_hint}\n\n{prompt}" if workspace_hint else prompt
            return asyncio.run(self._run_session(self._session, combined_prompt, options))
        except Exception as exc:
            print(f"[AGENT] Agent invocation failed: {exc}")
            return ""

    async def _run_session(self, session: SDKAgentSession, prompt: str, options: SiiAgentOptions) -> str:
        assistant_chunks: List[str] = []
        async for message in session.run(prompt, options=options):
            if isinstance(message, AssistantMessage):
                text = self._text_from_assistant(message)
                if text:
                    assistant_chunks.append(text)
        return "\n".join(chunk.strip() for chunk in assistant_chunks if chunk).strip()

    def _build_options(self, workspace: Path) -> SiiAgentOptions:
        workspace = workspace.resolve()
        # Pass minimal env to agent
        env_vars = {k: v for k, v in self.env_config.env_values.items() if v}
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
# Command runner (Adapted for Data Engineering Tasks)
# --------------------------------------------------------------------------- #


class CommandRunner:
    """Execute Python scripts found in datascripts/ and capture outputs."""

    def __init__(self, logs_dir: Path, conda_env: Optional[str] = None, conda_exe: Optional[str] = None):
        self.logs_dir = ensure_directory(logs_dir)
        self.conda_env = conda_env
        self.conda_exe = conda_exe

    def capture(self, subtask: str, root: Path) -> Dict[str, Any]:
        root = root.resolve()
        results: Dict[str, CommandResult] = {}
        
        # 1. Identify python scripts in datascripts/ folder
        # The task requires scripts to be placed in datascripts/
        scripts_dir = root / "datascripts"
        scripts = sorted(list(scripts_dir.glob("*.py"))) if scripts_dir.exists() else []
        
        # 2. Run Scripts
        if not scripts:
             # If no specific scripts, try running a basic check
             cmd = self._wrap_cmd(["python", "--version"])
             res = self._run(cmd, root, "check_python")
             results["check_python"] = res
        
        for script in scripts:
            rel_path = script.relative_to(root)
            cmd = self._wrap_cmd(["python", str(rel_path)])
            res = self._run(cmd, root, f"run_{script.stem}")
            results[res.name] = res

        # 3. List Generated Files (Lightweight 'tree')
        file_tree = self._list_files(root)
        results["file_structure"] = CommandResult(
            name="file_structure",
            command=["ls", "-R"],
            returncode=0,
            stdout=file_tree,
            stderr=""
        )

        self._persist(results)
        return {name: res.to_dict() for name, res in results.items()}

    def _run(self, cmd: List[str], cwd: Path, name: str) -> CommandResult:
        rc, out, err = run_command(cmd, cwd=cwd, timeout=300)
        return CommandResult(name=name, command=cmd, returncode=rc, stdout=out, stderr=err)

    def _wrap_cmd(self, cmd: Sequence[str]) -> List[str]:
        # If conda env is specified, wrap command
        if self.conda_env and self.conda_exe:
            return [self.conda_exe, "run", "--no-capture-output", "-n", self.conda_env, *cmd]
        return list(cmd)

    def _list_files(self, root: Path) -> str:
        lines = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and ".git" not in path.parts:
                lines.append(str(path.relative_to(root)))
        return "\n".join(lines)

    def _persist(self, results: Dict[str, CommandResult]) -> None:
        for name, result in results.items():
            log_path = self.logs_dir / f"{name}.txt"
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


# --------------------------------------------------------------------------- #
# Rubric evaluation (LLM-as-a-Judge Implementation)
# --------------------------------------------------------------------------- #


class RubricEvaluator:
    
    def __init__(self, env_config: EnvConfig):
        self.config = env_config

    def evaluate(
        self, subtask: str, evalspace: Path, command_results: Dict[str, Any]
    ) -> RubricResult:
        
        # 1. Gather Context
        context = self._gather_context(evalspace, command_results)
        
        # 2. Select Prompt based on Subtask
        system_prompt, user_prompt = self._build_prompts(subtask, context)
        print(f"[PROMPT][{subtask}] System:\n{system_prompt}\n\nUser:\n{user_prompt}")
        try:
            logs_dir = ensure_directory(evalspace / "logs")
            prompt_path = logs_dir / f"prompt_{subtask}.txt"
            prompt_body = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"
            prompt_path.write_text(prompt_body, encoding="utf-8")
        except Exception as exc:
            print(f"[PROMPT][{subtask}] Failed to write prompt log: {exc}")
        
        # 3. Call LLM
        print(f"[EVAL] Calling LLM Judge for {subtask}...")
        llm_response = call_llm_judge(
            self.config.eval_api_key,
            self.config.eval_api_base,
            self.config.eval_model,
            system_prompt,
            user_prompt
        )
        try:
            debug_dump = json.dumps(llm_response, indent=2)
            print(f"[LLM JUDGE RAW][{subtask}]\n{debug_dump}")
        except Exception:
            print(f"[LLM JUDGE RAW][{subtask}] <unserializable response: {llm_response}>")
        
        # 4. Parse Result
        raw_score = llm_response.get("score", 0)
        raw_pass = llm_response.get("pass_count", 0)
        raw_total = llm_response.get("total_points", 10) or 10
        # Normalize to 10-point scale
        if raw_total and raw_total != 10:
            factor = 10 / raw_total
            score = int(round(raw_score * factor))
            pass_count = int(round(raw_pass * factor))
            total_points = 10
        else:
            score = raw_score
            pass_count = raw_pass
            total_points = 10

        score = max(0, min(10, score))
        pass_count = max(0, min(10, pass_count))
        failed_points = llm_response.get("failed_points", [])
        if isinstance(failed_points, str):
            failed_points = [failed_points]
        feedback = llm_response.get("feedback", "No feedback provided.")

        return RubricResult(
            subtask=subtask,
            score=score,
            pass_count=pass_count,
            total_points=total_points,
            failed_points=failed_points,
            feedback=feedback
        )

    def _gather_context(self, root: Path, commands: Dict[str, Any]) -> str:
        """Collects file contents and command logs to show the LLM."""
        sections = []
        
        # Section A: File Structure
        file_tree = commands.get("file_structure", {}).get("stdout", "No file listing available.")
        sections.append(f"### File Structure (root={root.name})\n{file_tree}")
        
        # Section B: Markdown Reports (Orientation / Validation / READMEs)
        # Check root MD files
        for md_file in sorted(root.glob("*.md")):
            content = md_file.read_text(encoding="utf-8", errors="replace")
            sections.append(f"### Report: {md_file.name}\n{content[:6000]} ... [truncated]")

        # Check READMEs in dataraw
        dataraw_dir = root / "dataraw"
        if dataraw_dir.exists():
            for readme in sorted(dataraw_dir.glob("*.md")):
                content = readme.read_text(encoding="utf-8", errors="replace")
                sections.append(f"### README Artifact: dataraw/{readme.name}\n{content[:2000]} ... [truncated]")
        
        # Section C: JSON Deliverables (The core outputs)
        for json_file in sorted(root.glob("datasearch_data_*.json")):
            content = json_file.read_text(encoding="utf-8", errors="replace")
            sections.append(f"### Generated JSON: {json_file.name}\n{content[:6000]} ... [truncated]")
            
        # Section D: Scripts
        datascripts_dir = root / "datascripts"
        if datascripts_dir.exists():
            for py_file in datascripts_dir.glob("*.py"):
                content = py_file.read_text(encoding="utf-8", errors="replace")
                sections.append(f"### Script Content: datascripts/{py_file.name}\n{content[:3000]} ... [truncated]")

        # Section E: Execution Logs
        logs = []
        for name, cmd_data in commands.items():
            if name == "file_structure": continue
            stdout = cmd_data.get("stdout", "")
            stderr = cmd_data.get("stderr", "")
            rc = cmd_data.get("returncode")
            logs.append(f"Command: {name} (Return Code: {rc})\nSTDOUT:\n{stdout[-1000:]}\nSTDERR:\n{stderr[-1000:]}")
        
        sections.append("### Execution Logs\n" + "\n---\n".join(logs))
        
        return "\n\n".join(sections)

    def _build_prompts(self, subtask: str, context: str) -> Tuple[str, str]:
        system_prompt = (
            "You are an expert Data Engineer Evaluator. "
            "Your job is to grade a student's submission based on a strict rubric involving dataset collection and JSON schema compliance. "
            "Each criterion is scored 0-10; compute the raw total then scale the final score to a 0-10 range. "
            "Respond ONLY with a raw JSON object (no code fences, no markdown, no prose). "
            "Schema: {'score': int (0-10 scaled), 'pass_count': int (scaled points), 'total_points': 10, 'failed_points': [list of strings], 'feedback': 'string'}. "
            "If any required field is missing or cannot be determined, still return a JSON object with 0 scores and an explanatory failed_points entry."
        )

        rubric_criteria = ""
        if subtask == "subtask1":
            rubric_criteria = textwrap.dedent("""
                Rubric (Total 10 pts):
                - Score 10: Briefing (`datacontext_brief.md`) covers every constraint from the JSON spec and AGENTS instructions, references all initial files (test_data_*.json) with correct paths, and provides a step-by-step plan with risk mitigation notes.
                - Score 5: Briefing captures major requirements but misses at least one constraint (e.g., workspace-only rule or JSON field definition) or lacks an actionable plan.
                - Score 0: Briefing missing, written outside workspace, or omits most requirements.
                
                Check for: `datacontext_brief.md`.
            """)
        elif subtask == "subtask2":
            rubric_criteria = textwrap.dedent("""
                Rubric (Total 10 pts):
                - Score 10: Dataset is public and synthetic (AI consciousness theme), README + random sample captured via script. `datasearch_data_1.json` fields perfectly mirror the schema with precise descriptions. Script (`datascripts/fetch_consciousness_dataset.py`) reruns without manual edits.
                - Score 7: Dataset and JSON mostly correct but script has minor reproducibility gaps (e.g., hard-coded paths) or metadata omits one secondary field.
                - Score 3: Dataset mismatches requirements (not synthetic/QA) or README/sample missing; JSON incomplete.
                - Score 0: No deliverables or dataset not from Hugging Face.
                
                Check for: `datasearch_data_1.json`, `datascripts/fetch_consciousness_dataset.py`, `dataraw/*_readme.md`.
            """)
        elif subtask == "subtask3":
            rubric_criteria = textwrap.dedent("""
                Rubric (Total 10 pts):
                - Score 10: Dataset satisfies structural requirements (Academic MC, subject tags, 4 choices, integer labels 0-3, ~10K-15K samples). `datasearch_data_2.json` precisely documents the task. Script + README artifacts reproducible.
                - Score 6: Dataset mostly compliant but size or documentation slightly outside spec, or JSON misses one descriptive element.
                - Score 2: Dataset choice invalid (wrong task, size far off) or deliverables incomplete.
                - Score 0: Nothing produced.
                
                Check for: `datasearch_data_2.json`, `datascripts/fetch_transenv_dataset.py`, `dataraw/*_readme.md`.
            """)
        elif subtask == "subtask4":
            rubric_criteria = textwrap.dedent("""
                Rubric (Total 10 pts):
                - Score 10: Dataset matches Danish tweets, OFF/NOT labels, few-hundred samples. README + sample saved. Metadata descriptions grounded in evidence. Script reruns deterministically. Non-gated.
                - Score 6: Dataset fits but metadata is partially generic or sample handling is fragile.
                - Score 2: Dataset gated, wrong language, or JSON fails validation.
                - Score 0: Deliverables missing.
                
                Check for: `datasearch_data_3.json`, `datascripts/fetch_danish_offense_dataset.py`, `dataraw/*_readme.md`.
            """)
        elif subtask == "subtask5":
            rubric_criteria = textwrap.dedent("""
                Rubric (Total 10 pts):
                - Score 10: `datavalidation_report.md` enumerates every validation command with outcomes, links to all deliverables, and confirms schema/reproducibility.
                - Score 5: Report exists but omits one JSON check or lacks reproduction notes.
                - Score 1: Superficial report without concrete commands or evidence.
                - Score 0: No report or validation performed.
                
                Check for: `datavalidation_report.md`, Execution Logs (did they run validation?).
            """)

        user_prompt = f"""
        Task: {subtask}
        
        {rubric_criteria}
        
        EVIDENCE PROVIDED BELOW:
        {context}
        
        Evaluate the submission against the Rubric.
        If files are missing or scripts failed, deduct points heavily.
        """
        
        return system_prompt, user_prompt


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
        self.run_root = (self.base_dir / env_config.model_name).resolve()
        self.shared_workspace = self.run_root / "workspace"
        self.shared_evalspace = self.run_root / "evalspace"
        self.eval_only = getattr(env_config, "eval_only", False)
        # Allow disabling response_format json_object if backend does not support it
        # Use new LLM Evaluator
        self.rubric = RubricEvaluator(env_config)
        self.meta: Dict[str, Any] = {
            "model": env_config.model_name,
            "scorer": env_config.scorer_name,
            "max_attempts": env_config.max_attempts,
            "subtasks": [],
        }

    def prepare_layout(self) -> None:
        ensure_directory(self.run_root)
        # Use shared workspace/evalspace across all subtasks to keep artifacts available.
        if not self.eval_only:
            clear_directory(self.shared_workspace)
            clear_directory(self.shared_evalspace)
        ensure_directory(self.shared_workspace)
        ensure_directory(self.shared_evalspace)

    def run(self) -> Dict[str, Any]:
        self.prepare_layout()
        self.env_config.inject_defaults()
        
        # Check Conda
        conda_exe = self._resolve_conda_executable()
        conda_exe = self._verify_conda_environment(conda_exe)
        
        agent = None if self.eval_only else AgentRunner(self.env_config, visualize=self.visualize)
        subtask_count = int(self.description.get("subtask_count") or 0)
        start_index = max(1, getattr(self.env_config, "start_subtask", 1))

        print(f"[BOOT] Model={self.env_config.model_name} scorer={self.env_config.scorer_name}")
        
        for index in range(start_index, subtask_count + 1):
            subtask = f"subtask{index}"
            prompt = self.description.get(subtask, "")
            attempt_summaries: List[AttemptSummary] = []
            feedback: str = ""
            
            print(f"[SUBTASK] Starting {subtask}")
            attempt_limit = 1 if self.eval_only else self.env_config.max_attempts
            
            for attempt in range(1, attempt_limit + 1):
                workspace, evalspace = self._prepare_attempt_dirs(subtask, attempt)
                
                # Copy previous attempt or base data if needed
                self._provision_data(workspace)

                # 1. Run Agent
                if not self.eval_only and agent:
                    agent_output = agent.send(
                        prompt + ("\n\nRubric Feedback from previous attempt:\n" + feedback if feedback else ""),
                        f"You are working in {workspace}. Python environment is ready.",
                        workspace,
                    )
                else:
                    agent_output = ""

                # 2. Snapshot to Evalspace
                if workspace != evalspace:
                    # Preserve prior outputs for downstream subtasks (no clearing).
                    copy_workspace_filtered(workspace, evalspace, clear=False)
                logs_dir = ensure_directory(evalspace / "logs")
                
                # 3. Execute Verification Commands (Run Data Scripts)
                cmd_runner = CommandRunner(
                    logs_dir,
                    conda_env=self.env_config.conda_env_name if conda_exe else None,
                    conda_exe=conda_exe,
                )
                commands = cmd_runner.capture(subtask, evalspace)

                # 4. Evaluate with LLM Judge
                rubric_result = self.rubric.evaluate(subtask, evalspace, commands)
                feedback = rubric_result.feedback

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
                
                print(f"[RESULT] {subtask} attempt {attempt}: score={rubric_result.score}/10")
                if rubric_result.failed_points:
                    print(f"         Issues: {rubric_result.failed_points}")

                if not self.eval_only and rubric_result.score >= 10:
                    break

            # Save Meta
            best = max(attempt_summaries, key=lambda item: item.score) if attempt_summaries else None
            self.meta["subtasks"].append({
                "name": subtask,
                "attempts": [item.to_dict() for item in attempt_summaries],
                "best_score": best.score if best else 0
            })

        self._save_meta()
        return self.meta

    def _provision_data(self, workspace: Path) -> None:
        """Ensures the source data (test schemas) is available in the agent's workspace."""
        data_src = self.base_dir / "data"
        if data_src.exists():
            data_dst = workspace / "data"
            if not data_dst.exists():
                shutil.copytree(data_src, data_dst)

    def _prepare_attempt_dirs(
        self, subtask: str, attempt_index: int
    ) -> Tuple[Path, Path]:
        # All subtasks share a single workspace/evalspace to enable downstream integration.
        workspace = self.shared_workspace.resolve()
        evalspace = self.shared_evalspace.resolve()
        ensure_directory(workspace)
        ensure_directory(evalspace)
        return workspace, evalspace

    def _save_meta(self) -> None:
        meta_path = self.run_root / "meta_eval.json"
        meta_path.write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        print(f"[DONE] Wrote results to {meta_path}")

    def _resolve_conda_executable(self) -> Optional[str]:
        # Simple lookup
        candidates = [
            self.env_config.env_values.get("CONDA_EXE"),
            os.environ.get("CONDA_EXE"),
            shutil.which("conda"),
        ]
        for c in candidates:
            if c and os.path.exists(c): return c
        return None

    def _verify_conda_environment(self, conda_exe: Optional[str]) -> Optional[str]:
        if not conda_exe or not self.env_config.conda_env_name:
            return None
        # Quick check
        cmd = [conda_exe, "run", "-n", self.env_config.conda_env_name, "python", "--version"]
        rc, _, _ = run_command(cmd, timeout=30)
        return conda_exe if rc == 0 else None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task17 Data Engineering Evaluator")
    default_env = (SCRIPT_DIR.parent / ".env").resolve()
    default_description = (SCRIPT_DIR / "description.json").resolve()
    parser.add_argument("--env", default=str(default_env), help="Path to env file")
    parser.add_argument("--visualize", action="store_true", help="Enable Agent visualization")
    parser.add_argument("--description", default=str(default_description), help="Path to task description JSON")
    parser.add_argument("--start-subtask", type=int, default=1, help="Start from subtask N")
    parser.add_argument("--eval-only", action="store_true", help="Skip agent generation, only evaluate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_path = resolve_script_path(Path(args.env))
    # Load config with new Eval fields
    env_config = EnvConfig.load(env_path, visualize=args.visualize)
    env_config.start_subtask = args.start_subtask
    env_config.eval_only = args.eval_only
    
    description_path = resolve_script_path(Path(args.description))
    coordinator = EvaluationCoordinator(env_config, description_path, visualize=args.visualize)
    
    print("[INFO] Data Engineering Evaluation Started")
    coordinator.run()
    print("[INFO] Evaluation Finished")


if __name__ == "__main__":
    main()
