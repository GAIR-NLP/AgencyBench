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
import re
import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 直接内嵌 evaluation/task/evaluation.py 的判定提示与解析函数，避免额外依赖
VERIFICATION_PROMPT_TEMPLATE = """
Make a verification on whether the following response to the question is correct or not based on the ground truth answer.

Here is the question: 
<question>
{question}
</question>

Here is the response:
<response>
{response}
</response>

Here is the ground truth answer:
<ground_truth_answer>
{ground_truth_answer}
</ground_truth_answer>


Criteria for correctness:
- When the exact answer in the response is the same as the ground truth answer regardless of the format, the response is correct.
- For numerical answers, the response is correct if the exact answer in the response is within a small margin of the ground truth answer.

You should first think carefully for the correctness of the response.

You should output the verification result in the following format, and the result should be one of ["correct", "incorrect"]:
<verification_result>
( your verification result here, should be correct or incorrect )
</verification_result>
""".strip()


def get_content_from_tag(
    content: str,
    tag: str,
    default_value: Optional[str] = None
) -> Optional[str]:
    """
    extract the content from the first specified tag
    """
    if not content:
        return default_value

    pattern = rf"<{tag}>(.*?)(?=(</{tag}>|<\w+|$))"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return default_value


# 评分依赖 litellm，若不可用则回退
try:
    from litellm import batch_completion, ModelResponse
    LITELLM_AVAILABLE = True
except Exception:
    batch_completion = None
    ModelResponse = None
    LITELLM_AVAILABLE = False

# 尝试导入 SDK，如果不存在则设为 None (用于兼容)
try:
    from sii_agent_sdk import (  # type: ignore
        AssistantMessage,
        TextBlock,
        SiiAgentOptions,
        SiiAgentSession as SDKAgentSession,
    )
except ImportError:
    AssistantMessage = None
    TextBlock = None
    SiiAgentOptions = None
    SDKAgentSession = None

SCRIPT_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Helpers (无 LLM 调用)
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



def copy_workspace(src: Path, dst: Path) -> None:
    src = src.resolve()
    dst = dst.resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source workspace missing: {src}")
    ensure_directory(dst)
    clear_directory(dst)
    for root, dirs, files in os.walk(src):
        rel_root = Path(root).relative_to(src)
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
    
    # Evaluator Config (虽然Evaluator逻辑被删，但保留Config读取以免报错)
    eval_api_key: str
    eval_api_base: str
    eval_model: str
    
    eval_only: bool = False
    start_subtask: int = 1

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
        
        # Load Eval Config (Defaults)
        eval_key = fetch("EVAL_TEXT_API_KEY")
        eval_base = fetch("EVAL_TEXT_API_BASE_URL")
        eval_model = fetch("EVAL_TEXT_MODEL")
        
        if not eval_key:
            eval_key = "dummy_key"
            eval_base = "dummy_base"
            eval_model = "dummy_model"

        return cls(
            env_path=env_path,
            visualize=visualize,
            env_values=values,
            model_name=derive_model_name(sii_target),
            scorer_name="rubric", # Changed name
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
        scripts_dir = root / "datascripts"
        scripts = sorted(list(scripts_dir.glob("*.py"))) if scripts_dir.exists() else []
        
        # 2. Run Scripts
        if not scripts:
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
# Rubric evaluation (Interface ONLY - Logic Removed)
# --------------------------------------------------------------------------- #


class RubricEvaluator:
    
    def __init__(self, env_config: EnvConfig):
        self.config = env_config

    def evaluate(
        self, subtask: str, evalspace: Path, command_results: Dict[str, Any]
    ) -> RubricResult:
        """
        使用 evaluation/task/evaluation.py 的判分方式，保持评测代码和参考数据目录不变：
        - 预测文件：<evalspace>/outputs/test.json
        - 参考文件：<script_dir>/evaluation/data/references/test.json
        - 直接调用 evaluation.task.evaluation.TaskBenchmark.run_1
        """
        eval_module = self._load_eval_module()
        if eval_module is None:
            return RubricResult(
                subtask=subtask,
                score=0,
                pass_count=0,
                total_points=10,
                failed_points=["evaluation module missing"],
                feedback="evaluation/task/evaluation.py not found or failed to load."
            )

        # 构造与 evaluation/task/evaluation.py 一致的 config，调整路径：
        # - workspace 指向 evalspace（其中我们放 outputs/test.json）
        # - eval_workspace 指向脚本目录，task_name="evaluation" => 参考文件 evaluation/data/references/test.json
        batch_size = int(self.config.env_values.get("EVAL_BATCH_SIZE", "10") or 10)
        config_obj = SimpleNamespace(
            workspace=str(Path(evalspace).resolve()),
            eval_workspace=str(SCRIPT_DIR),
            task_name="evaluation",
            batch_size=batch_size,
            model_name=self.config.eval_model or self.config.model_name,
            api_key=self.config.eval_api_key,
            api_version=getattr(self.config, "eval_api_version", None),
            api_endpoint=self.config.eval_api_base,
        )

        try:
            ref_file = Path(SCRIPT_DIR) / "evaluation" / "data" / "references" / "test.json"
            pred_file = Path(evalspace).resolve() / "outputs" / "test.json"
            print(f"[EVAL][{subtask}] refs={ref_file} exists={ref_file.exists()} preds={pred_file} exists={pred_file.exists()} model={config_obj.model_name} batch_size={config_obj.batch_size}")
            try:
                ref_len = len(json.loads(ref_file.read_text(encoding="utf-8"))) if ref_file.exists() else "NA"
            except Exception as exc:
                ref_len = f"error:{exc}"
            try:
                pred_len = len(json.loads(pred_file.read_text(encoding="utf-8"))) if pred_file.exists() else "NA"
            except Exception as exc:
                pred_len = f"error:{exc}"
            print(f"[EVAL][{subtask}] ref_len={ref_len} pred_len={pred_len}")
            benchmark = eval_module.TaskBenchmark(config_obj)
            # 覆盖输出/参考路径：预测放在 <evalspace>/outputs
            benchmark.output_dir_path = os.path.join(str(Path(evalspace).resolve()), "outputs")
            benchmark.reference_dir_path = os.path.join(str(SCRIPT_DIR), "evaluation", "data", "references")
            result = benchmark.run_1()
        except Exception as exc:
            return RubricResult(
                subtask=subtask,
                score=0,
                pass_count=0,
                total_points=10,
                failed_points=[f"evaluation run failed: {exc}"],
                feedback=f"Failed to execute evaluation: {exc}"
            )

        if not isinstance(result, dict):
            return RubricResult(
                subtask=subtask,
                score=0,
                pass_count=0,
                total_points=10,
                failed_points=["invalid evaluation result"],
                feedback="Evaluation did not return a dict."
            )

        if result.get("error"):
            return RubricResult(
                subtask=subtask,
                score=0,
                pass_count=0,
                total_points=10,
                failed_points=[str(result.get("error"))],
                feedback=str(result.get("test_set_result", {}).get("message") or result.get("error"))
            )

        # evaluation/task/evaluation.py 返回 score 0-100，映射到 0-10 档
        raw_score = result.get("score", 0) or 0
        rubric_score = max(0, min(10, int(round(float(raw_score) / 10))))
        feedback = result.get("test_set_result", {}).get("message", f"score={raw_score}")

        return RubricResult(
            subtask=subtask,
            score=rubric_score,
            pass_count=rubric_score,
            total_points=10,
            failed_points=[] if rubric_score == 10 else ["Partial score"],
            feedback=feedback
        )

    def _load_eval_module(self):
        """
        动态加载 evaluation/task/evaluation.py；如缺失外部依赖 evaluations.base.*，注入最小桩以便导入成功。
        """
        eval_file = SCRIPT_DIR / "evaluation" / "task" / "evaluation.py"
        if not eval_file.exists():
            return None

        # 注入缺失依赖桩
        if "evaluations" not in sys.modules:
            sys.modules["evaluations"] = ModuleType("evaluations")
        if "evaluations.base" not in sys.modules:
            sys.modules["evaluations.base"] = ModuleType("evaluations.base")
        if "evaluations.base.data_classes" not in sys.modules:
            data_mod = ModuleType("evaluations.base.data_classes")
            class Config:  # minimal stub
                pass
            data_mod.Config = Config
            sys.modules["evaluations.base.data_classes"] = data_mod
            setattr(sys.modules["evaluations.base"], "data_classes", data_mod)
        if "evaluations.base.base_eval" not in sys.modules:
            base_mod = ModuleType("evaluations.base.base_eval")
            class BaseBenchmark:  # minimal stub
                def __init__(self, config=None):
                    self.config = config
            base_mod.BaseBenchmark = BaseBenchmark
            sys.modules["evaluations.base.base_eval"] = base_mod
            setattr(sys.modules["evaluations.base"], "base_eval", base_mod)

        spec = importlib.util.spec_from_file_location("evaluation.task.evaluation", eval_file)
        if not spec or not spec.loader:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules["evaluation.task.evaluation"] = module
        try:
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
        except Exception as exc:
            print(f"[EVAL] failed to load evaluation module: {exc}")
            return None
        return module


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
        
        self.rubric = RubricEvaluator(env_config)
        self.meta: Dict[str, Any] = {
            "model": env_config.model_name,
            "scorer": env_config.scorer_name,
            "max_attempts": env_config.max_attempts,
            "subtasks": [],
        }

    def prepare_layout(self) -> None:
        ensure_directory(self.run_root)
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
                
                if attempt == 1:
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
                    copy_workspace(workspace, evalspace)
                logs_dir = ensure_directory(evalspace / "logs")
                
                # 3. Execute Verification Commands
                cmd_runner = CommandRunner(
                    logs_dir,
                    conda_env=self.env_config.conda_env_name if conda_exe else None,
                    conda_exe=conda_exe,
                )
                commands = cmd_runner.capture(subtask, evalspace)

                # 4. Evaluate (Using the placeholder evaluator)
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
        source_src = self.base_dir / "source"
        if source_src.exists():
            source_dst = workspace 
            shutil.copytree(source_src, source_dst, dirs_exist_ok=True)

    def _prepare_attempt_dirs(
        self, subtask: str, attempt_index: int
    ) -> Tuple[Path, Path]:
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
