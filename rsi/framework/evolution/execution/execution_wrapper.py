"""ExecutionWrapper：运行 PipelineAgent 产出的 DataFlow 流水线代码并捕获崩溃栈。"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from rsi.framework.evolution.models import PipelineConfig
from rsi.framework.evolution.execution.step_cache import (
    MANIFEST_FILENAME,
    REUSE_PLAN_FILENAME,
    prepare_step_reuse,
    write_step_manifest,
)
from rsi.framework.evolution.execution.execution_observation import write_execution_observation
from rsi.framework.evolution.telemetry.pipeline_usage import (
    ATTEMPT_ENV,
    BOOTSTRAP_DIR,
    EXECUTION_ID_ENV,
    ITERATION_ENV,
    USAGE_PATH_ENV,
    write_combined_iteration_usage,
    write_pipeline_usage_summary,
)
from rsi.framework.evolution.utils.logging import get_logger

logger = get_logger("engine.execution")

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]

# Keep complete subprocess output on disk, but do not copy progress-bar noise into
# an Agent repair request. These limits stay below the provider input cap.
MAX_TRACEBACK_CHARS = 32 * 1024
MAX_DIAGNOSTICS_CHARS = 64 * 1024
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_PROGRESS_LINE_RE = re.compile(r"(?:\d{1,3}%\||it/s|Generating responses from prompts)")


def _bounded_text(value: str | None, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    marker = f"\n...[truncated to {limit} chars]...\n"
    budget = max(0, limit - len(marker))
    head = budget // 3
    return text[:head] + marker + text[-(budget - head):]


def _compact_failure_text(value: str | None) -> str:
    """Remove terminal progress noise while retaining the useful exception tail."""
    text = _ANSI_ESCAPE_RE.sub("", str(value or ""))
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("\r")[-1].strip()
        if not line or _PROGRESS_LINE_RE.search(line):
            continue
        lines.append(line)
    return _bounded_text("\n".join(lines), MAX_TRACEBACK_CHARS)


def _write_process_logs(
    iteration_dir: Path,
    stdout: str | None,
    stderr: str | None,
    prefix: str,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    for stream_name, content in (("stdout", stdout), ("stderr", stderr)):
        path = iteration_dir / f"{prefix}_{stream_name}.log"
        try:
            path.write_text(str(content or ""), encoding="utf-8")
            paths[f"{prefix}_{stream_name}_log"] = str(path.resolve())
        except OSError as exc:
            logger.warning("写执行日志失败：%s (%s)", path, exc)
    return paths


def _try_write_observation(
    iteration_dir: Path,
    *,
    config: PipelineConfig,
    raw_entry_path: str,
    final_dataset_path: str | None,
    status: str,
    failure_stage: str | None,
    error: str | None,
    sample_rows_per_step: int,
    max_sample_chars: int,
) -> str | None:
    try:
        return str(
            write_execution_observation(
                iteration_dir,
                config=config,
                raw_entry_path=raw_entry_path,
                final_dataset_path=final_dataset_path,
                status=status,
                failure_stage=failure_stage,
                error=error,
                sample_rows_per_step=sample_rows_per_step,
                max_sample_chars=max_sample_chars,
            )
        )
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("写 execution observation 失败（不影响主流程）：%s", exc)
        return None


@dataclass
class ExecutionResult:
    """流水线执行结果。"""

    success: bool
    dataset_path: str | None = None        # 成功时：生成的数据集 jsonl 路径
    traceback: str | None = None           # 失败时：崩溃栈，喂回 PipelineAgent 自修正
    stdout: str = ""
    stderr: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)
    diagnostics: str = ""
    failure_stage: str | None = None
    observation_path: str | None = None


class ExecutionWrapperABC(ABC):
    """流水线执行器接口。"""

    @abstractmethod
    def run(
        self,
        config: PipelineConfig,
        iteration_dir: Path,
        env_overrides: dict[str, str] | None = None,
        parent_iteration_dir: Path | None = None,
    ) -> ExecutionResult:
        """Execute the generated pipeline inside one iteration directory."""
        raise NotImplementedError


class SubprocessExecutionWrapper(ExecutionWrapperABC):
    """Run a generated pipeline from the canonical Agent workspace.

    ``workspace_dir`` contains Agent artifacts. Generated operators import the
    shared DataLite Pipeline and stores from this repository.
    """

    def __init__(
        self,
        workspace_dir: str,
        pipeline_filename: str = "pipeline.py",
        compile_timeout_sec: int = 120,
        timeout_sec: int = 1800,
        python_exe: str | None = None,
        extra_env: dict[str, str] | None = None,
        raw_entry_path: str | None = None,
        observation_sample_rows: int = 2,
        observation_max_sample_chars: int = 4000,
    ) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()
        self.pipeline_filename = pipeline_filename
        self.compile_timeout_sec = compile_timeout_sec
        self.timeout_sec = timeout_sec
        self.python_exe = python_exe or sys.executable
        self.extra_env = extra_env or {}
        self.raw_entry_path = str(Path(raw_entry_path).resolve()) if raw_entry_path else ""
        self.observation_sample_rows = max(0, int(observation_sample_rows))
        self.observation_max_sample_chars = max(256, int(observation_max_sample_chars))
        self._execution_attempts: dict[str, int] = {}

    def run(
        self,
        config: PipelineConfig,
        iteration_dir: Path,
        env_overrides: dict[str, str] | None = None,
        parent_iteration_dir: Path | None = None,
    ) -> ExecutionResult:
        iteration_dir = iteration_dir.resolve()
        pipeline_path = iteration_dir / self.pipeline_filename
        if not pipeline_path.exists():
            return ExecutionResult(
                False,
                traceback=f"流水线文件不存在：{pipeline_path}",
                failure_stage="artifact_validation",
            )

        artifact_errors = _validate_pipeline_artifact(pipeline_path)
        if artifact_errors:
            logger.warning("Pipeline artifact 校验失败：%s", "; ".join(artifact_errors))
            return ExecutionResult(
                False,
                traceback="Pipeline artifact validation failed:\n- " + "\n- ".join(artifact_errors),
                diagnostics=json.dumps(
                    {"stage": "artifact_validation", "errors": artifact_errors},
                    ensure_ascii=False,
                    indent=2,
                ),
                failure_stage="artifact_validation",
            )

        raw_entry_path = self.raw_entry_path
        if not raw_entry_path:
            return ExecutionResult(
                False,
                traceback="framework raw_entry_path is not configured",
                failure_stage="artifact_validation",
            )
        reuse_plan = prepare_step_reuse(
            config,
            iteration_dir,
            parent_iteration_dir=parent_iteration_dir,
            raw_entry_path=raw_entry_path,
            pipeline_filename=self.pipeline_filename,
        )
        logger.info(
            "Framework step cache：prefix=%d/%d source=%s reason=%s",
            reuse_plan.prefix_count,
            len(reuse_plan.operator_names),
            reuse_plan.entry_path,
            reuse_plan.reason,
        )

        env = {**os.environ, **self.extra_env, **reuse_plan.env()}
        # Generated code imports only the framework package from this repository.
        prev_pp = env.get("PYTHONPATH", "")
        inherited_paths = []
        for item in (p for p in prev_pp.split(os.pathsep) if p):
            try:
                if Path(item).resolve() == self.workspace_dir:
                    continue
            except OSError:
                pass
            inherited_paths.append(item)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(REPOSITORY_ROOT), str(BOOTSTRAP_DIR), *inherited_paths]
        )
        if env_overrides:
            env.update(env_overrides)
        # Compile-only is framework-internal state. Never let a parent shell or caller
        # accidentally turn the subsequent formal run into a second preflight.
        env.pop("DF_COMPILE_ONLY", None)
        key = str(iteration_dir)
        attempt = self._execution_attempts.get(key, 0) + 1
        self._execution_attempts[key] = attempt
        usage_dir = iteration_dir / "sessions" / "pipeline"
        events_path = usage_dir / "pipeline_llm_calls.jsonl"
        summary_path = iteration_dir / "pipeline_llm_token_usage.json"
        combined_path = iteration_dir / "token_usage.json"
        env.update(
            {
                USAGE_PATH_ENV: str(events_path),
                EXECUTION_ID_ENV: uuid.uuid4().hex,
                ITERATION_ENV: iteration_dir.name,
                ATTEMPT_ENV: str(attempt),
            }
        )
        artifacts = {
            "pipeline_llm_calls": str(events_path),
            "pipeline_llm_token_usage": str(summary_path),
            "token_usage": str(combined_path),
            "step_reuse_plan": str(iteration_dir / REUSE_PLAN_FILENAME),
            "step_manifest": str(iteration_dir / MANIFEST_FILENAME),
            "execution_observation": str(
                iteration_dir / "execution_observation.json"
            ),
        }

        def finalize_usage() -> None:
            try:
                write_pipeline_usage_summary(events_path, summary_path)
                write_combined_iteration_usage(iteration_dir)
            except (OSError, ValueError) as exc:
                logger.warning("记录 pipeline LLM token 失败（不影响主流程）：%s", exc)

        compile_env = {**env, "DF_COMPILE_ONLY": "1"}
        logger.info("开始 framework-owned compile 预检：%s", pipeline_path)
        try:
            compile_proc = subprocess.run(
                [self.python_exe, str(pipeline_path)],
                cwd=str(iteration_dir),
                capture_output=True,
                text=True,
                timeout=self.compile_timeout_sec,
                env=compile_env,
            )
        except subprocess.TimeoutExpired as exc:
            finalize_usage()
            artifacts.update(_write_process_logs(
                iteration_dir, exc.stdout, exc.stderr, f"attempt_{attempt:02d}_compile"
            ))
            logger.error("compile 预检超时（>%ds）：%s", self.compile_timeout_sec, pipeline_path)
            return ExecutionResult(
                False,
                traceback=f"CompileTimeoutExpired: compile 预检超过 {self.compile_timeout_sec}s",
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                artifacts=artifacts,
                diagnostics=json.dumps(
                    {"stage": "compile", "status": "timeout"}, ensure_ascii=False
                ),
                failure_stage="compile",
            )

        artifacts.update(_write_process_logs(
            iteration_dir, compile_proc.stdout, compile_proc.stderr,
            f"attempt_{attempt:02d}_compile",
        ))

        if compile_proc.returncode != 0:
            finalize_usage()
            logger.error("compile 预检失败（exit=%d）：%s", compile_proc.returncode, pipeline_path)
            return ExecutionResult(
                False,
                traceback=_compact_failure_text(compile_proc.stderr or f"compile preflight exited with code {compile_proc.returncode}"),
                stdout=compile_proc.stdout,
                stderr=compile_proc.stderr,
                artifacts=artifacts,
                diagnostics=json.dumps(
                    {
                        "stage": "compile",
                        "status": "failed",
                        "returncode": compile_proc.returncode,
                        "stdout_tail": _bounded_text(compile_proc.stdout, 2000),
                        "stderr_tail": _bounded_text(compile_proc.stderr, 4000),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                failure_stage="compile",
            )

        if reuse_plan.fully_reused:
            try:
                manifest_path = write_step_manifest(
                    config,
                    iteration_dir,
                    raw_entry_path=raw_entry_path,
                    pipeline_filename=self.pipeline_filename,
                    plan=reuse_plan,
                    executed_artifacts=[],
                    final_dataset_path=reuse_plan.entry_path,
                )
            except (OSError, ValueError) as exc:
                return ExecutionResult(
                    False,
                    traceback=f"step manifest validation failed: {exc}",
                    artifacts=artifacts,
                    failure_stage="output_validation",
                )
            artifacts["step_manifest"] = str(manifest_path)
            observation_path = _try_write_observation(
                iteration_dir,
                config=config,
                raw_entry_path=raw_entry_path,
                final_dataset_path=reuse_plan.entry_path,
                status="ok",
                failure_stage=None,
                error=None,
                sample_rows_per_step=self.observation_sample_rows,
                max_sample_chars=self.observation_max_sample_chars,
            )
            if observation_path:
                artifacts["execution_observation"] = observation_path
            logger.info("Framework step cache 全前缀命中：%s", reuse_plan.entry_path)
            return ExecutionResult(
                True,
                dataset_path=reuse_plan.entry_path,
                stdout=compile_proc.stdout,
                stderr=compile_proc.stderr,
                artifacts=artifacts,
                observation_path=observation_path,
            )

        logger.info("compile 预检通过，开始正式执行：%s", pipeline_path)
        start = time.time()
        try:
            proc = subprocess.run(
                [self.python_exe, str(pipeline_path)],
                cwd=str(iteration_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            finalize_usage()
            artifacts.update(_write_process_logs(
                iteration_dir, exc.stdout, exc.stderr, f"attempt_{attempt:02d}_execution"
            ))
            diagnostics = _jsonl_diagnostics(iteration_dir, since=start)
            logger.error("流水线执行超时（>%ds）：%s", self.timeout_sec, iteration_dir)
            return ExecutionResult(
                False, traceback=f"TimeoutExpired: 执行超过 {self.timeout_sec}s",
                stdout=exc.stdout or "", stderr=exc.stderr or "", artifacts=artifacts,
                diagnostics=diagnostics,
                failure_stage="execution",
            )

        finalize_usage()
        artifacts.update(_write_process_logs(
            iteration_dir, proc.stdout, proc.stderr, f"attempt_{attempt:02d}_execution"
        ))
        if proc.returncode != 0:
            diagnostics = _jsonl_diagnostics(iteration_dir, since=start)
            logger.error("流水线执行失败（exit=%d）：%s", proc.returncode, iteration_dir)
            if proc.stderr:
                logger.error("stderr:\n%s", proc.stderr[-2000:])
            return ExecutionResult(
                False,
                traceback=_compact_failure_text(proc.stderr or f"non-zero exit code {proc.returncode}"),
                stdout=proc.stdout, stderr=proc.stderr, artifacts=artifacts,
                diagnostics=diagnostics,
                failure_stage="execution",
            )

        dataset_path = _find_fresh_jsonl(iteration_dir, since=start)
        if dataset_path is None:
            diagnostics = _jsonl_diagnostics(iteration_dir, since=start)
            logger.warning("流水线执行成功但未发现有效输出 jsonl：%s", iteration_dir)
            return ExecutionResult(
                False, traceback="执行成功但最终 JSONL 不存在或为空",
                stdout=proc.stdout, stderr=proc.stderr, artifacts=artifacts,
                diagnostics=diagnostics,
                failure_stage="output_validation",
            )
        executed_artifacts = _find_fresh_step_jsonl(iteration_dir, since=start)
        try:
            manifest_path = write_step_manifest(
                config,
                iteration_dir,
                raw_entry_path=raw_entry_path,
                pipeline_filename=self.pipeline_filename,
                plan=reuse_plan,
                executed_artifacts=executed_artifacts,
                final_dataset_path=str(dataset_path),
            )
        except (OSError, ValueError) as exc:
            diagnostics = _jsonl_diagnostics(iteration_dir, since=start)
            return ExecutionResult(
                False,
                traceback=f"step manifest validation failed: {exc}",
                stdout=proc.stdout,
                stderr=proc.stderr,
                artifacts=artifacts,
                diagnostics=diagnostics,
                failure_stage="output_validation",
            )
        artifacts["step_manifest"] = str(manifest_path)
        observation_path = _try_write_observation(
            iteration_dir,
            config=config,
            raw_entry_path=raw_entry_path,
            final_dataset_path=str(dataset_path),
            status="ok",
            failure_stage=None,
            error=None,
            sample_rows_per_step=self.observation_sample_rows,
            max_sample_chars=self.observation_max_sample_chars,
        )
        if observation_path:
            artifacts["execution_observation"] = observation_path
        logger.info("流水线产出数据集：%s", dataset_path)
        return ExecutionResult(
            True, dataset_path=str(dataset_path), stdout=proc.stdout, stderr=proc.stderr,
            artifacts=artifacts, observation_path=observation_path,
        )


def _validate_pipeline_artifact(path: Path) -> list[str]:
    """Validate the mandatory compile contract without importing generated code."""
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"cannot read pipeline.py: {exc}"]
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"syntax error at line {exc.lineno}: {exc.msg}"]

    for module_path in (path, *(path.parent / "operators").glob("*.py")):
        try:
            module_tree = tree if module_path == path else ast.parse(
                module_path.read_text(encoding="utf-8"), filename=str(module_path)
            )
        except (OSError, SyntaxError) as exc:
            return [f"invalid generated operator {module_path.name}: {exc}"]
        for node in ast.walk(module_tree):
            if isinstance(node, ast.Import) and any(
                alias.name == "dataflow" or alias.name.startswith("dataflow.")
                for alias in node.names
            ):
                return [f"{module_path.name} may not import the legacy dataflow package"]
            if isinstance(node, ast.ImportFrom) and node.module and (
                node.module == "dataflow" or node.module.startswith("dataflow.")
            ):
                return [f"{module_path.name} may not import the legacy dataflow package"]

    # The authored shape is compile() -> DF_COMPILE_ONLY guard -> forward(),
    # matching the reference runtime. run_generated_pipeline is the earlier
    # key-based entry point, still accepted so existing artifacts keep running.
    shared_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "run_generated_pipeline"]
    if shared_calls:
        has_main_guard = any(isinstance(node, ast.If) and any(
            isinstance(part, ast.Name) and part.id == "__name__"
            for part in ast.walk(node.test)
        ) for node in ast.walk(tree))
        return [] if has_main_guard else ["run_generated_pipeline requires a __main__ guard"]

    compile_lines: list[int] = []
    forward_lines: list[int] = []
    guard_lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "compile":
                compile_lines.append(node.lineno)
            elif node.func.attr == "forward":
                forward_lines.append(node.lineno)
        elif isinstance(node, ast.If) and _is_compile_only_guard(node):
            guard_lines.append(node.lineno)

    errors: list[str] = []
    if not compile_lines:
        errors.append("missing pipeline.compile() call")
    if not guard_lines:
        errors.append("missing terminating DF_COMPILE_ONLY guard")
    if not forward_lines:
        errors.append("missing pipeline.forward() call")
    if not errors:
        compile_line = min(compile_lines)
        guard_line = min(guard_lines)
        forward_after_guard = [line for line in forward_lines if line > guard_line]
        if not (compile_line < guard_line and forward_after_guard):
            errors.append(
                "main execution order must be pipeline.compile() -> "
                "DF_COMPILE_ONLY guard -> pipeline.forward()"
            )
    return errors


def _is_compile_only_guard(node: ast.If) -> bool:
    mentions_flag = any(
        isinstance(child, ast.Constant) and child.value == "DF_COMPILE_ONLY"
        for child in ast.walk(node.test)
    )
    if not mentions_flag:
        return False
    for statement in ast.walk(ast.Module(body=node.body, type_ignores=[])):
        if isinstance(statement, (ast.Return, ast.Raise)):
            return True
        if isinstance(statement, ast.Call):
            function = statement.func
            if isinstance(function, ast.Name) and function.id == "exit":
                return True
            if isinstance(function, ast.Attribute) and function.attr == "exit":
                return True
    return False


_STEP_RE = re.compile(r"_step(\d+)\.jsonl$", re.IGNORECASE)
_METADATA_JSONL = {"review.jsonl", "agent_sessions.jsonl", "agent_tools.jsonl"}


def _is_pipeline_artifact(path: Path, iteration_dir: Path) -> bool:
    """Exclude framework/session telemetry from candidate dataset discovery."""
    try:
        relative = path.relative_to(iteration_dir)
    except ValueError:
        return False
    return bool(relative.parts) and relative.parts[0] != "sessions" and path.name not in _METADATA_JSONL


def _find_fresh_step_jsonl(iteration_dir: Path, since: float) -> list[Path]:
    fresh: list[tuple[int, Path]] = []
    for path in iteration_dir.rglob("*.jsonl"):
        if not path.is_file() or not _is_pipeline_artifact(path, iteration_dir):
            continue
        try:
            if path.stat().st_mtime < since - 1.0:
                continue
        except OSError:
            continue
        match = _STEP_RE.search(path.name)
        if match:
            fresh.append((int(match.group(1)), path))
    fresh.sort(key=lambda item: item[0])
    return [path for _, path in fresh]


def _find_fresh_jsonl(iteration_dir: Path, since: float) -> Path | None:
    """Return the final JSONL created by this iteration.

    The shared RecordStore names generated stage files by their logical step
    number. Select the highest step, independent of timestamp ordering.

    When no numbered stage file exists, use the latest nonempty custom JSONL.
    """
    fresh = [
        p for p in iteration_dir.rglob("*.jsonl")
        if p.is_file()
        and _is_pipeline_artifact(p, iteration_dir)
        and p.stat().st_mtime >= since - 1.0
    ]
    if not fresh:
        return None

    # 最大 step 就是最终产物；为空时明确失败，不能回退到中间步骤。
    stepped = []
    for p in fresh:
        m = _STEP_RE.search(p.name)
        if m:
            stepped.append((int(m.group(1)), p))
    if stepped:
        final = max(stepped, key=lambda item: item[0])[1]
        return final if _jsonl_nonempty(final) else None

    # 回退：无 step 命名 → 最新的非空；若全为空 → 最新文件（交由下游 Review 判其不合格）
    fresh.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for p in fresh:
        if _jsonl_nonempty(p):
            return p
    return None


def _jsonl_diagnostics(iteration_dir: Path, since: float) -> str:
    """Summarize row counts and first-row schemas for execution repair."""
    rows: list[dict[str, object]] = []
    for path in sorted(iteration_dir.rglob("*.jsonl")):
        if not _is_pipeline_artifact(path, iteration_dir):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_mtime < since - 1.0:
            continue
        line_count = 0
        fields: list[str] = []
        parse_error: str | None = None
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    line_count += 1
                    if not fields:
                        try:
                            value = json.loads(line)
                            if isinstance(value, dict):
                                fields = sorted(str(key) for key in value)
                        except json.JSONDecodeError as exc:
                            parse_error = f"line 1: {exc.msg}"
        except OSError as exc:
            parse_error = str(exc)
        rows.append({
            "path": str(path.relative_to(iteration_dir)),
            "bytes": stat.st_size,
            "rows": line_count,
            "fields": fields,
            "parse_error": parse_error,
        })
    return _bounded_text(json.dumps(rows, ensure_ascii=False, indent=2), MAX_DIAGNOSTICS_CHARS) if rows else "no fresh JSONL artifacts"


def _jsonl_nonempty(path: Path) -> bool:
    """是否至少有一行非空内容。"""
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    return True
    except OSError:
        return False
    return False
