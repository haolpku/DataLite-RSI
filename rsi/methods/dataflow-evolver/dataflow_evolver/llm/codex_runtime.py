"""Codex JSON CLI runtime for PipelineAgent."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from dataflow_evolver.llm.agent_credentials import (
    backend_config,
    backend_value,
    resolve_agent_endpoint,
)
from dataflow_evolver.llm.agent_sdk import (
    AgentRunResult,
    _RESUME_NUDGE,
    _agent_env,
    _is_retryable,
    _record_tool_uses,
    _truncate_arg,
    default_system_prompt_append,
)
from dataflow_evolver.llm.session_state import (
    finalize_agent_session,
    prepare_agent_session_store,
    record_codex_turn_usage,
)
from dataflow_evolver.llm.skill_isolation import isolated_agent_cwd
from dataflow_evolver.utils.config import Config
from dataflow_evolver.utils.logging import get_logger

logger = get_logger("llm.codex_runtime")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True, exclude_none=True, mode="json")
    if hasattr(value, "value"):
        return _jsonable(value.value)
    if hasattr(value, "__dict__"):
        return {
            str(key): _jsonable(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def _pick(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def normalize_codex_usage(value: Any) -> dict[str, int] | None:
    """Normalize the SDK's per-turn ``usage.last`` shape without double counting."""
    raw = _jsonable(value)
    if not isinstance(raw, dict):
        return None
    selected = _pick(raw, "last", "last_usage", "lastUsage")
    if not isinstance(selected, dict):
        selected = _pick(raw, "total", "total_usage", "totalUsage")
    if not isinstance(selected, dict):
        selected = raw
    values = {
        "input_tokens": int(_pick(selected, "input_tokens", "inputTokens") or 0),
        "cached_input_tokens": int(
            _pick(selected, "cached_input_tokens", "cachedInputTokens") or 0
        ),
        "output_tokens": int(_pick(selected, "output_tokens", "outputTokens") or 0),
        "reasoning_tokens": int(
            _pick(
                selected,
                "reasoning_tokens",
                "reasoningTokens",
                "reasoning_output_tokens",
                "reasoningOutputTokens",
            )
            or 0
        ),
        "total_tokens": int(_pick(selected, "total_tokens", "totalTokens") or 0),
    }
    if not values["total_tokens"]:
        values["total_tokens"] = (
            values["input_tokens"] + values["output_tokens"]
        )
    return values if any(values.values()) else None


def _item_payload(item: Any) -> dict[str, Any]:
    payload = _jsonable(item)
    return payload if isinstance(payload, dict) else {"value": payload}


def _tool_trace(items: list[Any]) -> list[dict[str, Any]]:
    uses: list[dict[str, Any]] = []
    for item in items:
        payload = _item_payload(item)
        kind = str(payload.get("type") or payload.get("kind") or "").lower()
        name = ""
        if "mcp" in kind:
            server = str(_pick(payload, "server", "server_name", "serverName") or "mcp")
            tool = str(_pick(payload, "tool", "tool_name", "toolName") or "tool")
            name = f"{server}_{tool}"
        elif "command" in kind:
            name = "shell_command"
        elif "filechange" in kind or "file_change" in kind:
            name = "apply_patch"
        elif "tool" in kind:
            name = str(_pick(payload, "name", "tool", "tool_name", "toolName") or kind)
        if name:
            uses.append({"turn": 1, "name": name, "input": _truncate_arg(payload)})
    return uses


def _codex_provider_overrides(
    agent_cfg: Config,
) -> tuple[list[str], dict[str, str], str | None]:
    """Resolve Codex CLI provider overrides and its process environment."""
    cfg = backend_config(agent_cfg, "codex")
    endpoint = resolve_agent_endpoint(agent_cfg, "codex", provider="openai")
    env = os.environ.copy()
    env.update(_agent_env(agent_cfg, "codex"))
    if endpoint.api_key:
        env["CODEX_API_KEY"] = endpoint.api_key
        env["OPENAI_API_KEY"] = endpoint.api_key

    provider = str(cfg.get("model_provider", "") or "").strip() or None
    if endpoint.base_url and not provider:
        provider = "dataflow-openai-compatible"
    overrides: list[str] = []
    if provider and endpoint.base_url:
        provider_key = f"model_providers.{provider}"
        overrides.extend(
            [
                f"{provider_key}.name={json.dumps(provider)}",
                f"{provider_key}.base_url={json.dumps(endpoint.base_url)}",
                f'{provider_key}.wire_api="responses"',
                f"{provider_key}.supports_websockets=false",
                f'{provider_key}.env_key="CODEX_API_KEY"',
                f"model_provider={json.dumps(provider)}",
            ]
        )
    return overrides, env, provider


def result_from_codex_cli(
    stdout: str,
    stderr: str,
    *,
    exit_code: int,
    timed_out: bool = False,
) -> AgentRunResult:
    """Convert ``codex exec --json`` events into the common agent result."""
    session_id: str | None = None
    assistant_texts: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    usage: dict[str, int] | None = None
    errors: list[str] = []
    completed = False

    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type") or "")
        if event_type == "thread.started":
            session_id = str(event.get("thread_id") or "") or session_id
        elif event_type == "turn.completed":
            completed = True
            usage = normalize_codex_usage(event.get("usage"))
        elif event_type in {"turn.failed", "error"}:
            payload = event.get("error")
            if isinstance(payload, dict):
                message = str(_pick(payload, "message", "details") or payload)
            else:
                message = str(payload or event.get("message") or "Codex CLI error")
            errors.append(message)
        elif event_type in {"item.started", "item.completed"}:
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            kind = str(item.get("type") or "").lower()
            if event_type == "item.completed" and kind == "agent_message":
                text = str(item.get("text") or "").strip()
                if text:
                    assistant_texts.append(text)
            if event_type == "item.completed":
                tool_uses.extend(_tool_trace([item]))

    result_text = assistant_texts[-1] if assistant_texts else ""
    if timed_out:
        error = errors[-1] if errors else "Codex CLI run timed out"
        error_type = "Timeout"
    elif errors or exit_code != 0 or not completed:
        error = errors[-1] if errors else stderr.strip() or "Codex CLI did not complete the turn"
        error_type = "CodexCLIError"
    else:
        error = None
        error_type = None
    return AgentRunResult(
        success=completed and exit_code == 0 and not errors and not timed_out,
        result_text=result_text,
        assistant_texts=assistant_texts,
        error=error,
        error_type=error_type,
        exit_code=exit_code,
        session_id=session_id,
        tool_uses=tool_uses,
        usage=usage,
    )


def _terminate_cli(process: subprocess.Popen[str]) -> tuple[str, str]:
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        return process.communicate(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        return process.communicate()


def _run_once_cli(
    prompt: str,
    *,
    cwd: str,
    agent_cfg: Config,
    resume: str | None,
    timeout: float,
    session_env: dict[str, str] | None,
    system_prompt_append: str | None = None,
) -> AgentRunResult:
    cfg = backend_config(agent_cfg, "codex")
    read_only = agent_cfg.get("read_only", False) is True
    bypass_sandbox = cfg.get("bypass_sandbox", False) is True and not read_only
    sandbox_mode = (
        "read-only"
        if read_only
        else str(
            cfg.get("sandbox_mode", "workspace-write") or "workspace-write"
        ).strip()
    )
    if sandbox_mode not in {"read-only", "workspace-write", "danger-full-access"}:
        raise ValueError(
            "agent.codex.sandbox_mode must be read-only, workspace-write, or danger-full-access"
        )
    overrides, env, _ = _codex_provider_overrides(agent_cfg)
    if session_env:
        env.update(session_env)
    overrides.append('web_search="disabled"')
    if not bypass_sandbox:
        overrides.extend(
            [
                'approval_policy="never"',
                f'sandbox_mode="{sandbox_mode}"',
            ]
        )
        if sandbox_mode == "workspace-write":
            overrides.append('sandbox_workspace_write.network_access=false')

    binary = str(cfg.get("binary", "") or "codex").strip() or "codex"
    binary_path = Path(binary).expanduser()
    if binary_path.is_absolute():
        env["PATH"] = str(binary_path.parent) + os.pathsep + env.get("PATH", "")
    model = str(backend_value(agent_cfg, "codex", "model", default="gpt-5.4"))
    effort = str(backend_value(agent_cfg, "codex", "effort", default="high") or "high")
    execution_flag = (
        "--dangerously-bypass-approvals-and-sandbox"
        if bypass_sandbox
        else "--full-auto"
    )
    args = [binary, "exec"]
    if resume:
        args.extend(["resume", "--json", "--skip-git-repo-check", execution_flag])
    else:
        args.extend(
            [
                "--json",
                "--skip-git-repo-check",
                execution_flag,
                "--cd",
                cwd,
            ]
        )
    args.extend(["--model", model, "-c", f"model_reasoning_effort={json.dumps(effort)}"])
    for override in overrides:
        args.extend(["-c", override])
    if resume:
        args.append(resume)
    else:
        developer = str(
            system_prompt_append or default_system_prompt_append("codex")
        ).strip()
        if developer:
            prompt = f"{developer}\n\n{prompt}"
    # Codex accepts ``-`` as an explicit request to read the prompt from stdin.
    # Passing a large downstream-feedback prompt as one argv item can exceed
    # Linux MAX_ARG_STRLEN before the Codex process is even started.
    args.append("-")

    try:
        process = subprocess.Popen(
            args,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=os.name != "nt",
        )
    except OSError as exc:
        return AgentRunResult(
            False,
            error=str(exc),
            error_type=type(exc).__name__,
        )
    try:
        stdout, stderr = process.communicate(
            input=prompt,
            timeout=timeout if timeout > 0 else None,
        )
        return result_from_codex_cli(
            stdout,
            stderr,
            exit_code=int(process.returncode or 0),
        )
    except subprocess.TimeoutExpired:
        stdout, stderr = _terminate_cli(process)
        return result_from_codex_cli(
            stdout,
            stderr,
            exit_code=int(process.returncode or -1),
            timed_out=True,
        )


def _run_once(
    prompt: str,
    *,
    cwd: str,
    agent_cfg: Config,
    resume: str | None,
    timeout: float,
    session_env: dict[str, str] | None = None,
    system_prompt_append: str | None = None,
) -> AgentRunResult:
    return _run_once_cli(
        prompt,
        cwd=cwd,
        agent_cfg=agent_cfg,
        resume=resume,
        timeout=timeout,
        session_env=session_env,
        system_prompt_append=system_prompt_append,
    )


def run_codex_agent(
    prompt: str,
    *,
    cwd: str,
    agent_cfg: Config,
    tool_log_dir: str | None = None,
    phase: str = "",
    system_prompt_append: str | None = None,
) -> AgentRunResult:
    """Run Codex with the same retry/resume contract as the other backends."""
    max_retries = max(1, int(agent_cfg.get("max_retries", 4)))
    base_delay = float(agent_cfg.get("retry_base_delay", 10))
    timeout = float(agent_cfg.get("run_timeout_sec", 3600))
    resume_enabled = bool(agent_cfg.get("resume_on_retry", True))
    session_store = prepare_agent_session_store(
        agent_cfg,
        backend="codex",
        cwd=cwd,
        tool_log_dir=tool_log_dir,
    )
    session_env = session_store.environment() if session_store is not None else None
    resume_id: str | None = None
    result = AgentRunResult(False, error="未发起任何 Codex 运行", error_type="NoAttempt")
    for attempt in range(1, max_retries + 1):
        used_resume = bool(resume_enabled and resume_id)
        current_prompt = _RESUME_NUDGE if used_resume else prompt
        logger.info(
            "启动 Codex CLI（尝试 %d/%d，%s）：cwd=%s",
            attempt,
            max_retries,
            f"resume={resume_id}" if used_resume else "全新会话",
            cwd,
        )
        result = _run_once(
            current_prompt,
            cwd=cwd,
            agent_cfg=agent_cfg,
            resume=resume_id if used_resume else None,
            timeout=timeout,
            session_env=session_env,
            system_prompt_append=system_prompt_append,
        )
        label = phase if attempt == 1 else f"{phase}#retry{attempt - 1}"
        if tool_log_dir:
            _record_tool_uses(tool_log_dir, result.tool_uses, phase=label)
        if session_store is not None and result.usage:
            record_codex_turn_usage(
                session_store.root,
                phase=label,
                session_id=result.session_id,
                usage=result.usage,
            )
        if result.session_id:
            resume_id = result.session_id
        if result.success:
            return finalize_agent_session(session_store, phase=phase, result=result)

        result.retryable = result.retryable or _is_retryable(result)
        if used_resume and not result.retryable:
            logger.warning("Codex resume 失败，下一次改用新会话：%s", result.error)
            resume_id = None
            result.retryable = True
        if attempt < max_retries and result.retryable:
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "Codex 第 %d/%d 次失败，%.0fs 后重试：%s",
                attempt,
                max_retries,
                delay,
                result.error,
            )
            time.sleep(delay)
            continue
        logger.error(
            "Codex 运行失败 [%s]（retryable=%s）：%s",
            result.error_type,
            result.retryable,
            result.error,
        )
        return finalize_agent_session(session_store, phase=phase, result=result)
    return finalize_agent_session(session_store, phase=phase, result=result)


class CodexAgentRuntime:
    """AgentRuntime implementation backed by ``codex exec --json``."""

    def __init__(self, system_prompt_append: str, *, disable_project_skills: bool = False) -> None:
        self.system_prompt_append = system_prompt_append
        self.disable_project_skills = disable_project_skills

    def run(
        self,
        prompt: str,
        *,
        cwd: str,
        agent_cfg: Config,
        tool_log_dir: str | None,
        phase: str,
    ) -> AgentRunResult:
        with isolated_agent_cwd(cwd, enabled=self.disable_project_skills) as agent_cwd:
            return run_codex_agent(
                prompt,
                cwd=str(agent_cwd),
                agent_cfg=agent_cfg,
                tool_log_dir=tool_log_dir,
                phase=phase,
                system_prompt_append=self.system_prompt_append,
            )
