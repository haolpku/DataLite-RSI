"""OpenCode headless runtime for PipelineAgent."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from rsi.framework.evolution.providers.agent_credentials import backend_value, resolve_agent_endpoint
from rsi.framework.evolution.providers.agent_sdk import (
    AgentRunResult,
    _RESUME_NUDGE,
    _agent_env,
    _is_retryable,
    _record_tool_uses,
    _truncate_arg,
    default_system_prompt_append,
)
from rsi.framework.evolution.providers.session_state import (
    finalize_agent_session,
    prepare_agent_session_store,
)
from rsi.framework.evolution.providers.skill_isolation import isolated_agent_cwd
from rsi.framework.evolution.utils.config import Config
from rsi.framework.evolution.utils.logging import get_logger

logger = get_logger("llm.opencode_runtime")

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_CONFIG_SCHEMA = "https://opencode.ai/config.json"


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Config):
        return value.to_dict()
    return dict(value or {})


def _opencode_cfg(agent_cfg: Config) -> dict[str, Any]:
    return _mapping(agent_cfg.get("opencode", {}))


def _model(agent_cfg: Config) -> str:
    cfg = _opencode_cfg(agent_cfg)
    raw = str(
        backend_value(
            agent_cfg,
            "opencode",
            "model",
            default="claude-opus-4-8",
        )
        or ""
    ).strip()
    if not raw:
        raise ValueError("agent.opencode.model 不能为空")
    provider = str(cfg.get("provider", "anthropic") or "anthropic").strip()
    return raw if "/" in raw else f"{provider}/{raw}"


def build_opencode_config(
    agent_cfg: Config,
    *,
    cwd: str,
    system_prompt: str | None = None,
) -> tuple[dict[str, Any], str, str]:
    """Build an invocation-local config without serializing credentials."""
    cfg = _opencode_cfg(agent_cfg)
    model = _model(agent_cfg)
    provider = model.split("/", 1)[0]
    agent_name = "dataflow-pipeline"
    read_only = bool(cfg.get("read_only", agent_cfg.get("read_only", False)))
    raw_permission = cfg.get("permission", "allow")
    if read_only:
        permission: Any = {
            "read": "allow",
            "glob": "allow",
            "grep": "allow",
            "list": "allow",
            "bash": "deny",
            "edit": "deny",
            "write": "deny",
            "webfetch": "deny",
        }
    else:
        permission = str(raw_permission or "allow").lower()
        if permission not in {"allow", "ask", "deny"}:
            raise ValueError("agent.opencode.permission must be allow, ask, or deny")

    endpoint = resolve_agent_endpoint(agent_cfg, "opencode", provider=provider)
    key_env = endpoint.api_key_env
    base_url = endpoint.base_url
    options: dict[str, Any] = {}
    if key_env:
        options["apiKey"] = f"{{env:{key_env}}}"
    if base_url:
        options["baseURL"] = base_url

    selected_prompt = str(
        system_prompt or default_system_prompt_append("opencode")
    )
    config: dict[str, Any] = {
        "$schema": _CONFIG_SCHEMA,
        "agent": {
            agent_name: {
                "description": "Write and repair DataFlow pipelines",
                "mode": "primary",
                "model": model,
                "prompt": selected_prompt,
                "steps": int(cfg.get("max_steps", agent_cfg.get("max_turns", 50)) or 50),
                "permission": permission,
            }
        },
    }
    if options:
        config["provider"] = {provider: {"options": options}}
    return config, agent_name, model


def parse_opencode_events(stdout: str | bytes) -> AgentRunResult:
    """Parse ``opencode run --format json`` NDJSON events."""
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    texts: list[str] = []
    tool_uses: list[dict[str, Any]] = []
    errors: list[str] = []
    session_id: str | None = None
    turn = 0

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        session_id = str(event.get("sessionID") or session_id or "") or None
        kind = event.get("type")
        part = event.get("part") or {}
        if kind == "step_start":
            turn += 1
        elif kind == "text":
            text = str(part.get("text") or "")
            if text:
                texts.append(text)
        elif kind == "tool_use":
            state = part.get("state") or {}
            tool_uses.append(
                {
                    "turn": max(turn, 1),
                    "name": str(part.get("tool") or "unknown"),
                    "input": _truncate_arg(state.get("input")),
                }
            )
        elif kind == "error":
            payload = event.get("error")
            if isinstance(payload, dict):
                data = payload.get("data")
                message = data.get("message") if isinstance(data, dict) else None
                errors.append(
                    str(message or payload.get("message") or payload.get("name") or payload)
                )
            elif payload:
                errors.append(str(payload))

    return AgentRunResult(
        success=not errors,
        result_text="\n".join(texts),
        assistant_texts=texts,
        error="\n".join(errors) or None,
        error_type="OpenCodeError" if errors else None,
        session_id=session_id,
        tool_uses=tool_uses,
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
    cfg = _opencode_cfg(agent_cfg)
    config, agent_name, model = build_opencode_config(
        agent_cfg, cwd=cwd, system_prompt=system_prompt_append
    )
    binary = str(cfg.get("binary", "opencode") or "opencode")
    command = [
        binary,
        "run",
        "--format",
        "json",
        "--agent",
        agent_name,
        "--model",
        model,
        "--dir",
        cwd,
    ]
    variant = str(cfg.get("variant", "") or "").strip()
    if variant:
        command.extend(["--variant", variant])
    if resume:
        command.extend(["--session", resume])

    env = os.environ.copy()
    env.update(_agent_env(agent_cfg, "opencode"))
    if session_env:
        env.update(session_env)
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_PROJECT_ROOT) + (
        os.pathsep + pythonpath if pythonpath else ""
    )
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config, ensure_ascii=False)

    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout if timeout > 0 else None,
            check=False,
        )
    except FileNotFoundError as exc:
        return AgentRunResult(False, error=str(exc), error_type="CLINotFoundError")
    except subprocess.TimeoutExpired as exc:
        result = parse_opencode_events(exc.stdout or "")
        result.success = False
        result.error = f"OpenCode 运行超过 {timeout:.0f}s 超时"
        result.error_type = "Timeout"
        return result
    except OSError as exc:
        return AgentRunResult(False, error=str(exc), error_type=type(exc).__name__)

    result = parse_opencode_events(completed.stdout)
    result.exit_code = completed.returncode
    if completed.returncode != 0:
        result.success = False
        result.error = (
            result.error
            or completed.stderr.strip()
            or completed.stdout.strip()
            or f"OpenCode exited with code {completed.returncode}"
        )
        result.error_type = result.error_type or "OpenCodeProcessError"
    return result


def run_opencode_agent(
    prompt: str,
    *,
    cwd: str,
    agent_cfg: Config,
    tool_log_dir: str | None = None,
    phase: str = "",
    system_prompt_append: str | None = None,
) -> AgentRunResult:
    """Run OpenCode with the Claude adapter's retry and resume semantics."""
    max_retries = max(1, int(agent_cfg.get("max_retries", 4)))
    base_delay = float(agent_cfg.get("retry_base_delay", 10))
    timeout = float(agent_cfg.get("run_timeout_sec", 3600))
    resume_enabled = bool(agent_cfg.get("resume_on_retry", True))
    session_store = prepare_agent_session_store(
        agent_cfg,
        backend="opencode",
        cwd=cwd,
        tool_log_dir=tool_log_dir,
    )
    session_env = session_store.environment() if session_store is not None else None
    resume_id: str | None = None
    result = AgentRunResult(False, error="未发起任何 OpenCode 运行", error_type="NoAttempt")

    for attempt in range(1, max_retries + 1):
        used_resume = bool(resume_enabled and resume_id)
        current_prompt = _RESUME_NUDGE if used_resume else prompt
        logger.info(
            "启动 OpenCode（尝试 %d/%d，%s）：cwd=%s",
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
        if tool_log_dir:
            label = phase if attempt == 1 else f"{phase}#retry{attempt - 1}"
            _record_tool_uses(tool_log_dir, result.tool_uses, phase=label)
        if result.session_id:
            resume_id = result.session_id
        if result.success:
            return finalize_agent_session(
                session_store, phase=phase, result=result
            )

        result.retryable = _is_retryable(result)
        if used_resume and not result.retryable:
            logger.warning("OpenCode resume 失败，下一次改用新会话：%s", result.error)
            resume_id = None
            result.retryable = True
        if attempt < max_retries and result.retryable:
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "OpenCode 第 %d/%d 次失败，%.0fs 后重试：%s",
                attempt,
                max_retries,
                delay,
                result.error,
            )
            time.sleep(delay)
            continue
        logger.error(
            "OpenCode 运行失败 [%s]（retryable=%s）：%s",
            result.error_type,
            result.retryable,
            result.error,
        )
        return finalize_agent_session(session_store, phase=phase, result=result)
    return finalize_agent_session(session_store, phase=phase, result=result)


class OpenCodeAgentRuntime:
    """AgentRuntime implementation backed by ``opencode run``."""

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
            return run_opencode_agent(
                prompt,
                cwd=str(agent_cwd),
                agent_cfg=agent_cfg,
                tool_log_dir=tool_log_dir,
                phase=phase,
                system_prompt_append=self.system_prompt_append,
            )
