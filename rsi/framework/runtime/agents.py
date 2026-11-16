"""Provider-neutral agent requests with lazy provider dependencies."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from ..core.contracts import TaskEnvelope
from ..core.pipeline import PipelineSpec
from .skills import SkillRef


ROLES = frozenset(
    {
        "pipeline_builder",
        "pipeline_repairer",
        "pipeline_diagnostic",
        "feedback_analyst",
        "policy_planner",
        "question_generator",
    }
)


class MissingProviderDependency(RuntimeError):
    pass


@dataclass(frozen=True)
class AgentRequest:
    task: TaskEnvelope
    method_id: str
    pipeline_spec: PipelineSpec
    current_candidate: Mapping[str, Any]
    feedback_context: Mapping[str, Any]
    repair_context: str
    workspace: Path
    skill_bundle: tuple[SkillRef, ...]
    provider: str
    role: str
    system_prompt: str
    prompt: str
    diagnostic_isolation: bool = False
    provider_config: Mapping[str, Any] = field(default_factory=dict)
    tool_log_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.method_id != self.task.method_id:
            raise ValueError("AgentRequest method_id differs from TaskEnvelope")
        if self.role not in ROLES:
            raise ValueError(f"unknown agent role {self.role!r}")
        if not self.system_prompt.strip():
            raise ValueError(f"agent role {self.role} requires a dedicated system prompt")
        if any(skill.method_id != self.method_id for skill in self.skill_bundle):
            raise ValueError("cross-method skill loading is forbidden")
        if self.role == "pipeline_diagnostic" and not self.diagnostic_isolation:
            raise ValueError("pipeline_diagnostic requires diagnostic isolation")
        if self.role == "pipeline_diagnostic" and self.skill_bundle:
            raise ValueError("diagnostic agent cannot load project skills")


@dataclass(frozen=True)
class AgentResult:
    success: bool
    text: str
    provider: str
    role: str
    session_id: str | None = None
    error: str | None = None
    tool_log_path: str | None = None


class AgentRuntime(Protocol):
    def run(self, request: AgentRequest) -> AgentResult: ...


def _tool_log(request: AgentRequest, stdout: str, stderr: str) -> str | None:
    if request.tool_log_dir is None:
        return None
    request.tool_log_dir.mkdir(parents=True, exist_ok=True)
    path = request.tool_log_dir / f"{request.provider}-{request.role}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stdout": stdout, "stderr": stderr}, ensure_ascii=False) + "\n")
    return str(path)


def _system_prompt(request: AgentRequest) -> str:
    sections = [request.system_prompt]
    for skill in request.skill_bundle:
        sections.append(
            f"Method-scoped {skill.kind} skill {skill.ref} "
            f"(sha256 {skill.fingerprint}):\n{skill.path.read_text(encoding='utf-8')}"
        )
    return "\n\n".join(sections)


def _cwd(request: AgentRequest):
    if request.diagnostic_isolation:
        return tempfile.TemporaryDirectory(prefix="rsi-diagnostic-")
    return None


class CodexRuntime:
    def run(self, request: AgentRequest) -> AgentResult:
        binary = str(request.provider_config.get("binary") or "codex")
        if shutil.which(binary) is None:
            raise MissingProviderDependency("Codex runtime requires the codex CLI")
        isolated = _cwd(request)
        try:
            cwd = Path(isolated.name) if isolated else request.workspace
            prompt = _system_prompt(request) + "\n\n" + request.prompt
            args = [
                binary, "exec", "--json", "--skip-git-repo-check",
                "--cd", str(cwd),
            ]
            if request.diagnostic_isolation:
                args += ["-c", 'sandbox_mode="read-only"', "-c", 'approval_policy="never"']
            else:
                # Codex 0.160 removed `--full-auto`; set the same policy through
                # config overrides, which both old and new CLIs accept.
                args += [
                    "-c", 'sandbox_mode="workspace-write"',
                    "-c", 'approval_policy="never"',
                ]
            model = request.provider_config.get("model")
            if model:
                args += ["--model", str(model)]
            args.append("-")
            try:
                completed = subprocess.run(
                    args, input=prompt, text=True, capture_output=True,
                    cwd=cwd, timeout=float(request.provider_config.get("timeout_sec", 3600)),
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return AgentResult(False, "", "codex", request.role, error=str(exc))
            log_path = _tool_log(request, completed.stdout, completed.stderr)
            return AgentResult(
                completed.returncode == 0, completed.stdout, "codex", request.role,
                error=completed.stderr if completed.returncode else None,
                tool_log_path=log_path,
            )
        finally:
            if isolated:
                isolated.cleanup()


class OpenCodeRuntime:
    def run(self, request: AgentRequest) -> AgentResult:
        binary = str(request.provider_config.get("binary") or "opencode")
        if shutil.which(binary) is None:
            raise MissingProviderDependency("OpenCode runtime requires the opencode CLI")
        isolated = _cwd(request)
        try:
            cwd = Path(isolated.name) if isolated else request.workspace
            permission: Any = (
                {"read": "allow", "glob": "allow", "grep": "allow", "list": "allow",
                 "bash": "deny", "edit": "deny", "write": "deny", "webfetch": "deny"}
                if request.diagnostic_isolation else "allow"
            )
            model = str(request.provider_config.get("model") or "")
            if not model:
                raise ValueError("OpenCode runtime requires provider_config.model")
            config = {
                "agent": {
                    "rsi-role": {
                        "mode": "primary",
                        "model": model,
                        "prompt": _system_prompt(request),
                        "permission": permission,
                    }
                }
            }
            env = os.environ.copy()
            env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config, ensure_ascii=False)
            try:
                completed = subprocess.run(
                    [binary, "run", "--format", "json", "--agent", "rsi-role",
                     "--model", model, "--dir", str(cwd)],
                    input=request.prompt, text=True, capture_output=True, cwd=cwd,
                    env=env, timeout=float(request.provider_config.get("timeout_sec", 3600)),
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return AgentResult(False, "", "opencode", request.role, error=str(exc))
            log_path = _tool_log(request, completed.stdout, completed.stderr)
            return AgentResult(
                completed.returncode == 0, completed.stdout, "opencode", request.role,
                error=completed.stderr if completed.returncode else None,
                tool_log_path=log_path,
            )
        finally:
            if isolated:
                isolated.cleanup()


class ClaudeRuntime:
    def run(self, request: AgentRequest) -> AgentResult:
        try:
            from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
        except ImportError as exc:
            raise MissingProviderDependency(
                "Claude runtime requires the optional claude-agent-sdk package"
            ) from exc
        isolated = _cwd(request)
        try:
            cwd = Path(isolated.name) if isolated else request.workspace
            options = ClaudeAgentOptions(
                cwd=str(cwd),
                system_prompt={
                    "type": "preset", "preset": "claude_code",
                    "append": _system_prompt(request),
                },
                permission_mode="plan" if request.diagnostic_isolation else "bypassPermissions",
                setting_sources=[],
                skills=[],
                max_turns=int(request.provider_config.get("max_turns", 50)),
            )

            async def collect() -> tuple[str, str | None]:
                result_text = ""
                session_id = None
                async for message in query(prompt=request.prompt, options=options):
                    if isinstance(message, ResultMessage):
                        result_text = str(message.result or "")
                        session_id = getattr(message, "session_id", None)
                return result_text, session_id

            result_text, session_id = asyncio.run(collect())
            log_path = _tool_log(request, result_text, "")
            return AgentResult(True, result_text, "claude", request.role, session_id, tool_log_path=log_path)
        finally:
            if isolated:
                isolated.cleanup()


def build_agent_runtime(provider: str) -> AgentRuntime:
    normalized = provider.strip().lower()
    if normalized == "codex":
        return CodexRuntime()
    if normalized == "claude":
        return ClaudeRuntime()
    if normalized == "opencode":
        return OpenCodeRuntime()
    raise ValueError(f"unknown agent provider {provider!r}")
