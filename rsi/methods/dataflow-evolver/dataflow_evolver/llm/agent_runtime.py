"""Runtime boundary shared by PipelineAgent backends."""
from __future__ import annotations

from typing import Protocol

from dataflow_evolver.llm.agent_sdk import (
    AgentRunResult,
    default_diagnostic_system_prompt_append,
    default_system_prompt_append,
    run_pipeline_agent,
)
from dataflow_evolver.llm.skill_isolation import isolated_agent_cwd
from dataflow_evolver.utils.config import Config


class AgentRuntime(Protocol):
    """Execute one autonomous agent session."""

    def run(
        self,
        prompt: str,
        *,
        cwd: str,
        agent_cfg: Config,
        tool_log_dir: str | None,
        phase: str,
    ) -> AgentRunResult: ...


class ClaudeAgentRuntime:
    """Adapter around the existing Claude Agent SDK implementation."""

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
            return run_pipeline_agent(
                prompt,
                cwd=str(agent_cwd),
                agent_cfg=agent_cfg,
                tool_log_dir=tool_log_dir,
                phase=phase,
                system_prompt_append=self.system_prompt_append,
                disable_project_skills=self.disable_project_skills,
            )


def _build_agent_runtime(
    agent_cfg: Config,
    system_prompt_append: str,
    *,
    disable_project_skills: bool = False,
) -> AgentRuntime:
    backend = str(agent_cfg.get("backend", "claude") or "claude").strip().lower()
    if backend == "claude":
        return ClaudeAgentRuntime(system_prompt_append, disable_project_skills=disable_project_skills)
    if backend == "opencode":
        from dataflow_evolver.llm.opencode_runtime import OpenCodeAgentRuntime

        return OpenCodeAgentRuntime(system_prompt_append, disable_project_skills=disable_project_skills)
    if backend == "codex":
        from dataflow_evolver.llm.codex_runtime import CodexAgentRuntime

        return CodexAgentRuntime(system_prompt_append, disable_project_skills=disable_project_skills)
    raise ValueError(
        f"unknown agent.backend: {backend!r}; expected claude, opencode, or codex"
    )


def build_agent_runtime(agent_cfg: Config) -> AgentRuntime:
    """Build PipelineAgent with its fixed authoring system prompt."""
    backend = str(agent_cfg.get("backend", "claude") or "claude").strip().lower()
    return _build_agent_runtime(agent_cfg, default_system_prompt_append(backend))


def build_pipeline_diagnostic_runtime(agent_cfg: Config) -> AgentRuntime:
    """Build Call B with its fixed read-only diagnostic system prompt."""
    backend = str(agent_cfg.get("backend", "claude") or "claude").strip().lower()
    return _build_agent_runtime(
        agent_cfg, default_diagnostic_system_prompt_append(backend),
        disable_project_skills=True,
    )
