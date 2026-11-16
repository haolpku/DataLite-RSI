"""Provider-neutral PipelineAgent credential and endpoint resolution."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dataflow_evolver.utils.config import Config


@dataclass(frozen=True)
class AgentEndpoint:
    """One backend's resolved secret source and optional compatible base URL."""

    api_key_env: str
    api_key: str
    base_url: str


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Config):
        return value.to_dict()
    return dict(value or {})


def backend_config(agent_cfg: Config, backend: str) -> dict[str, Any]:
    return _mapping(agent_cfg.get(backend, {}))


def backend_value(
    agent_cfg: Config,
    backend: str,
    key: str,
    *,
    default: Any = None,
) -> Any:
    """Resolve backend-specific value, then provider-neutral value, then default."""
    section = backend_config(agent_cfg, backend)
    value = section.get(key)
    if value not in (None, ""):
        return value
    value = agent_cfg.get(key, None)
    return default if value in (None, "") else value


def resolve_agent_endpoint(
    agent_cfg: Config,
    backend: str,
    *,
    provider: str = "",
) -> AgentEndpoint:
    """Resolve generic credentials while preserving legacy provider env fallbacks."""
    key_env = str(
        backend_value(
            agent_cfg,
            backend,
            "api_key_env",
            default="DF_AGENT_API_KEY",
        )
        or "DF_AGENT_API_KEY"
    ).strip()
    api_key = os.environ.get(key_env, "") if key_env else ""

    provider = provider.strip().lower()
    legacy_key_envs: tuple[str, ...]
    if backend == "claude" or provider == "anthropic":
        legacy_key_envs = ("ANTHROPIC_API_KEY",)
    elif backend == "codex" or provider == "openai":
        legacy_key_envs = ("CODEX_API_KEY", "OPENAI_API_KEY")
    else:
        legacy_key_envs = ()
    if not api_key:
        for candidate in legacy_key_envs:
            value = os.environ.get(candidate, "")
            if value:
                key_env, api_key = candidate, value
                break

    base_url = str(
        backend_value(agent_cfg, backend, "base_url", default="") or ""
    ).strip()
    if not base_url:
        legacy_base_envs = (
            ("ANTHROPIC_BASE_URL",)
            if backend == "claude" or provider == "anthropic"
            else (("OPENAI_BASE_URL",) if backend == "codex" or provider == "openai" else ())
        )
        for candidate in legacy_base_envs:
            value = os.environ.get(candidate, "")
            if value:
                base_url = value.strip()
                break

    return AgentEndpoint(api_key_env=key_env, api_key=api_key, base_url=base_url)
