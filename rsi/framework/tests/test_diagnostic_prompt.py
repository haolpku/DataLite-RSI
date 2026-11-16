"""The diagnostic role uses a dedicated built-in prompt, never project skills."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsi.framework.core.contracts import fingerprint
from rsi.framework.evolution.providers.agent_sdk import (
    AgentRunResult,
    default_diagnostic_system_prompt_append,
    default_system_prompt_append,
)
from rsi.framework.evolution.providers.agent_runtime import build_pipeline_diagnostic_runtime
from rsi.framework.evolution.runner import DataFlowPlugin
from rsi.framework.evolution.utils.config import Config
from rsi.framework.io.storage import StorageBundle


def test_diagnostic_prompt_is_independent_and_contains_read_only_rules():
    for backend in ("codex", "claude", "opencode"):
        runtime = build_pipeline_diagnostic_runtime(Config({"backend": backend}))
        prompt = runtime.system_prompt_append
        assert runtime.disable_project_skills is True
        assert prompt == default_diagnostic_system_prompt_append(backend)
        assert prompt != default_system_prompt_append(backend)
        assert "流水线执行诊断 Agent" in prompt
        assert "不要加载或发现 PipelineAgent" in prompt
        assert "不要给数据质量评分" in prompt
        assert "best-so-far" in prompt


def test_diagnostic_manifest_records_prompt_identity_only_when_enabled(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        "pipeline_diagnostic:\n  enabled: ${DF_DIAGNOSTIC_ENABLED:false}\n"
        "  backend: codex\n  read_only: true\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("DF_DIAGNOSTIC_ENABLED", raising=False)
    disabled = DataFlowPlugin().diagnostic_isolation(None, {"config_path": config})
    assert disabled["enabled"] is False
    assert "prompt_fingerprint" not in disabled

    monkeypatch.setenv("DF_DIAGNOSTIC_ENABLED", "true")
    enabled = DataFlowPlugin().diagnostic_isolation(None, {"config_path": config})
    assert enabled["enabled"] is True
    assert enabled["project_skill_discovery"] is False
    assert enabled["prompt_source"] == "builtin"
    assert enabled["prompt_fingerprint"] == fingerprint(
        default_diagnostic_system_prompt_append("codex")
    )
    assert "skill_ref" not in enabled
    with pytest.raises(ValueError, match="pipeline_builder role"):
        DataFlowPlugin().method_skill_refs(SimpleNamespace(role="pipeline_diagnostic"))


def test_run_id_cannot_claim_a_different_diagnostic_prompt(tmp_path):
    store = StorageBundle(tmp_path, "diagnostic-run")
    store.run.write_manifest({"run_id": "diagnostic-run", "diagnostic_isolation": {
        "enabled": True, "prompt_fingerprint": "first",
    }})
    with pytest.raises(ValueError, match="different run manifest"):
        store.run.write_manifest({"run_id": "diagnostic-run", "diagnostic_isolation": {
            "enabled": True, "prompt_fingerprint": "second",
        }})


@pytest.mark.parametrize(
    "backend,module_name,run_name",
    [
        ("codex", "rsi.framework.evolution.providers.codex_runtime", "run_codex_agent"),
        ("claude", "rsi.framework.evolution.providers.agent_runtime", "run_pipeline_agent"),
        ("opencode", "rsi.framework.evolution.providers.opencode_runtime", "run_opencode_agent"),
    ],
)
def test_diagnostic_dispatch_uses_builtin_prompt_and_isolated_cwd(
    tmp_path, monkeypatch, backend, module_name, run_name
):
    calls = []

    def fake_run(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return AgentRunResult(True, result_text="{}")

    monkeypatch.setattr(importlib.import_module(module_name), run_name, fake_run)
    runtime = build_pipeline_diagnostic_runtime(Config({"backend": backend, "read_only": True}))
    result = runtime.run(
        "diagnose", cwd=str(tmp_path),
        agent_cfg=Config({"backend": backend, "read_only": True}),
        tool_log_dir=None, phase="pipeline_diagnostic",
    )
    assert result.success
    assert calls[0][0] == "diagnose"
    assert calls[0][1]["system_prompt_append"] == default_diagnostic_system_prompt_append(backend)
    assert calls[0][1]["cwd"] != str(tmp_path)
    assert Path(calls[0][1]["cwd"]).name.startswith("dfe-agent-isolated-")
    if backend == "claude":
        assert calls[0][1]["disable_project_skills"] is True
