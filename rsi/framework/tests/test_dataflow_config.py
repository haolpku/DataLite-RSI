"""Private configuration checks retained from the former source adapters."""

from __future__ import annotations

import yaml

from rsi.framework.evolution.runner import FRAMEWORK_ROOT, _missing_required_env


def test_env_scan_excludes_task_overrides_and_respects_defaults(monkeypatch):
    for name in ("DF_INPUT_PATH", "DF_WORKSPACE_DIR", "DF_RUN_NAME", "DF_REVIEW_MODEL"):
        monkeypatch.delenv(name, raising=False)
    raw = yaml.safe_load(
        (FRAMEWORK_ROOT / "configs" / "math-periodic.yaml").read_text(encoding="utf-8")
    )
    absent = set(_missing_required_env(raw))
    assert "DF_INPUT_PATH" not in absent
    assert "DF_WORKSPACE_DIR" not in absent
    assert "DF_RUN_NAME" not in absent
    assert "DF_REVIEW_MODEL" in absent
    assert _missing_required_env({"llm": {"model": "${RSI_DEFAULTED_MODEL:gpt-4o}"}}) == []
