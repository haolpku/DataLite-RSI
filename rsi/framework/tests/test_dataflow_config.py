"""Private configuration checks retained from the former source adapters."""

from __future__ import annotations

import yaml
import pytest

from rsi.framework.evolution.runner import FRAMEWORK_ROOT, _missing_required_env
from rsi.framework.evolution.selection import SelectionPolicy
from rsi.framework.evolution.utils.config import load_config


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


def test_selection_policy_file_is_relative_to_yaml_and_fingerprinted(tmp_path):
    policy_file = tmp_path / "policy.py"
    policy_file.write_text(
        "def select(context): return {'iteration': None, 'reason': 'Wait.'}\n",
        encoding="utf-8",
    )
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "selection_policy:\n  kind: python\n  policy_file: policy.py\n",
        encoding="utf-8",
    )
    raw = load_config(config_file).to_dict()["selection_policy"]
    assert raw["policy_file"] == str(policy_file.resolve())
    first = SelectionPolicy.from_config(raw, base_dir=tmp_path)
    policy_file.write_text(
        "def select(context): return {'iteration': None, 'reason': 'Changed.'}\n",
        encoding="utf-8",
    )
    second = SelectionPolicy.from_config(raw, base_dir=tmp_path)
    assert first.identity()["source_sha256"] != second.identity()["source_sha256"]


@pytest.mark.parametrize("raw", [
    {"kind": "python"},
    {"kind": "unknown"},
    {"kind": "criteria", "policy_file": "unused.py"},
    {"kind": "python", "policy_file": "missing.py"},
    {"kind": "criteria", "timeout_sec": 0},
])
def test_invalid_selection_configuration_fails_early(tmp_path, raw):
    with pytest.raises(ValueError):
        SelectionPolicy.from_config(raw, base_dir=tmp_path)
