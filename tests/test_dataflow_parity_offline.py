"""Offline check: ``rsi.framework`` against the recorded baseline.

The baseline in ``baseline_open_dataflow.json`` was recorded from the real
``open-dataflow==1.0.10`` package plus the DataFlow-Evolver ``compat`` fixes by
``record_baseline.py``. This module never imports that package, so the active
runtime's alignment is checked in an environment without it — which is the
environment the framework actually runs in.

``test_open_dataflow_parity.py`` re-derives the same cases from the real
package when it is installed, so the recording stays honest rather than
becoming an echo of this implementation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "parity"))

import adapters  # noqa: E402
import cases  # noqa: E402


BASELINE_PATH = Path(__file__).resolve().parent / "parity" / "baseline_open_dataflow.json"


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def observed(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("framework-parity")
    return cases.evaluate(adapters.framework_api(), root)


def _pandas_major() -> str:
    import pandas

    return pandas.__version__.split(".")[0]


def _group_items(group: str, baseline: dict, observed: dict):
    expected = baseline["results"][group]
    actual = observed[group]
    assert set(expected) == set(actual), (
        f"case set drifted for {group}: "
        f"missing={sorted(set(expected) - set(actual))} "
        f"extra={sorted(set(actual) - set(expected))}"
    )
    return expected, actual


def _assert_group(group: str, baseline: dict, observed: dict) -> None:
    expected, actual = _group_items(group, baseline, observed)
    recorded_major = baseline["environment"]["pandas"].split(".")[0]
    if group in cases.PANDAS_SENSITIVE_GROUPS and _pandas_major() != recorded_major:
        pytest.skip(
            f"{group} depends on pandas inference; baseline recorded on pandas "
            f"{baseline['environment']['pandas']}, running on {_pandas_major()}.x"
        )
    mismatches = [
        f"\n--- {name}\n  baseline: {json.dumps(expected[name], ensure_ascii=False, sort_keys=True)}"
        f"\n  observed: {json.dumps(actual[name], ensure_ascii=False, sort_keys=True)}"
        for name in sorted(expected)
        if expected[name] != actual[name]
    ]
    assert not mismatches, (
        f"{len(mismatches)} of {len(expected)} {group} cases differ from the "
        f"open-dataflow baseline:" + "".join(mismatches)
    )


def test_storage_matches_baseline(baseline, observed):
    _assert_group("storage", baseline, observed)


def test_pipeline_compile_forward_matches_baseline(baseline, observed):
    _assert_group("pipeline", baseline, observed)


def test_batched_pipelines_match_baseline(baseline, observed):
    _assert_group("batched", baseline, observed)


def test_serving_matches_baseline(baseline, observed):
    _assert_group("serving", baseline, observed)


def test_baseline_recording_is_from_the_reference_runtime(baseline):
    environment = baseline["environment"]
    assert environment["open_dataflow"] == "1.0.10"
    assert environment["compat_source"].startswith("DataFlow-Evolver")
    assert baseline["results"].keys() == cases.CASE_GROUPS.keys()


def test_active_runtime_does_not_require_open_dataflow():
    assert "dataflow" not in sys.modules
    import rsi.framework  # noqa: F401

    assert "dataflow" not in sys.modules
