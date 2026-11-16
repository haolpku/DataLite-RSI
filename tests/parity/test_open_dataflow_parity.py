"""Optional check against the real ``open-dataflow==1.0.10`` package.

This group is skipped unless the reference package is installed, and is
excluded from the default run by the ``parity`` marker. Run it in the isolated
environment:

    conda run -n dfe-open-dataflow-parity python -m pytest tests/parity -m parity

It does two things the offline suite cannot:

1. Re-derives every case from the reference and asserts the recording in
   ``baseline_open_dataflow.json`` still holds. Without this the recording
   could drift into an echo of ``rsi.framework``.
2. Compares the reference and ``rsi.framework`` directly in one process, so a
   difference is observed rather than inferred from a stored file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adapters  # noqa: E402
import cases  # noqa: E402


pytestmark = pytest.mark.parity

BASELINE_PATH = Path(__file__).resolve().parent / "baseline_open_dataflow.json"

dataflow = pytest.importorskip("dataflow", reason="open-dataflow is not installed")


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("reference-parity")
    return cases.evaluate(adapters.reference_api(), root)


@pytest.fixture(scope="module")
def framework(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("framework-parity")
    return cases.evaluate(adapters.framework_api(), root)


@pytest.fixture(scope="module")
def recorded() -> dict:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _diff(group: str, left: dict, right: dict, left_name: str, right_name: str) -> list[str]:
    expected, actual = left[group], right[group]
    assert set(expected) == set(actual), f"case set drifted for {group}"
    return [
        f"\n--- {name}\n  {left_name}: {json.dumps(expected[name], ensure_ascii=False, sort_keys=True)}"
        f"\n  {right_name}: {json.dumps(actual[name], ensure_ascii=False, sort_keys=True)}"
        for name in sorted(expected)
        if expected[name] != actual[name]
    ]


@pytest.mark.parametrize("group", sorted(cases.CASE_GROUPS))
def test_recorded_baseline_still_matches_the_reference(group, reference, recorded):
    mismatches = _diff(group, recorded["results"], reference, "recorded", "reference")
    assert not mismatches, (
        f"the recorded baseline no longer matches open-dataflow "
        f"{dataflow.__version__} for {group}; re-run record_baseline.py if the "
        f"reference environment changed intentionally:" + "".join(mismatches)
    )


@pytest.mark.parametrize("group", sorted(cases.CASE_GROUPS))
def test_framework_matches_the_reference_in_one_process(group, reference, framework):
    mismatches = _diff(group, reference, framework, "open-dataflow", "rsi.framework")
    assert not mismatches, (
        f"{len(mismatches)} {group} cases differ between open-dataflow "
        f"{dataflow.__version__} and rsi.framework:" + "".join(mismatches)
    )


def test_reference_environment_is_the_pinned_version():
    assert dataflow.__version__ == "1.0.10"
