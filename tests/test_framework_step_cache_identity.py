"""Step-cache identity under the authored compile/forward pipeline shape.

Prefix reuse is only safe if an operator's fingerprint tracks what changes its
behavior and ignores what does not. The reference-aligned shape puts operator
wiring inside a pipeline class, so a naive "everything outside the operator
class is shared source" rule would make every operator non-reusable whenever
any sibling is renamed. These tests pin both directions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsi.framework.evolution.execution.step_cache import operator_descriptors
from rsi.framework.evolution.models import PipelineConfig


PIPELINE_TEMPLATE = '''
import os
from pathlib import Path

from rsi.framework import FileStorage, OperatorABC, PipelineABC
from rsi.framework.evolution.execution.step_cache import FrameworkStepCache

ITERATION_DIR = Path(__file__).resolve().parent
CACHE_DIR = str(ITERATION_DIR / "cache")
ENTRY_PATH = "/fixed/raw.jsonl"
OPERATOR_NAMES = ["CopyRows", "{final_class}"]
HELPER_LIMIT = {helper_limit}


class CopyRows(OperatorABC):
    def __init__(self, keep: str = "{keep}"):
        super().__init__()
        self.keep = keep

    def run(self, storage, **kwargs):
        storage.write(storage.read("dict"))


class {final_class}(OperatorABC):
    def __init__(self):
        super().__init__()

    def run(self, storage):
        storage.write(storage.read("dict"))


class Pipeline(PipelineABC):
    def __init__(self):
        super().__init__()
        self.framework_cache = FrameworkStepCache.from_env(
            raw_entry_path=ENTRY_PATH, operator_names=OPERATOR_NAMES,
        )
        self.storage = FileStorage(
            first_entry_file_name=self.framework_cache.entry_path,
            cache_path=CACHE_DIR,
            file_name_prefix="pipeline_step",
            cache_type="{cache_type}",
        )
        self.copy_rows = CopyRows()
        self.finish_rows = {final_class}()

    def forward(self) -> None:
        if self.framework_cache.should_run(0):
            self.copy_rows.run(storage=self.storage.step(){run_arg})
        if self.framework_cache.should_run(1):
            self.finish_rows.run(storage=self.storage.step())


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.compile()
    if os.getenv("DF_COMPILE_ONLY") == "1":
        raise SystemExit(0)
    pipeline.forward()
'''

DEFAULTS = {
    "final_class": "FinishRows",
    "helper_limit": "10",
    "keep": "a",
    "cache_type": "jsonl",
    "run_arg": "",
}


def _first_operator_descriptor(tmp_path: Path, tag: str, **overrides) -> dict:
    settings = {**DEFAULTS, **overrides}
    iteration = tmp_path / tag
    iteration.mkdir(parents=True)
    (iteration / "pipeline.py").write_text(
        PIPELINE_TEMPLATE.format(**settings), encoding="utf-8"
    )
    config = PipelineConfig(
        [{"name": "CopyRows"}, {"name": settings["final_class"]}],
        "raw -> step_1 -> step_2",
        "",
        "fixture",
    )
    return operator_descriptors(config, iteration, "pipeline.py")[0]


def _fingerprint(tmp_path: Path, tag: str, **overrides) -> str:
    return _first_operator_descriptor(tmp_path, tag, **overrides)["fingerprint"]


def test_first_operator_is_cacheable_and_stable(tmp_path):
    descriptor = _first_operator_descriptor(tmp_path, "base")
    assert descriptor["cacheable"] is True
    assert descriptor["operator_name"] == "CopyRows"
    assert descriptor["logical_index"] == 0
    assert _fingerprint(tmp_path, "again") == descriptor["fingerprint"]


def test_renaming_a_later_operator_keeps_the_prefix_reusable(tmp_path):
    """The whole point of prefix reuse: changing step 2 must not rerun step 1."""
    assert _fingerprint(tmp_path, "renamed", final_class="FinishRowsV2") == _fingerprint(
        tmp_path, "base"
    )


@pytest.mark.parametrize(
    "tag,overrides,reason",
    [
        ("body", {"keep": "b"}, "operator's own constructor default changed"),
        ("helper", {"helper_limit": "20"}, "shared module-level helper changed"),
        ("storage", {"cache_type": "csv"}, "storage settings handed to it changed"),
        ("wiring", {"run_arg": ", mode='strict'"}, "its own run() argument changed"),
    ],
)
def test_behavior_changes_invalidate_the_prefix(tmp_path, tag, overrides, reason):
    assert _fingerprint(tmp_path, tag, **overrides) != _fingerprint(tmp_path, "base"), reason


def test_renaming_a_sibling_operator_module_invalidates_the_prefix(tmp_path):
    """Observed on a real run: renaming a later operator's *module* blocks reuse.

    The `shared` component spans every file under `operators/`, so adding,
    removing or renaming a sibling module changes it for all operators -- even
    one whose own class, params, constructor call and wiring are untouched.

    This is stricter than the reference runtime, which fingerprints only
    operator name, `decision.json` params and constructor AST and would have
    reused the prefix here. The extra strictness is safe (it recomputes rather
    than reusing a stale artifact) but it does cost reuse whenever the agent
    reorganizes files, which coding agents do routinely. Recorded here so the
    behavior is a decision rather than a surprise; see
    docs/dataflow-serving-parity.md.
    """
    def fingerprint(tag: str, sibling_module: str) -> str:
        iteration = tmp_path / tag
        (iteration / "operators").mkdir(parents=True)
        (iteration / "pipeline.py").write_text(
            PIPELINE_TEMPLATE.format(**DEFAULTS), encoding="utf-8"
        )
        # The first operator's own module is byte-identical in both variants.
        (iteration / "operators" / "copy_rows.py").write_text(
            "FIRST_OPERATOR_HELPER = 1\n", encoding="utf-8"
        )
        (iteration / "operators" / sibling_module).write_text(
            "SECOND_OPERATOR_HELPER = 2\n", encoding="utf-8"
        )
        config = PipelineConfig(
            [{"name": "CopyRows"}, {"name": "FinishRows"}], "raw -> s1 -> s2", "", "fix"
        )
        return operator_descriptors(config, iteration, "pipeline.py")[0]["fingerprint"]

    assert fingerprint("before", "finish_rows.py") != fingerprint(
        "after", "finish_rows_v2.py"
    )


def test_unparseable_operator_module_is_not_cacheable(tmp_path):
    iteration = tmp_path / "broken"
    (iteration / "operators").mkdir(parents=True)
    (iteration / "pipeline.py").write_text(
        PIPELINE_TEMPLATE.format(**DEFAULTS), encoding="utf-8"
    )
    (iteration / "operators" / "broken.py").write_text("def (", encoding="utf-8")
    config = PipelineConfig([{"name": "CopyRows"}], "raw -> step_1", "", "fixture")
    descriptor = operator_descriptors(config, iteration, "pipeline.py")[0]
    assert descriptor["cacheable"] is False


def test_duplicate_class_definition_is_not_cacheable(tmp_path):
    iteration = tmp_path / "duplicate"
    (iteration / "operators").mkdir(parents=True)
    (iteration / "pipeline.py").write_text(
        PIPELINE_TEMPLATE.format(**DEFAULTS), encoding="utf-8"
    )
    (iteration / "operators" / "copy_rows.py").write_text(
        "from rsi.framework import OperatorABC\n\n"
        "class CopyRows(OperatorABC):\n"
        "    def run(self, storage):\n        pass\n",
        encoding="utf-8",
    )
    config = PipelineConfig([{"name": "CopyRows"}], "raw -> step_1", "", "fixture")
    descriptor = operator_descriptors(config, iteration, "pipeline.py")[0]
    assert descriptor["cacheable"] is False


def test_decision_params_participate_in_identity(tmp_path):
    iteration = tmp_path / "params"
    iteration.mkdir()
    (iteration / "pipeline.py").write_text(
        PIPELINE_TEMPLATE.format(**DEFAULTS), encoding="utf-8"
    )

    def fingerprint(params):
        config = PipelineConfig(
            [{"name": "CopyRows", "params": params}], "raw -> step_1", "", "fixture"
        )
        return operator_descriptors(config, iteration, "pipeline.py")[0]["fingerprint"]

    assert fingerprint({"semantic_revision": 1}) != fingerprint({"semantic_revision": 2})
