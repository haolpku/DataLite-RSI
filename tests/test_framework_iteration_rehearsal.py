"""Offline rehearsal of the full iteration strategy.

The server run exercises the same loop with real coding agents; this covers the
paths that are expensive or non-deterministic to trigger there, using the
shipped ``default.yaml`` config and the real ``rsi.framework.run`` entry:

* two accepted iterations with verified prefix reuse on the second,
* a failing first attempt repaired by ``self_correct`` within the repair budget,
* a repair budget that bounds retries and never scores a failing candidate.

The coding agent is stubbed so no API call is made; everything else -- config
loading, skill mirroring, artifact validation, compile preflight, subprocess
execution, output discovery, step manifest, observation, provenance -- is the
production path.

Diagnostic-agent isolation is **not** covered here: ``default.yaml`` declares no
``pipeline_diagnostic`` block and ``runner.py`` does not forward a diagnostic
through ``resources``, so a test at this level could only assert a hardcoded
literal. The real property -- that the diagnostic's score is stripped before it
reaches the next prompt -- is asserted in
``test_framework_evolution_loop.py::test_pipeline_diagnosis_is_next_iteration_feedback_not_selection_input``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from rsi.framework import CandidateFeedback, run
from rsi.framework.evolution.models import PipelineConfig


#: Shipped with the package, so this suite needs no example config on disk.
CONFIG_REF = "default.yaml"


def _pipeline_source(entry: Path, *, operator: str, crash: bool) -> str:
    """The authored compile/forward shape, optionally crashing at runtime."""
    body = (
        '        raise ValueError("injected operator failure")\n'
        if crash
        else (
            '        rows = storage.read("dict")\n'
            '        storage.write([\n'
            '            {"instruction": str(r.get("instruction") or r.get("question") or ""),\n'
            '             "output": str(r.get("output") or ""),\n'
            '             "source": "gsm8k"}\n'
            '            for r in rows if str(r.get("output") or "").strip()\n'
            '        ])\n'
        )
    )
    return f'''
import os
from pathlib import Path

from rsi.framework import FileStorage, OperatorABC, PipelineABC
from rsi.framework.evolution.execution.step_cache import FrameworkStepCache

ITERATION_DIR = Path(__file__).resolve().parent
CACHE_DIR = str(ITERATION_DIR / "cache")
ENTRY_PATH = {str(entry)!r}
OPERATOR_NAMES = ["{operator}"]


class {operator}(OperatorABC):
    def __init__(self, minimum_length: int = 1):
        super().__init__()
        self.minimum_length = minimum_length

    def run(self, storage):
{body}

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
            cache_type="jsonl",
        )
        self.normalize = {operator}()

    def forward(self):
        if self.framework_cache.should_run(0):
            self.normalize.run(storage=self.storage.step())


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.compile()
    if os.getenv("DF_COMPILE_ONLY") == "1":
        raise SystemExit(0)
    pipeline.forward()
'''


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    entry = tmp_path / "gsm8k_small.jsonl"
    entry.write_text(
        "".join(
            json.dumps(
                {
                    "question": f"Problem {i}?",
                    "answer": f"{i}",
                    "instruction": f"Problem {i}?",
                    "output": f"Work it out. \\boxed{{{i}}}",
                }
            )
            + "\n"
            for i in range(8)
        ),
        encoding="utf-8",
    )
    return entry


@pytest.fixture(autouse=True)
def offline_env(monkeypatch):
    """Satisfy the config's required env vars and keep the run offline."""
    for name in ("DF_API_KEY", "DF_AGENT_API_KEY", "DF_PIPELINE_API_KEY"):
        monkeypatch.setenv(name, "offline-placeholder")
    # The config's python_exe points at the server interpreter.
    monkeypatch.setenv("DF_AGENT_PYTHON", sys.executable)
    monkeypatch.setenv("DF_DIAGNOSTIC_ENABLED", "false")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def _task(entry: Path, workspace: Path, task_id: str) -> dict:
    return {
        "schema_version": "0.1",
        "task_id": task_id,
        "method_id": "dataflow-evolver",
        "objective": "Normalize the fixed corpus into instruction/output records.",
        "data": {"input_path": str(entry)},
        "output": {
            "required_keys": ["evolution_result"],
            "artifact_types": ["structured_records"],
        },
        "metadata": {"method_config_ref": CONFIG_REF},
        "workspace": str(workspace),
        "provider": "codex",
        "role": "pipeline_builder",
    }


class _Agent:
    """Writes the authored pipeline shape; never calls a model."""

    def __init__(self, entry: Path, *, crash_first: bool = False):
        self.entry = entry
        self.crash_first = crash_first
        self.generated = 0
        self.repairs = 0
        self.prompts: list[str] = []

    def _write(self, iteration_dir: Path, operator: str, crash: bool) -> PipelineConfig:
        iteration_dir.mkdir(parents=True, exist_ok=True)
        source = _pipeline_source(self.entry, operator=operator, crash=crash)
        (iteration_dir / "pipeline.py").write_text(source, encoding="utf-8")
        (iteration_dir / "decision.json").write_text(
            json.dumps(
                {
                    "ops": [{"name": operator, "purpose": "normalize", "params": {"semantic_revision": 1}}],
                    "field_flow": "question/answer -> instruction/output",
                    "reason": "offline rehearsal",
                }
            ),
            encoding="utf-8",
        )
        return PipelineConfig(
            [{"name": operator, "params": {"semantic_revision": 1}}],
            "question/answer -> instruction/output",
            source,
            "offline rehearsal",
        )

    def generate_initial(self, *, iteration_dir, **kwargs):
        self.generated += 1
        return self._write(iteration_dir, "NormalizeRecords", self.crash_first)

    def generate(self, *, iteration_dir, **kwargs):
        self.generated += 1
        return self._write(iteration_dir, "NormalizeRecords", False)

    def self_correct(self, iteration_dir, traceback_text, diagnostics, **kwargs):
        self.repairs += 1
        # The repair request must carry the failure evidence.
        self.prompts.append(
            json.dumps(
                {"traceback": traceback_text, "diagnostics": diagnostics, **kwargs},
                default=str,
            )
        )
        return self._write(iteration_dir, "NormalizeRecords", False)


class _Evaluator:
    def __init__(self):
        self.calls = 0
        self.seen: list[int] = []

    def review(self, dataset_path, task, **kwargs):
        self.calls += 1
        rows = [
            json.loads(line)
            for line in Path(dataset_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert rows, "candidate dataset must not be empty"
        assert all(r.get("instruction") and r.get("output") for r in rows)
        self.seen.append(len(rows))
        return CandidateFeedback(score=0.5 + 0.1 * self.calls, passed=True)


def test_two_iterations_accept_and_reuse_verified_prefix(tmp_path, corpus, monkeypatch):
    monkeypatch.setenv("DF_MAX_ITERATIONS", "2")
    workspace = tmp_path / "ws"
    agent, evaluator = _Agent(corpus), _Evaluator()

    manifest = run(
        _task(corpus, workspace, "rehearsal-accept"),
        resources={"pipeline_agent": agent, "candidate_evaluator": evaluator},
        run_id="rehearsal",
    )

    assert manifest["status"] == "completed", manifest.get("failure")
    assert agent.generated == 2 and agent.repairs == 0
    assert evaluator.calls == 2
    assert manifest["acceptance"]["accepted_iterations"] == [0, 1]
    assert manifest["acceptance"]["diagnostic_used_for_score"] is False

    root = workspace / "runs" / "rehearsal"
    final = root / "records" / "dataflow" / "final_dataset.jsonl"
    rows = [json.loads(line) for line in final.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 8
    assert sorted(rows[0]) == ["instruction", "output", "source"]

    evolution = root / "native" / "runs" / "evolution"
    first = json.loads((evolution / "iteration_001" / "step_reuse_plan.json").read_text(encoding="utf-8"))
    second = json.loads((evolution / "iteration_002" / "step_reuse_plan.json").read_text(encoding="utf-8"))
    assert first["prefix_count"] == 0
    # Identical operator identity across iterations must be reused, not recomputed.
    assert second["prefix_count"] == 1
    manifest_two = json.loads((evolution / "iteration_002" / "step_manifest.json").read_text(encoding="utf-8"))
    assert [step["reused"] for step in manifest_two["steps"]] == [True]
    assert not list((evolution / "iteration_002" / "cache").glob("*.jsonl"))

    observation = json.loads((root / "execution_observation.json").read_text(encoding="utf-8"))
    assert observation["status"] == "completed"
    assert (root / "provenance" / "events.jsonl").is_file()


def test_failed_execution_is_repaired_within_budget(tmp_path, corpus, monkeypatch):
    monkeypatch.setenv("DF_MAX_ITERATIONS", "1")
    monkeypatch.setenv("DF_EXECUTION_REPAIRS", "1")
    workspace = tmp_path / "ws"
    agent, evaluator = _Agent(corpus, crash_first=True), _Evaluator()

    manifest = run(
        _task(corpus, workspace, "rehearsal-repair"),
        resources={"pipeline_agent": agent, "candidate_evaluator": evaluator},
        run_id="rehearsal-repair",
    )

    assert manifest["status"] == "completed", manifest.get("failure")
    # The first attempt crashed; exactly one repair recovered it.
    assert agent.repairs == 1
    assert evaluator.calls == 1
    assert manifest["acceptance"]["accepted_iterations"] == [0]
    repair_prompt = agent.prompts[0]
    assert "injected operator failure" in repair_prompt
    assert "execution" in repair_prompt


def test_repair_budget_is_bounded(tmp_path, corpus, monkeypatch):
    """A pipeline that keeps failing must stop at the budget, not loop."""
    monkeypatch.setenv("DF_MAX_ITERATIONS", "1")
    monkeypatch.setenv("DF_EXECUTION_REPAIRS", "1")
    workspace = tmp_path / "ws"

    class AlwaysCrashes(_Agent):
        def self_correct(self, iteration_dir, traceback_text, diagnostics, **kwargs):
            self.repairs += 1
            return self._write(iteration_dir, "NormalizeRecords", True)

    agent, evaluator = AlwaysCrashes(corpus, crash_first=True), _Evaluator()
    manifest = run(
        _task(corpus, workspace, "rehearsal-budget"),
        resources={"pipeline_agent": agent, "candidate_evaluator": evaluator},
        run_id="rehearsal-budget",
    )

    assert agent.repairs == 1, "repair budget must bound the retries"
    assert evaluator.calls == 0, "a failing candidate is never scored"
    assert manifest["status"] in {"completed", "failed"}
    if manifest["status"] == "completed":
        assert manifest["acceptance"]["accepted_iterations"] == []
