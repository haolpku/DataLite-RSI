"""Run the DataFlow lifecycle offline with generated shared-runtime operators."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from rsi.framework import run
from rsi.framework.runtime.framework import METHOD_REGISTRY


def _pipeline_source(raw: Path, final_class: str) -> str:
    """The shape the provider skills teach: OperatorABC + compile/forward."""
    return f'''
import os
from pathlib import Path

from rsi.framework import FileStorage, OperatorABC, PipelineABC
from rsi.framework.evolution.execution.step_cache import FrameworkStepCache

ITERATION_DIR = Path(__file__).resolve().parent
CACHE_DIR = str(ITERATION_DIR / "cache")
ENTRY_PATH = {json.dumps(str(raw))}
OPERATOR_NAMES = ["CopyRows", "{final_class}"]


class CopyRows(OperatorABC):
    def __init__(self):
        super().__init__()

    def run(self, storage: FileStorage) -> None:
        storage.write(storage.read("dict"))


class {final_class}(OperatorABC):
    def __init__(self):
        super().__init__()

    def run(self, storage: FileStorage) -> None:
        rows = storage.read("dict")
        storage.write([{{**row, "output": "offline answer"}} for row in rows])


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
        self.copy_rows = CopyRows()
        self.finish_rows = {final_class}()

    def forward(self) -> None:
        if self.framework_cache.should_run(0):
            self.copy_rows.run(storage=self.storage.step())
        if self.framework_cache.should_run(1):
            self.finish_rows.run(storage=self.storage.step())


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.compile()
    if os.getenv("DF_COMPILE_ONLY") == "1":
        raise SystemExit(0)
    pipeline.forward()
'''


def test_dataflow_public_runner_preserves_incumbent_cache_and_diagnostic_boundary(
    tmp_path, monkeypatch
):
    plugin = METHOD_REGISTRY.get("dataflow-evolver")
    module = inspect.getmodule(plugin.__class__)
    module._load_native()
    from rsi.framework.evolution.models import PipelineConfig, ReviewResult, TaskSpec
    from rsi.framework.evolution.execution.execution_wrapper import SubprocessExecutionWrapper
    from rsi.framework.evolution.controller import ReviewOnlyLoop
    from rsi.framework.evolution.utils.config import Config, load_config

    raw = tmp_path / "fixed.jsonl"
    raw.write_text('{"instruction":"Question?","output":"raw answer"}\n', encoding="utf-8")

    class OfflinePipelineAgent:
        def __init__(self):
            self.calls = 0

        def _write(self, iteration_dir, final_class):
            self.calls += 1
            iteration_dir.mkdir(parents=True, exist_ok=True)
            source = _pipeline_source(raw, final_class)
            (iteration_dir / "pipeline.py").write_text(source, encoding="utf-8")
            return PipelineConfig(
                [{"name": "CopyRows"}, {"name": final_class}],
                "source_records -> step_1 -> step_2", source, "offline fixture",
            )

        def generate_initial(self, *, task, iteration_dir, downstream_feedback=None):
            return self._write(iteration_dir, "FinishRows")

        def generate(self, *, task, iteration_dir, **kwargs):
            return self._write(iteration_dir, "FinishRowsV2")

        def self_correct(self, *args, **kwargs):
            raise AssertionError("the smoke pipeline should need no repair")

    class OfflineReviewer:
        def __init__(self):
            self.calls = 0

        def review(self, dataset_path, task, **kwargs):
            self.calls += 1
            rows = [json.loads(line) for line in Path(dataset_path).read_text(
                encoding="utf-8").splitlines()]
            assert len(rows) == 1 and rows[0]["output"] == "offline answer"
            return ReviewResult(
                schema_score=1.0, relevance_score=1.0, passed=True,
                correctness_score=1.0, difficulty_score=0.5,
                review_score=0.8 if self.calls == 1 else 0.7,
            )

    class OfflineDiagnostic:
        def __init__(self):
            self.calls = 0

        def diagnose(self, *, observation_path, output_path, **kwargs):
            self.calls += 1
            assert Path(observation_path).is_file()
            payload = {"status": "diagnosed", "review_score": 999.0}
            Path(output_path).write_text(json.dumps(payload), encoding="utf-8")
            return payload

    agent = OfflinePipelineAgent()
    reviewer = OfflineReviewer()
    diagnostic = OfflineDiagnostic()

    def offline_build_loop(config):
        project = config.to_dict()["project"]
        executor = SubprocessExecutionWrapper(
            project["workspace_dir"], raw_entry_path=project["input_path"]
        )
        return ReviewOnlyLoop(
            pipeline_agent=agent, executor=executor, reviewer=reviewer,
            task=TaskSpec("offline", {"instruction": "str", "output": "str"}, []),
            workspace=Path(project["workspace_dir"]), run_name=project["run_name"],
            max_iterations=2, execution_repairs=1,
            pipeline_diagnostic_agent=diagnostic, pipeline_diagnostic_enabled=True,
        )

    monkeypatch.setattr(module, "_load_native", lambda: (offline_build_loop, Config, load_config))
    task = {
        "schema_version": "0.1", "task_id": "tiny-dataflow",
        "method_id": "dataflow-evolver", "objective": "Improve the fixed raw corpus",
        "data": {"input_path": str(raw)},
        "output": {"required_keys": ["evolution_result"]},
        "metadata": {"method_config_ref": "default.yaml"},
        "workspace": str(tmp_path / "workspace"),
        "provider": "codex", "role": "pipeline_builder",
    }
    manifest = run(task, run_id="tiny-dataflow-run")
    assert manifest["status"] == "completed", manifest.get("failure")
    assert agent.calls == reviewer.calls == diagnostic.calls == 2
    assert manifest["acceptance"]["best_score"] == 0.8
    assert manifest["acceptance"]["accepted_iterations"] == [0]
    assert manifest["acceptance"]["diagnostic_used_for_score"] is False
    root = tmp_path / "workspace" / "runs" / "tiny-dataflow-run"
    assert (root / "records" / "dataflow" / "final_dataset.jsonl").is_file()
    assert (root / "diagnostics" / "dataflow-iteration-001.json").is_file()
    plan = json.loads((root / "native" / "runs" / "evolution" /
                       "iteration_002" / "step_reuse_plan.json").read_text(encoding="utf-8"))
    assert plan["prefix_count"] == 1
    assert (root / "execution_observation.json").is_file()
    assert (root / "provenance" / "events.jsonl").is_file()
    resumed = run(task, run_id="tiny-dataflow-run")
    assert resumed["status"] == "completed", resumed.get("failure")
    assert agent.calls == reviewer.calls == diagnostic.calls == 2
    observation = json.loads((root / "execution_observation.json").read_text(encoding="utf-8"))
    assert observation["stages"][0]["resumed"] is True
