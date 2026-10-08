"""Active DataFlow runtime remains importable without the source methods tree."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


FRAMEWORK = Path(__file__).resolve().parents[1] / "rsi" / "framework"


def test_framework_python_imports_and_authoring_skills_do_not_use_source_packages():
    forbidden = ("dataflow", "dataflow_evolver", "rsi.methods")
    for path in FRAMEWORK.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = ()
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules = (node.module or "",)
            assert all(
                not (module == prefix or module.startswith(prefix + "."))
                for module in modules for prefix in forbidden
            ), path
    skills = sorted((FRAMEWORK / "skills" / "providers").rglob("SKILL.md"))
    assert len(skills) == 3, skills
    for path in skills:
        content = path.read_text(encoding="utf-8")
        assert "dataflow_evolver.engine" not in content
        # The authored contract is the reference-aligned compile/forward shape.
        for required in (
            "rsi.framework",
            "OperatorABC",
            "PipelineABC",
            "FileStorage",
            "pipeline.compile()",
            "pipeline.forward()",
            "DF_COMPILE_ONLY",
        ):
            assert required in content, (path, required)
        # Generated code must never be told to import the reference package.
        assert "from dataflow" not in content
        assert "import dataflow" not in content


def test_provider_skills_stay_byte_identical():
    """One contract, mirrored per provider: drift would brief agents differently."""
    providers = FRAMEWORK / "skills" / "providers"
    names = sorted(path.name for path in providers.iterdir() if path.is_dir())
    assert names == ["claude", "codex", "opencode"], names
    for relative in ("SKILL.md", "references/serving-contract.md"):
        variants = {
            name: (providers / name / "dataflow-evolver-pipeline" / relative).read_bytes()
            for name in names
        }
        assert len(set(variants.values())) == 1, (
            f"{relative} differs across providers: {sorted(variants)}"
        )


def test_framework_copy_runs_authored_compile_forward_shape(tmp_path):
    """The shape the skills actually teach must run from a framework-only copy."""
    package = tmp_path / "rsi" / "framework"
    package.parent.mkdir()
    shutil.copytree(FRAMEWORK, package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    entry = tmp_path / "fixed.jsonl"
    entry.write_text(
        '{"instruction":"a","note":null}\n{"instruction":"b","note":"keep"}\n',
        encoding="utf-8",
    )
    iteration = tmp_path / "iteration"
    (iteration / "operators").mkdir(parents=True)
    (iteration / "operators" / "__init__.py").write_text("", encoding="utf-8")
    (iteration / "operators" / "tag_rows.py").write_text(
        '''
from rsi.framework import FileStorage, OperatorABC


class TagRows(OperatorABC):
    def __init__(self, label: str = "tagged"):
        super().__init__()
        self.label = label

    def run(self, storage: FileStorage) -> None:
        rows = storage.read("dict")
        storage.write([{**row, "stage": self.label} for row in rows])
''',
        encoding="utf-8",
    )
    pipeline = iteration / "pipeline.py"
    pipeline.write_text(
        '''
import os
from pathlib import Path

from rsi.framework import FileStorage, PipelineABC
from rsi.framework.evolution.execution.step_cache import FrameworkStepCache

from operators.tag_rows import TagRows

ITERATION_DIR = Path(__file__).resolve().parent
CACHE_DIR = str(ITERATION_DIR / "cache")
ENTRY_PATH = __ENTRY__
OPERATOR_NAMES = ["TagRows"]


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
        self.tag_rows = TagRows()

    def forward(self) -> None:
        if self.framework_cache.should_run(0):
            self.tag_rows.run(storage=self.storage.step())


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.compile()
    if os.getenv("DF_COMPILE_ONLY") == "1":
        raise SystemExit(0)
    pipeline.forward()
'''.replace("__ENTRY__", repr(str(entry))),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)

    # compile preflight must not write any step artifact
    preflight = subprocess.run(
        [sys.executable, str(pipeline)], cwd=iteration,
        env={**env, "DF_COMPILE_ONLY": "1"},
        capture_output=True, text=True, timeout=60,
    )
    assert preflight.returncode == 0, preflight.stderr
    assert not (iteration / "cache").exists()

    completed = subprocess.run(
        [sys.executable, str(pipeline)], cwd=iteration, env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    rows = iteration / "cache" / "pipeline_step_step1.jsonl"
    assert [json.loads(line) for line in rows.read_text(encoding="utf-8").splitlines()] == [
        {"instruction": "a", "note": None, "stage": "tagged"},
        {"instruction": "b", "note": "keep", "stage": "tagged"},
    ]


def test_framework_copy_runs_generated_pipeline_without_methods(tmp_path):
    """Backward compatibility: artifacts authored against the key-based entry.

    New pipelines use the compile/forward shape above; this keeps the earlier
    ``run_generated_pipeline`` adapter working for artifacts already on disk.
    """
    package = tmp_path / "rsi" / "framework"
    package.parent.mkdir()
    shutil.copytree(FRAMEWORK, package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    entry = tmp_path / "fixed.jsonl"
    entry.write_text('{"instruction":"small input"}\n', encoding="utf-8")
    iteration = tmp_path / "iteration"
    iteration.mkdir()
    pipeline = iteration / "pipeline.py"
    pipeline.write_text(
        """
from pathlib import Path
from rsi.framework import Operator
from rsi.framework.evolution.execution.generated_pipeline import (
    iter_jsonl, write_step_records, run_generated_pipeline,
)

class Copy(Operator):
    name = "copy"
    version = "1"
    input_keys = ("source_records",)
    output_keys = ("step_1",)

    def run(self, state, context):
        return {"step_1": write_step_records(context, iter_jsonl(state["source_records"]))}

if __name__ == "__main__":
    run_generated_pipeline(
        [Copy()], raw_entry_path=ENTRY_PATH,
        iteration_dir=Path(__file__).resolve().parent,
    )
""".replace("ENTRY_PATH", repr(str(entry))),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    completed = subprocess.run(
        [sys.executable, str(pipeline)], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    rows = iteration / "cache" / "pipeline_step_step1.jsonl"
    assert [json.loads(line) for line in rows.read_text(encoding="utf-8").splitlines()] == [
        {"instruction": "small input"}
    ]


def test_framework_copy_runs_outer_evolution_without_methods(tmp_path):
    package = tmp_path / "rsi" / "framework"
    package.parent.mkdir()
    shutil.copytree(FRAMEWORK, package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    entry = tmp_path / "fixed.jsonl"
    entry.write_text('{"instruction":"small input"}\n', encoding="utf-8")
    driver = tmp_path / "run_isolated.py"
    driver.write_text(
        """
import json
from pathlib import Path
from rsi.framework import CandidateFeedback, run
from rsi.framework.evolution.models import PipelineConfig

entry = Path(__ENTRY__)
workspace = Path(__WORKSPACE__)
source = '''
import os
from pathlib import Path

from rsi.framework import FileStorage, OperatorABC, PipelineABC
from rsi.framework.evolution.execution.step_cache import FrameworkStepCache

ITERATION_DIR = Path(__file__).resolve().parent
CACHE_DIR = str(ITERATION_DIR / "cache")
ENTRY_PATH = INPUT_PATH
OPERATOR_NAMES = ["Copy"]


class Copy(OperatorABC):
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
            cache_type="jsonl",
        )
        self.copy = Copy()

    def forward(self):
        if self.framework_cache.should_run(0):
            self.copy.run(storage=self.storage.step())


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.compile()
    if os.getenv("DF_COMPILE_ONLY") == "1":
        raise SystemExit(0)
    pipeline.forward()
'''.replace("INPUT_PATH", repr(str(entry)))

class Agent:
    def generate_initial(self, *, iteration_dir, **kwargs):
        iteration_dir.mkdir(parents=True, exist_ok=True)
        (iteration_dir / "pipeline.py").write_text(source, encoding="utf-8")
        return PipelineConfig([{"name": "Copy"}], "raw fields -> step_1", source, "copy")
    def generate(self, **kwargs):
        raise AssertionError("one iteration requested")
    def self_correct(self, **kwargs):
        raise AssertionError("no repair expected")

class Evaluator:
    def review(self, dataset_path, task, **kwargs):
        assert Path(dataset_path).is_file()
        return CandidateFeedback(score=0.6, passed=True)

manifest = run({
    "task_id": "isolated", "method_id": "dataflow-evolver",
    "objective": "Copy the fixed input", "data": {"input_path": str(entry)},
    "output": {"required_keys": ["evolution_result"]},
    "metadata": {"method_config_ref": "default.yaml"},
    "workspace": str(workspace), "provider": "codex", "role": "pipeline_builder",
}, resources={"pipeline_agent": Agent(), "candidate_evaluator": Evaluator()},
   run_id="isolated-run")
assert manifest["status"] == "completed", manifest.get("failure")
assert manifest["acceptance"]["accepted_iterations"] == [0]
assert (workspace / "runs" / "isolated-run" / "records" / "dataflow" /
        "final_dataset.jsonl").is_file()
""".replace("__ENTRY__", repr(str(entry))).replace("__WORKSPACE__", repr(str(tmp_path / "workspace"))),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    env["DF_MAX_ITERATIONS"] = "1"
    completed = subprocess.run(
        [sys.executable, str(driver)], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=45,
    )
    assert completed.returncode == 0, completed.stderr


def test_active_config_and_skills_resolve_inside_framework(tmp_path, monkeypatch):
    from rsi.framework.core.contracts import TaskEnvelope
    from rsi.framework.runtime.framework import METHOD_REGISTRY, SKILL_REGISTRY

    entry = tmp_path / "fixed.jsonl"
    entry.write_text('{"instruction":"small input"}\n', encoding="utf-8")
    task = TaskEnvelope.from_mapping({
        "task_id": "location-check", "method_id": "dataflow-evolver",
        "objective": "Test framework locations", "data": {"input_path": str(entry)},
        "output": {"required_keys": ["evolution_result"]},
        "metadata": {"method_config_ref": "default.yaml"},
        "provider": "codex", "role": "pipeline_builder",
    })
    plugin = METHOD_REGISTRY.get(task.method_id)
    config = plugin.load_private_config(task)["config_path"]
    skill = SKILL_REGISTRY.load(task.method_id, plugin.method_skill_refs(task)[0])
    assert config.is_relative_to(FRAMEWORK)
    assert skill.path.is_relative_to(FRAMEWORK)
    first = plugin.build_pipeline(task, {"config_path": config}, {}).compile(task.data.keys())
    monkeypatch.setenv("DF_PIPELINE_MODEL", f"fingerprint-{tmp_path.name}")
    second = plugin.build_pipeline(task, {"config_path": config}, {}).compile(task.data.keys())
    assert first.fingerprint != second.fingerprint
