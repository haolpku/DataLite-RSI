from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsi.framework.runtime.agents import AgentRequest, MissingProviderDependency, build_agent_runtime
from rsi.framework.core.contracts import TaskEnvelope
from rsi.framework.core.pipeline import (
    Operator, Pipeline, RepairBudget, RunContext, bounded_repair_context, bounded_text,
)
from rsi.framework.runtime.framework import METHOD_REGISTRY, SKILL_REGISTRY
from rsi.framework.runtime.routing import load_task_config
from rsi.framework.io.storage import StorageBundle


ROOT = Path(__file__).resolve().parents[1]
def task(method_id="dataflow-evolver", *, objective="目标", data=None, workspace="outputs/rsi"):
    return {
        "schema_version": "0.1", "task_id": "contract-test", "method_id": method_id,
        "objective": objective, "data": {"x": 1} if data is None else data,
        "output": {"required_keys": ["y"]}, "metadata": {"modality": "image"},
        "workspace": workspace,
    }


class AddOne(Operator):
    name = "add_one"
    input_keys = ("x",)
    output_keys = ("y",)
    calls = 0

    def run(self, state, context):
        self.calls += 1
        return {"y": state["x"] + 1}


def test_method_selection_is_only_explicit_method_id():
    with pytest.raises(ValueError, match="explicit method_id"):
        TaskEnvelope.from_mapping({k: v for k, v in task().items() if k != "method_id"})
    with pytest.raises(ValueError, match="unknown method_id"):
        TaskEnvelope.from_mapping(task("not-a-method"))
    for objective in ("中文 VideoRSI policy IF/VC/VQ", "image edit and video reasoning"):
        item = task(objective=objective)
        item["metadata"]["modality"] = "video"
        item["data"]["profile"] = "policy"
        assert TaskEnvelope.from_mapping(item).method_id == "dataflow-evolver"


def test_json_task_loader_expands_environment_without_changing_method(monkeypatch, tmp_path):
    config = task(objective="DataFlow-Evolver policy")
    config["data"] = {"uri": "${RSI_TEST_URI}"}
    path = tmp_path / "task.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="missing environment variable"):
        load_task_config(path)
    monkeypatch.setenv("RSI_TEST_URI", "video:example")
    loaded = load_task_config(path)
    assert loaded.method_id == "dataflow-evolver"
    assert loaded.data["uri"] == "video:example"


def test_active_registry_selects_only_dataflow_without_provider_sdk():
    assert METHOD_REGISTRY.ids() == ("dataflow-evolver",)
    assert METHOD_REGISTRY.get("dataflow-evolver").method_id == "dataflow-evolver"
    with pytest.raises(ValueError, match="unknown method_id"):
        METHOD_REGISTRY.get("video-rsi")


def test_compile_rejects_missing_input_collision_and_result():
    pipeline = Pipeline("p", [AddOne()], result_keys=("y",))
    with pytest.raises(KeyError, match="missing keys"):
        pipeline.compile(())
    with pytest.raises(KeyError, match="overwrite keys"):
        pipeline.compile(("x", "y"))
    with pytest.raises(KeyError, match="result is missing keys"):
        Pipeline("p", [AddOne()], result_keys=("z",)).compile(("x",))


def test_checkpoint_resume_is_scoped_to_run_and_input_and_observed():
    with tempfile.TemporaryDirectory() as temp:
        envelope = TaskEnvelope.from_mapping(task(workspace=temp))
        op = AddOne()
        pipeline = Pipeline("p", [op], result_keys=("y",))
        first = StorageBundle(temp, "run-a")
        context = RunContext(envelope, "run-a", first)
        assert pipeline.run({"x": 1}, context).output == {"y": 2}
        assert pipeline.run({"x": 1}, context).output == {"y": 2}
        assert op.calls == 1
        observation = json.loads((first.root / "execution_observation.json").read_text(encoding="utf-8"))
        assert observation["stages"][0]["resumed"] is True
        assert first.provenance.read()[0]["operator"] == "add_one"
        pipeline.run({"x": 2}, context)
        assert op.calls == 2
        other = RunContext(envelope, "run-b", StorageBundle(temp, "run-b"))
        pipeline.run({"x": 1}, other)
        assert op.calls == 3


def test_common_stores_cover_records_blob_image_video_diagnostic():
    with tempfile.TemporaryDirectory() as temp:
        store = StorageBundle(temp, "storage")
        store.records.write_json("structured/one", {"value": None})
        store.records.write_jsonl("structured/rows", [{"value": None}, {"value": 2}])
        assert store.records.read_json("structured/one") == {"value": None}
        assert store.records.read_jsonl("structured/rows")[0]["value"] is None
        blob = store.blobs.put_bytes(b"bytes")
        assert store.blobs.get_bytes(blob) == b"bytes"
        image = store.artifacts.put_image("pair/before", b"image", media_type="image/png")
        assert store.blobs.get_bytes(image) == b"image"
        video = store.artifacts.put_video_reference("clip", "video:clip", metadata={"timestamp_sec": 3})
        assert video.kind == "video_reference"
        store.run.write_diagnostic("review", {"status": "ok"})
        assert (store.root / "diagnostics" / "review.json").is_file()
        with pytest.raises(ValueError, match="unsafe storage key"):
            store.records.write_json("../outside", {})


def test_bounded_repair_and_attempt_budget():
    assert len(bounded_text("x" * 1000, 3)) <= 3
    context = bounded_repair_context("execution", "x" * 100000, "y" * 100000, max_chars=300)
    assert len(context) <= 300
    budget = RepairBudget(2)
    assert [budget.claim(), budget.claim(), budget.claim()] == [True, True, False]


def test_diagnostic_role_rejects_project_skills():
    with pytest.raises(ValueError, match="unknown method_id"):
        SKILL_REGISTRY.load("video-rsi", "providers/codex/dataflow-evolver-pipeline/SKILL.md")
    task_envelope = TaskEnvelope.from_mapping(task("dataflow-evolver"))
    spec = Pipeline("p", [AddOne()], result_keys=("y",)).compile(("x",))
    authoring = SKILL_REGISTRY.load("dataflow-evolver", "providers/codex/dataflow-evolver-pipeline/SKILL.md")
    common = dict(
        task=task_envelope, method_id="dataflow-evolver", pipeline_spec=spec,
        current_candidate={}, feedback_context={}, repair_context="", workspace=ROOT,
        provider="codex", role="pipeline_diagnostic", system_prompt="diagnose only",
        prompt="read observation", diagnostic_isolation=True,
    )
    with pytest.raises(ValueError, match="project skills"):
        AgentRequest(skill_bundle=(authoring,), **common)
    assert AgentRequest(skill_bundle=(), **common).role == "pipeline_diagnostic"


def test_missing_provider_cli_has_clear_error(monkeypatch):
    monkeypatch.setattr("rsi.framework.runtime.agents.shutil.which", lambda _: None)
    envelope = TaskEnvelope.from_mapping(task("dataflow-evolver"))
    spec = Pipeline("p", [AddOne()], result_keys=("y",)).compile(("x",))
    request = AgentRequest(
        task=envelope, method_id="dataflow-evolver", pipeline_spec=spec,
        current_candidate={}, feedback_context={}, repair_context="", workspace=ROOT,
        skill_bundle=(), provider="codex", role="pipeline_builder",
        system_prompt="DataFlow pipeline builder", prompt="generate",
    )
    with pytest.raises(MissingProviderDependency, match="codex CLI"):
        build_agent_runtime("codex").run(request)


def test_codex_diagnostic_uses_separate_prompt_and_isolated_cwd(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr("rsi.framework.runtime.agents.shutil.which", lambda _: "codex")
    monkeypatch.setattr("rsi.framework.runtime.agents.subprocess.run", fake_run)
    authoring = SKILL_REGISTRY.load(
        "dataflow-evolver", "providers/codex/dataflow-evolver-pipeline/SKILL.md"
    )
    spec = Pipeline("p", [AddOne()], result_keys=("y",)).compile(("x",))
    runtime = build_agent_runtime("codex")

    for role, skills, system_prompt, isolated in (
        ("pipeline_builder", (authoring,), "authoring role only", False),
        ("pipeline_diagnostic", (), "diagnostic role only", True),
    ):
        raw = task("dataflow-evolver", workspace=str(tmp_path))
        raw.update({"provider": "codex", "role": role})
        request = AgentRequest(
            task=TaskEnvelope.from_mapping(raw), method_id="dataflow-evolver",
            pipeline_spec=spec, current_candidate={}, feedback_context={},
            repair_context="", workspace=tmp_path, skill_bundle=skills,
            provider="codex", role=role, system_prompt=system_prompt,
            prompt="inspect", diagnostic_isolation=isolated,
        )
        assert runtime.run(request).success

    builder_args, builder_call = calls[0]
    diagnostic_args, diagnostic_call = calls[1]
    assert builder_call["cwd"] == tmp_path
    # Codex 0.160 removed `--full-auto`; the policy is set through config
    # overrides, which every supported CLI version accepts.
    assert "--full-auto" not in builder_args
    assert 'sandbox_mode="workspace-write"' in builder_args
    assert 'approval_policy="never"' in builder_args
    assert "authoring role only" in builder_call["input"]
    assert "diagnostic role only" not in builder_call["input"]
    assert diagnostic_call["cwd"] != tmp_path
    assert "--full-auto" not in diagnostic_args
    assert 'sandbox_mode="read-only"' in diagnostic_args
    assert "diagnostic role only" in diagnostic_call["input"]
    assert "authoring role only" not in diagnostic_call["input"]
    assert authoring.path.read_text(encoding="utf-8") not in diagnostic_call["input"]
