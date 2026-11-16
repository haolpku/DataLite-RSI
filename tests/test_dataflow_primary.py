"""DataFlow is the primary evolution loop even for mixed-modality records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsi.framework import CandidateFeedback, resolve_local_input_artifact, run
from rsi.framework import ArtifactRef, InputContract
from rsi.framework.runtime.framework import METHOD_REGISTRY
from rsi.framework.runtime.routing import load_task_config


def test_mixed_input_contract_validates_references_without_reading_media(tmp_path):
    contract = InputContract.from_mapping({
        "modalities": ["text", "image", "video"],
        "required_fields": ["instruction"],
        "artifact_fields": {"image": "image", "video": "video_reference"},
    })
    record = {
        "instruction": "Describe the change.",
        "image": {"kind": "image", "uri": "assets/before.png", "media_type": "image/png"},
        "video": {"kind": "video_reference", "uri": "assets/clip.mp4", "media_type": "video/mp4"},
    }
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert contract.validate_entry(path) == 1
    with pytest.raises(ValueError, match="must be 'image'"):
        contract.validate_record({**record, "image": record["video"]})
    with pytest.raises(ValueError, match=r"image/\*"):
        ArtifactRef.from_mapping({"kind": "image", "uri": "x", "media_type": "video/mp4"})


def test_local_input_artifact_resolution_is_relative_to_manifest(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    frame = media / "frame.png"
    frame.write_bytes(b"small")
    entry = tmp_path / "fixed.jsonl"
    entry.write_text("{}\n", encoding="utf-8")
    assert resolve_local_input_artifact(entry, {
        "kind": "image", "uri": "media/frame.png", "media_type": "image/png",
    }) == frame
    with pytest.raises(ValueError, match="remote input artifact"):
        resolve_local_input_artifact(entry, {
            "kind": "video_reference", "uri": "https://example.invalid/clip.mp4",
            "media_type": "video/mp4",
        })
    with pytest.raises(ValueError, match="finite number"):
        CandidateFeedback(score=float("nan"), passed=True)


def test_primary_entry_refuses_a_legacy_method_id():
    with pytest.raises(ValueError, match="unknown method_id"):
        run({
            "schema_version": "0.1", "task_id": "other", "method_id": "video-rsi",
            "objective": "Use the DataFlow entry", "data": {},
            "output": {"required_keys": ["pool_candidates"]}, "metadata": {},
        })


def test_mixed_task_template_keeps_dataflow_method_id(tmp_path, monkeypatch):
    entry = tmp_path / "fixed.jsonl"
    entry.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("DF_INPUT_PATH", str(entry))
    template = Path(__file__).resolve().parents[1] / "rsi" / "framework" / "tasks" / "dataflow-mixed.json"
    task = load_task_config(template)
    assert task.method_id == "dataflow-evolver"
    assert task.input_contract.modalities == ("text", "image", "video")
    assert task.metadata["method_config_ref"] == "multimodal.example.yaml"
    assert METHOD_REGISTRY.get(task.method_id).load_private_config(task)["config_path"].is_file()


def test_fixed_corpus_content_changes_pipeline_identity(tmp_path):
    entry = tmp_path / "fixed.jsonl"
    entry.write_text('{"instruction":"first"}\n', encoding="utf-8")
    config = {
        "schema_version": "0.1", "task_id": "identity",
        "method_id": "dataflow-evolver", "objective": "Improve fixed records",
        "data": {"input_path": str(entry)},
        "output": {"required_keys": ["evolution_result"]},
        "metadata": {"method_config_ref": "default.yaml"},
        "provider": "codex", "role": "pipeline_builder",
    }
    task = load_task_config(config)
    plugin = METHOD_REGISTRY.get(task.method_id)
    private = plugin.load_private_config(task)
    before = plugin.build_pipeline(task, private, {}).compile(task.data.keys()).fingerprint
    entry.write_text('{"instruction":"changed"}\n', encoding="utf-8")
    after = plugin.build_pipeline(task, private, {}).compile(task.data.keys()).fingerprint
    assert before != after


def test_task_evaluator_feedback_does_not_claim_text_review_metrics(tmp_path):
    plugin = METHOD_REGISTRY.get("dataflow-evolver")
    __import__(plugin.__class__.__module__, fromlist=["_load_native"])._load_native()
    from rsi.framework.evolution.corpus import CorpusFile, InputCorpus
    from rsi.framework.evolution.models import ReviewResult, TaskSpec
    from rsi.framework.evolution.prompts import build_pipeline_prompt

    path = tmp_path / "fixed.jsonl"
    path.write_text('{"image":{"kind":"image","uri":"frame.png","media_type":"image/png"}}\n', encoding="utf-8")
    corpus_file = CorpusFile(path, path.name, path.stat().st_size, rows=1)
    corpus = InputCorpus(str(path), path.parent, (corpus_file,), corpus_file)
    task = TaskSpec(
        "Improve frame data", {"image": "object"}, [],
        input_contract=InputContract.from_mapping({"modalities": ["image"]}),
        evaluator_kind="task",
    )
    review = ReviewResult(
        schema_score=0.0, relevance_score=0.0, passed=True,
        review_score=0.7, domain_feedback={"visual_consistency": 0.9},
    )
    prompt = build_pipeline_prompt(task, corpus, str(tmp_path / "iteration"), parent_review=review)
    assert "visual_consistency" in prompt
    assert "候选得分：0.700" in prompt
    assert "抽样四维" not in prompt
    assert "全量硬指标" not in prompt


def test_dataflow_mixed_task_runs_two_candidates_and_rehomes_media(tmp_path):
    plugin = METHOD_REGISTRY.get("dataflow-evolver")
    plugin_module = __import__(plugin.__class__.__module__, fromlist=["_load_native"])
    plugin_module._load_native()
    from rsi.framework.evolution.models import PipelineConfig

    image = tmp_path / "tiny.png"
    image.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000b49444154789c636000020000050001a5f645400000000049454e44ae426082"
        )
    )
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"small-video-reference-only")
    raw = tmp_path / "fixed.jsonl"
    raw.write_text(json.dumps({
        "instruction": "Describe both media references.", "output": "seed",
        "image": {"kind": "image", "uri": "tiny.png", "media_type": "image/png"},
        "source_image": {"kind": "image", "uri": "tiny.png", "media_type": "image/png"},
        "video": {"kind": "video_reference", "uri": "clip.mp4", "media_type": "video/mp4"},
    }) + "\n", encoding="utf-8")

    class PipelineAgent:
        def __init__(self):
            self.calls = 0
            self.parent_scores = []

        def _write(self, iteration_dir, quality):
            self.calls += 1
            iteration_dir.mkdir(parents=True, exist_ok=True)
            final_class = f"MediaRowsV{quality}"
            source = f'''
import os
from pathlib import Path

from rsi.framework import FileStorage, OperatorABC, PipelineABC, artifact_store_for
from rsi.framework.evolution.media import resolve_local_input_artifact
from rsi.framework.evolution.execution.step_cache import FrameworkStepCache

ITERATION_DIR = Path(__file__).resolve().parent
CACHE_DIR = str(ITERATION_DIR / "cache")
ENTRY_PATH = {str(raw)!r}
OPERATOR_NAMES = ["CopyRows", "{final_class}"]


class CopyRows(OperatorABC):
    def __init__(self):
        super().__init__()

    def run(self, storage):
        storage.write(storage.read("dict"))


class {final_class}(OperatorABC):
    def __init__(self, revision: str = "{quality}"):
        super().__init__()
        self.revision = revision

    def run(self, storage):
        artifacts = artifact_store_for(ITERATION_DIR)
        output_rows = []
        for row in storage.read("dict"):
            image = artifacts.put_image(
                "edited/image", resolve_local_input_artifact(ENTRY_PATH, row["image"])
            )
            video = artifacts.put_video_reference(
                "source/video", row["video"]["uri"],
                media_type=row["video"]["media_type"],
            )
            output_rows.append({{
                **row, "image": image.to_dict(), "video": video.to_dict(),
                "output": str({quality}),
            }})
        storage.write(output_rows)


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
        self.media_rows = {final_class}()

    def forward(self):
        if self.framework_cache.should_run(0):
            self.copy_rows.run(storage=self.storage.step())
        if self.framework_cache.should_run(1):
            self.media_rows.run(storage=self.storage.step())


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.compile()
    if os.getenv("DF_COMPILE_ONLY") == "1":
        raise SystemExit(0)
    pipeline.forward()
'''
            (iteration_dir / "pipeline.py").write_text(source, encoding="utf-8")
            return PipelineConfig(
                [{"name": "CopyRows"}, {"name": final_class}],
                "raw fields -> step_1 -> step_2", source, "mixed media fixture",
            )

        def generate_initial(self, *, task, iteration_dir, downstream_feedback=None):
            assert task.modalities == ("text", "image", "video")
            return self._write(iteration_dir, 1)

        def generate(self, *, task, iteration_dir, parent_review=None, **kwargs):
            self.parent_scores.append(parent_review.review_score)
            return self._write(iteration_dir, 2)

        def self_correct(self, *args, **kwargs):
            raise AssertionError("this pipeline should not require repair")

    class MediaEvaluator:
        def __init__(self):
            self.calls = 0

        def review(self, dataset_path, task, **kwargs):
            self.calls += 1
            rows = [json.loads(line) for line in Path(dataset_path).read_text(
                encoding="utf-8").splitlines()]
            assert len(rows) == 1
            assert task.modalities == ("text", "image", "video")
            assert rows[0]["image"]["kind"] == "image"
            assert rows[0]["video"]["kind"] == "video_reference"
            quality = int(rows[0]["output"])
            return CandidateFeedback(
                score=quality / 3, passed=True,
                domain_feedback={"media_kinds": ["image", "video_reference"]},
            )

    agent, evaluator = PipelineAgent(), MediaEvaluator()
    task = {
        "schema_version": "0.1", "task_id": "mixed-dataflow",
        "method_id": "dataflow-evolver", "objective": "Improve a mixed media record pipeline",
        "data": {"input_path": str(raw)},
        "input_contract": {
            "modalities": ["text", "image", "video"],
            "required_fields": ["instruction"],
            "artifact_fields": {
                "image": "image", "source_image": "image", "video": "video_reference"
            },
        },
        "output": {
            "required_keys": ["evolution_result"],
            "artifact_types": ["image", "video_reference"],
        },
        "metadata": {"method_config_ref": "default.yaml"},
        "workspace": str(tmp_path / "workspace"), "provider": "codex",
        "role": "pipeline_builder",
    }
    from rsi.framework.evolution.controller import build_loop as real_build_loop
    from rsi.framework.evolution.utils.config import Config
    # Keep the native loop but bound this smoke run to two candidates.
    def two_round_loop(config, **kwargs):
        value = config.to_dict()
        value["loop"]["max_iterations"] = 2
        return real_build_loop(Config(value), **kwargs)
    original = plugin_module._load_native
    plugin_module._load_native = lambda: (two_round_loop, Config, original()[2])
    try:
        manifest = run(task, run_id="mixed-media-run", resources={
            "pipeline_agent": agent, "candidate_evaluator": evaluator,
        })
    finally:
        plugin_module._load_native = original
    assert manifest["status"] == "completed", manifest.get("failure")
    assert agent.calls == evaluator.calls == 2
    assert len(agent.parent_scores) == 1
    assert manifest["acceptance"]["accepted_iterations"] == [0, 1]
    assert manifest["input_contract"]["modalities"] == ["text", "image", "video"]
    assert {"image", "video_reference"} <= set(manifest["artifact_types"])
    assert manifest["feedback"]["review"]["evaluator_kind"] == "task"
    output_root = tmp_path / "workspace" / "runs" / "mixed-media-run"
    row = json.loads((output_root / "records" / "dataflow" / "final_dataset.jsonl")
                     .read_text(encoding="utf-8").strip())
    assert row["image"]["uri"].startswith("blobs/")
    assert (output_root / row["image"]["uri"]).is_file()
    assert row["source_image"]["uri"].startswith("blobs/")
    assert (output_root / row["source_image"]["uri"]).is_file()
    assert row["video"]["uri"] == str(video)
    assert row["video"]["media_type"] == "video/mp4"
    assert (output_root / "records" / "artifacts").is_dir()
    assert (output_root / "provenance" / "events.jsonl").is_file()
