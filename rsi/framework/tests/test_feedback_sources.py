"""Configuration and command protocol of independent named feedback sources."""

import json
import sys

import pytest

from rsi.framework.evolution.feedback_sources import resolve_feedback_sources
from rsi.framework.evolution.controller import build_loop
from rsi.framework.evolution.evaluation.downstream_eval import (
    CommandDownstreamEvaluator, DownstreamEvaluationRequest,
)
from rsi.framework.evolution.models import TaskSpec
from rsi.framework.evolution.prompts import _downstream_feedback_block
from rsi.framework.evolution.utils.config import Config


def test_legacy_shortcut_adds_one_named_command_source():
    sources = resolve_feedback_sources(None, {
        "enabled": True, "command": ["python", "evaluate.py"],
        "benchmarks": ["math500"], "interval": 2,
        "baseline": {"enabled": True},
    })
    assert [(source.name, source.kind) for source in sources] == [
        ("criteria", "criteria"), ("downstream", "command"),
    ]
    assert sources[1].shortcut
    assert sources[1].settings["trains_model"] is True


def test_multiple_named_sources_are_kept_separate():
    sources = resolve_feedback_sources({
        "quality": {"kind": "criteria"},
        "api-judge": {
            "kind": "command", "command": ["python", "api.py"],
            "benchmarks": ["quality"], "interval": 1, "trains_model": False,
        },
        "sft-math": {
            "kind": "command", "command": ["python", "train.py"],
            "benchmarks": ["math500"], "interval": 3, "trains_model": True,
        },
    }, None)
    assert [source.name for source in sources] == ["quality", "api-judge", "sft-math"]
    assert not any(source.shortcut for source in sources)


@pytest.mark.parametrize("sources,shortcut", [
    ({"quality": {"kind": "command", "command": ["x"], "benchmarks": ["x"]}}, None),
    ({"quality": {"kind": "criteria"}, "other": {"kind": "criteria"}}, None),
    ({"criteria": {"kind": "criteria"}, "bad/name": {"kind": "criteria"}}, None),
    ({"criteria": {"kind": "criteria"}}, {"enabled": True, "source_name": "criteria"}),
])
def test_invalid_source_registry_is_rejected(sources, shortcut):
    with pytest.raises(ValueError):
        resolve_feedback_sources(sources, shortcut)


def test_build_loop_combines_named_source_and_downstream_shortcut(tmp_path):
    workspace = tmp_path / "workspace"
    skill = workspace / ".agents/skills/dataflow-evolver-pipeline/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# Test skill\n", encoding="utf-8")
    raw = tmp_path / "raw.jsonl"
    raw.write_text('{}\n', encoding="utf-8")
    config = {
        "project": {"workspace_dir": str(workspace), "input_path": str(raw)},
        "agent": {"backend": "codex"},
        "task": {"target_schema": {"output": "str"}, "quality_criteria": ["complete"]},
        "llm": {"api": {}},
        "review": {},
        "loop": {"max_iterations": 2},
        "feedback_sources": {
            "criteria": {"kind": "criteria"},
            "quality": {
                "kind": "command", "command": ["python", "quality.py"],
                "benchmarks": ["quality"], "interval": 1,
                "trains_model": False, "baseline": True,
            },
        },
        "downstream_eval": {
            "enabled": True, "source_name": "sft",
            "command": ["python", "sft.py"],
            "benchmarks": ["math"], "interval": 2,
        },
    }
    loop = build_loop(
        Config(config), candidate_evaluator=object(), pipeline_agent_override=object(),
    )
    assert loop.review_source_name == "criteria"
    assert [(source.name, source.interval, source.trains_model, source.shortcut)
            for source in loop.command_feedback_sources] == [
        ("quality", 1, False, False), ("sft", 2, True, True),
    ]
    assert loop.command_feedback_sources[0].evaluator.baseline_enabled
    assert loop.downstream_evaluator is loop.command_feedback_sources[1].evaluator

    config["review"] = {"mode": "four_dimensions"}
    with pytest.raises(ValueError, match="four-dimensional review was removed"):
        build_loop(
            Config(config), candidate_evaluator=object(), pipeline_agent_override=object(),
        )


def test_named_command_accepts_score_only_result_but_shortcut_keeps_strict_contract(tmp_path):
    script = tmp_path / "score_only.py"
    script.write_text(
        "import argparse, json\n"
        "from pathlib import Path\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--request', required=True)\n"
        "p.add_argument('--result', required=True)\n"
        "a = p.parse_args()\n"
        "Path(a.result).write_text(json.dumps({'status': 'ok', "
        "'benchmarks': {'quality': {'score': 0.75}}}))\n",
        encoding="utf-8",
    )
    request = DownstreamEvaluationRequest(
        checkpoint_iteration=1, incumbent_iteration=1,
        dataset_path=str(tmp_path / "dataset.jsonl"), pipeline_path=None,
        task=TaskSpec("quality", {"output": "str"}, ["complete"]),
        expected_benchmarks=["quality"], bad_cases_per_benchmark=3,
        training_performed=False, source="quality",
    )
    generic = CommandDownstreamEvaluator(
        [sys.executable, str(script)], expected_benchmarks=["quality"],
        strict_counts=False, use_legacy_baseline_cache=False,
    ).run(request, tmp_path / "generic")
    assert generic.succeeded and generic.source == "quality"
    assert generic.model == "quality"
    assert generic.benchmarks["quality"].incorrect_count is None
    assert json.loads((tmp_path / "generic/request.json").read_text())["source"] == "quality"
    prompt = _downstream_feedback_block(generic)
    assert "反馈源：quality" in prompt and "score=0.750000" in prompt
    assert "incorrect=None" not in prompt

    strict = CommandDownstreamEvaluator(
        [sys.executable, str(script)], expected_benchmarks=["quality"],
    ).run(request, tmp_path / "strict")
    assert strict.status == "failed" and strict.benchmarks == {}

    script.write_text(
        script.read_text(encoding="utf-8").replace("'status': 'ok'", "'status': 'failed'"),
        encoding="utf-8",
    )
    failed = CommandDownstreamEvaluator(
        [sys.executable, str(script)], expected_benchmarks=["quality"],
        strict_counts=False, use_legacy_baseline_cache=False,
    ).run(request, tmp_path / "external_failure")
    assert failed.status == "failed" and failed.benchmarks == {}
