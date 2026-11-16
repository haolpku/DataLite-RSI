import json

import pytest

from rsi.framework.evolution.agents.review_agent import ReviewAgent
from rsi.framework.evolution.models import EmbeddingQualityResult, TaskSpec


class FakeServing:
    def __init__(self, payload):
        self.payload = payload

    def generate_one(self, prompt):
        return json.dumps(self.payload, ensure_ascii=False)


class FakeEmbeddingQualityEvaluator:
    def evaluate(self, dataset_path, *, candidate_user_field, candidate_assistant_field):
        return EmbeddingQualityResult(
            enabled=True,
            status="ok",
            metric="vendi,nearest_neighbor",
            enabled_metrics=["vendi", "nearest_neighbor"],
            mmd=0.0,
            embedding_diversity={
                "cosine_vendi_ratio": 1.0,
                "nearest_neighbor_cosine_p95": -1.0,
            },
        )


def _task():
    return TaskSpec(
        task_description="education dataset",
        target_schema={"messages": "list"},
        quality_criteria=["criterion one", "criterion two"],
    )


def _agent(**kwargs):
    return ReviewAgent(
        FakeServing({
            "criteria_assessments": [
                {"criterion_index": 1, "met": True, "evidence": "a"},
                {"criterion_index": 2, "met": False, "evidence": "b"},
            ],
            "critical_failures": [],
        }),
        mode="criteria",
        validation_enabled=True,
        **kwargs,
    )


def test_criteria_mode_uses_only_rubric_rate_and_forwards_evidence(tmp_path):
    path = tmp_path / "candidate.jsonl"
    path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]})
        + "\n",
        encoding="utf-8",
    )
    agent = _agent()
    result = agent.review(str(path), _task())

    assert result.review_score == 0.5
    assert result.llm_composite == 0.5
    assert result.score_components["embedding_weight"] == 0.0
    assert result.domain_feedback["review_mode"] == "criteria"
    assert result.domain_feedback["criteria_assessments"][0]["evidence"] == "a"
    assert result.passed is False


def test_criteria_mode_blends_embedding_quality_when_enabled(tmp_path):
    path = tmp_path / "candidate.jsonl"
    path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]})
        + "\n",
        encoding="utf-8",
    )
    agent = _agent(
        embedding_quality_evaluator=FakeEmbeddingQualityEvaluator(),
        embedding_review_score_weight=0.4,
    )
    result = agent.review(str(path), _task())

    # Rubric met rate is 0.5; the fake embedding quality is 1.0.
    assert result.llm_composite == 0.5
    assert result.embedding_quality.status == "ok"
    assert result.score_components["embedding_weight"] == 0.4
    assert result.score_components["embedding_quality"] == 1.0
    assert result.review_score == 0.7


def test_criteria_mode_requires_nonempty_quality_criteria(tmp_path):
    path = tmp_path / "candidate.jsonl"
    path.write_text(json.dumps({"messages": []}) + "\n", encoding="utf-8")
    task = TaskSpec("education", {"messages": "list"}, [])
    result = _agent().review(str(path), task)

    assert result.passed is False
    assert "quality_criteria" in result.issues[0]


def test_embedding_evaluator_requires_explicit_blend_weight():
    with pytest.raises(ValueError, match="review_score_weight"):
        _agent(embedding_quality_evaluator=FakeEmbeddingQualityEvaluator())


def test_embedding_blend_weight_must_be_between_zero_and_one():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _agent(
            embedding_quality_evaluator=FakeEmbeddingQualityEvaluator(),
            embedding_review_score_weight=1.1,
        )
