from __future__ import annotations

import json

from rsi.framework.evolution.evaluation import VideoRSICandidateEvaluator
from rsi.framework.evolution.controller import Candidate, _is_better
from rsi.framework.evolution.models import TaskSpec
from rsi.framework.evolution.models import ReviewResult


def test_video_rsi_feedback_aggregates_routes_without_hard_difficulty_filter(tmp_path):
    path = tmp_path / "candidate.jsonl"
    rows = [
        {
            "video": {"kind": "video_reference", "uri": "clip.mp4", "media_type": "video/mp4"},
            "question": "What happens after the cup is lifted?",
            "choices": ["It is placed down", "It disappears", "It melts", "Nothing happens"],
            "answer": "It is placed down",
            "selected_evidence": [{"id": "e1", "start_sec": 1, "end_sec": 3}],
            "producer_route": "caption_entity",
            "pool_status": "accepted",
            "text_only_outcome": "match",
            "difficulty_label": "too_easy",
            "duplicate_outcome": "unique",
        },
        {
            "question": "Broken",
            "choices": ["a", "b", "c"],
            "answer": "a",
            "selected_evidence": [],
            "producer_route": "video_only",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    task = TaskSpec("video", {}, [])

    feedback = VideoRSICandidateEvaluator().review(str(path), task)

    assert feedback.passed is True
    assert feedback.domain_feedback["hard_valid_candidates"] == 1
    assert feedback.domain_feedback["accepted_candidates"] == 1
    assert feedback.domain_feedback["accepted_frontier_novel_count"] == 1
    assert feedback.score == 0.5
    assert feedback.domain_feedback["difficulty_labels"]["too_easy"] == 1
    assert feedback.domain_feedback["route_metrics"]["caption_entity"]["accepted"] == 1
    assert feedback.domain_feedback["hard_invalid_reasons"]["choices_not_four"] == 1


def test_video_rsi_feedback_uses_enriched_duplicate_signal_for_frontier_count(tmp_path):
    path = tmp_path / "candidate.jsonl"
    row = {
        "question": "What is picked up?",
        "choices": ["A cup", "A book", "A hat", "A key"],
        "answer": "A cup",
        "selected_evidence": [{"id": "e1", "start_sec": 1, "end_sec": 3}],
        "producer_route": "caption_entity",
        "pool_status": "accepted",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    feedback = VideoRSICandidateEvaluator(
        signal_provider=lambda _: {"duplicate_outcome": "duplicate"}
    ).review(str(path), TaskSpec("video", {}, []))

    assert feedback.domain_feedback["accepted_candidates"] == 1
    assert feedback.domain_feedback["accepted_frontier_novel_count"] == 0
    assert feedback.score == 0.0


def test_video_rsi_feedback_admits_hard_valid_unlabelled_candidate(tmp_path):
    """Absent soft signals must not silently turn a valid probe into zero."""
    path = tmp_path / "candidate.jsonl"
    row = {
        "question": "What is picked up?",
        "choices": ["A cup", "A book", "A hat", "A key"],
        "answer": "A cup",
        "selected_evidence": [{"id": "e1", "start_sec": 1, "end_sec": 3}],
        "producer_route": "caption_entity",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    feedback = VideoRSICandidateEvaluator().review(str(path), TaskSpec("video", {}, []))

    assert feedback.domain_feedback["accepted_candidates"] == 1
    assert feedback.domain_feedback["accepted_frontier_novel_count"] == 1
    assert feedback.domain_feedback["text_only_outcomes"] == {"unknown": 1}
    assert feedback.domain_feedback["difficulty_labels"] == {"unknown": 1}
    assert feedback.domain_feedback["evaluation_model_role"] == "none"
    assert feedback.score == 1.0


def test_injected_remote_judge_defaults_to_reference_role(tmp_path):
    path = tmp_path / "candidate.jsonl"
    row = {
        "question": "What is picked up?",
        "choices": ["A cup", "A book", "A hat", "A key"],
        "answer": "A cup",
        "selected_evidence": [{"id": "e1", "start_sec": 1, "end_sec": 3}],
        "producer_route": "video_only",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    feedback = VideoRSICandidateEvaluator(
        signal_provider=lambda _: {}, model_name="provider/general-vlm"
    ).review(str(path), TaskSpec("video", {}, []))

    assert feedback.domain_feedback["evaluation_model_role"] == "reference"
    assert feedback.domain_feedback["evaluation_model_name"] == "provider/general-vlm"


def test_video_rsi_feedback_respects_explicit_reserve_status(tmp_path):
    path = tmp_path / "candidate.jsonl"
    row = {
        "question": "What is picked up?",
        "choices": ["A cup", "A book", "A hat", "A key"],
        "answer": "A cup",
        "selected_evidence": [{"id": "e1", "start_sec": 1, "end_sec": 3}],
        "producer_route": "caption_entity",
        "pool_status": "reserve",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    feedback = VideoRSICandidateEvaluator().review(str(path), TaskSpec("video", {}, []))

    assert feedback.domain_feedback["accepted_candidates"] == 0
    assert feedback.domain_feedback["route_metrics"]["caption_entity"]["reserve"] == 1
    assert feedback.score == 0.0


def test_video_rsi_feedback_accepts_unique_within_source_label(tmp_path):
    path = tmp_path / "candidate.jsonl"
    row = {
        "question": "What is picked up?",
        "choices": ["A cup", "A book", "A hat", "A key"],
        "answer": "A cup",
        "selected_evidence": [{"id": "e1", "start_sec": 1, "end_sec": 3}],
        "producer_route": "caption_entity",
        "duplicate_outcome": "unique_within_source",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    feedback = VideoRSICandidateEvaluator().review(str(path), TaskSpec("video", {}, []))

    assert feedback.domain_feedback["accepted_frontier_novel_count"] == 1


def test_video_rsi_feedback_prefers_labelled_training_ready_frontier(tmp_path):
    path = tmp_path / "candidate.jsonl"
    common = {
        "choices": ["A cup", "A book", "A hat", "A key"],
        "selected_evidence": [{"id": "e1", "start_sec": 1, "end_sec": 3}],
        "producer_route": "caption_entity",
        "pool_status": "accepted",
        "duplicate_outcome": "unique",
        "visual_grounding_outcome": "grounded",
        "distractor_quality": "strong",
    }
    rows = [
        {**common, "question": "What is picked up?", "answer": "A cup", "training_value_label": "high", "caption_shortcut_risk": "low"},
        {**common, "question": "What is put down?", "answer": "A book", "training_value_label": "low", "caption_shortcut_risk": "high"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    audit_index = 0

    def independent_audit(_row):
        nonlocal audit_index
        current = audit_index
        audit_index += 1
        return {
            "training_value_label": "high" if current == 0 else "low",
            "visual_grounding_outcome": "grounded",
            "caption_shortcut_risk": "low" if current == 0 else "high",
            "distractor_quality": "strong",
        }

    feedback = VideoRSICandidateEvaluator(signal_provider=independent_audit).review(
        str(path), TaskSpec("video", {}, [])
    )

    assert feedback.domain_feedback["accepted_frontier_novel_count"] == 2
    assert feedback.domain_feedback["training_ready_frontier_novel_count"] == 1
    assert feedback.domain_feedback["promotion_frontier_count"] == 1
    assert feedback.score == 0.5


def test_pipeline_quality_claims_are_not_treated_as_independent_audits(tmp_path):
    path = tmp_path / "candidate.jsonl"
    row = {
        "question": "What is picked up?",
        "choices": ["A cup", "A book", "A hat", "A key"],
        "answer": "A cup",
        "selected_evidence": [{"id": "e1", "start_sec": 1, "end_sec": 3}],
        "producer_route": "video_only",
        "training_value_label": "high",
        "visual_grounding_outcome": "grounded",
        "caption_shortcut_risk": "low",
        "distractor_quality": "strong",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    feedback = VideoRSICandidateEvaluator().review(str(path), TaskSpec("video", {}, []))

    assert feedback.domain_feedback["training_ready_frontier_novel_count"] == 0
    assert feedback.domain_feedback["quality_feedback_mode"] == "pipeline_claims_unverified"
    assert feedback.domain_feedback["independent_quality_audit_coverage"] == 0
    assert feedback.score == 0


def test_independent_quality_audit_can_be_required(tmp_path):
    path = tmp_path / "candidate.jsonl"
    row = {
        "question": "What is picked up?",
        "choices": ["A cup", "A book", "A hat", "A key"],
        "answer": "A cup",
        "selected_evidence": [{"id": "e1", "start_sec": 1, "end_sec": 3}],
        "producer_route": "video_only",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    feedback = VideoRSICandidateEvaluator(require_independent_quality=True).review(
        str(path), TaskSpec("video", {}, [])
    )

    assert feedback.passed is False


def test_video_rsi_feedback_reports_pool_balance_and_does_not_promote_unknown_audits(tmp_path):
    path = tmp_path / "candidate.jsonl"
    common = {
        "choices": ["A cup", "A book", "A hat", "A key"],
        "selected_evidence": [{"id": "e1", "start_sec": 1, "end_sec": 3}],
        "producer_route": "caption_entity",
        "pool_status": "accepted",
        "duplicate_outcome": "unique",
        "training_value_label": "high",
        "visual_grounding_outcome": "unknown",
        "caption_shortcut_risk": "low",
        "distractor_quality": "strong",
        "video": {"uri": "clip.mp4"},
    }
    rows = [
        {**common, "question": f"What is picked up {index}?", "answer": "A cup"}
        for index in range(8)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    feedback = VideoRSICandidateEvaluator().review(str(path), TaskSpec("video", {}, []))

    diagnostics = feedback.domain_feedback["training_pool_diagnostics"]
    assert feedback.domain_feedback["accepted_frontier_novel_count"] == 8
    assert feedback.domain_feedback["training_ready_frontier_novel_count"] == 0
    assert feedback.domain_feedback["promotion_frontier_count"] == 0
    assert diagnostics["answer_position"]["counts"] == {"A": 8, "B": 0, "C": 0, "D": 0}
    assert diagnostics["answer_position"]["severe_skew"] is True
    assert diagnostics["same_video"]["correlated_variant_count"] == 7


def test_video_rsi_can_continue_a_passing_tie(tmp_path):
    config = object()
    review = dict(schema_score=1.0, relevance_score=1.0, review_score=1.0, passed=True)
    incumbent = Candidate(0, tmp_path / "v0", config, review=ReviewResult(**review))
    challenger = Candidate(1, tmp_path / "v1", config, review=ReviewResult(**review))

    assert _is_better(challenger, incumbent) is False
    assert _is_better(challenger, incumbent, accept_equal_score=True) is True
