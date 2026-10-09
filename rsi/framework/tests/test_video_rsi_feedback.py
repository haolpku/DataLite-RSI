from __future__ import annotations

import json

from rsi.framework.evolution.evaluation import VideoRSICandidateEvaluator
from rsi.framework.evolution.models import TaskSpec


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
