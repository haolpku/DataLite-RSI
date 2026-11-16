import tempfile
import unittest
from pathlib import Path

from video_rsi.core import RunContext
from video_rsi.pipelines import build_task_centric_rsi_pipeline
from video_rsi.storage import RunStore


class FakeRewatch:
    name = "fake_rewatch"

    def inspect(self, *, video, candidate, request):
        return {"description": "The requested interval confirms the visible action."}


class FakeGenerator:
    name = "fake_generator"

    def __init__(self):
        self.calls = 0

    def generate(self, *, candidate, evidence, rewatch, prompt):
        self.calls += 1
        return {
            "question": f"What happens in candidate {candidate['candidate_id']}?",
            "answer": "the person reaches the door",
            "choices": [
                "The person reaches the door",
                "The person walks away from the door",
                "The person opens a window",
                "The person sits beside the table",
            ],
            "rationale": "The cited observations show the movement.",
            "evidence_ids": [item["evidence_id"] for item in evidence],
        }


class FakeVerifier:
    name = "fake_verifier"

    def verify(self, *, sample, evidence, rewatch):
        return {
            "valid": True,
            "answer_supported": True,
            "unambiguous": True,
            "video_training_value": True,
            "reason": "grounded",
        }


class FakeEnhancer:
    name = "fake_enhancer"

    def enhance(self, *, sample, evidence, prompt):
        # Replace a deliberately duplicated distractor with a valid hard
        # negative.  The quality gate must see this enhanced output.
        return {
            "choices": [
                sample["answer"],
                "The person walks away from the door",
                "The person opens a window",
                "The person sits beside the table",
            ]
        }


class FakeTextOnly:
    name = "fake_text_only"

    def answer_text(self, *, sample):
        return "cannot determine"


class FakeFrozenTarget:
    name = "fake_frozen_target"

    def answer_video(self, *, sample, video, trial_index):
        if trial_index < 2:
            return sample["answer"]
        return "incorrect"


class FullPipelineTests(unittest.TestCase):
    def test_task_centric_pipeline_reaches_pool_without_semantic_events(self):
        observations = []
        for index in range(3):
            observations.append(
                {
                    "segment_index": index,
                    "start_sec": index * 60.0,
                    "end_sec": (index + 1) * 60.0,
                    "duration_sec": 60.0,
                    "caption": "[00:00-00:20] A man walks toward a door.",
                    "entity_updates": [
                        {
                            "entity_id": "person_01",
                            "appearance": "a man in a blue shirt",
                            "evidence_intervals": [
                                {"start_sec": 0, "end_sec": 20}
                            ],
                        }
                    ],
                    "observed_events": [
                        {
                            "start_sec": 0,
                            "end_sec": 20,
                            "participants": ["person_01"],
                            "description": "The man walks toward and reaches the door.",
                            "visible_preconditions": "The man is away from the door.",
                            "visible_outcome": "The man reaches the door.",
                            "uncertainty": "",
                        }
                    ],
                }
            )
        state = {
            "video_key": "video_1",
            "video": {"video_id": "video_1", "duration_sec": 180.0},
            "source_result": "caption/result.json",
            "observations": observations,
        }
        pipeline = build_task_centric_rsi_pipeline(
            generator_backend=FakeGenerator(),
            verifier_backend=FakeVerifier(),
            text_only_backend=FakeTextOnly(),
            frozen_target_backend=FakeFrozenTarget(),
            max_per_task=2,
        )
        compiled = pipeline.compile(state.keys())
        self.assertNotIn("semantic_events", compiled)
        with tempfile.TemporaryDirectory() as tmp:
            context = RunContext("test", RunStore(Path(tmp), "test"))
            result = pipeline.run_record(state, context)
        self.assertGreater(len(result["pool_candidates"]), 0)
        self.assertGreater(result["feedback_signals"]["accepted_frontier"], 0)
        self.assertTrue(
            all(
                item["target_frontier"]["bucket"] == "frontier"
                for item in result["pool_candidates"]
            )
        )

    def test_internal_variants_keep_one_frontier_sample_per_task(self):
        generator = FakeGenerator()
        state = {
            "video_key": "video_variants",
            "video": {"video_id": "video_variants", "duration_sec": 180.0},
            "source_result": "caption/result.json",
            "observations": [
                {
                    "segment_index": index,
                    "start_sec": index * 60.0,
                    "end_sec": (index + 1) * 60.0,
                    "duration_sec": 60.0,
                    "caption": "[00:00-00:20] A man walks toward a door.",
                    "entity_updates": [{
                        "entity_id": "person_01",
                        "appearance": "a man in a blue shirt",
                        "evidence_intervals": [{"start_sec": 0, "end_sec": 20}],
                    }],
                    "observed_events": [{
                        "start_sec": 0, "end_sec": 20,
                        "participants": ["person_01"],
                        "description": "The man walks toward and reaches the door.",
                        "visible_preconditions": "The man is away from the door.",
                        "visible_outcome": "The man reaches the door.",
                        "uncertainty": "",
                    }],
                }
                for index in range(3)
            ],
        }
        pipeline = build_task_centric_rsi_pipeline(
            generator_backend=generator,
            verifier_backend=FakeVerifier(),
            text_only_backend=FakeTextOnly(),
            frozen_target_backend=FakeFrozenTarget(),
            max_per_task=1,
            candidate_variants=3,
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = pipeline.run_record(
                state, RunContext("test_variants", RunStore(Path(tmp), "test_variants"))
            )
        self.assertGreater(generator.calls, len(result["pool_candidates"]))
        task_types = [item["task_type"] for item in result["pool_candidates"]]
        self.assertEqual(len(task_types), len(set(task_types)))

    def test_quality_gate_consumes_enhanced_samples(self):
        # Keep this test small and isolate the graph contract at compile time.
        from video_rsi.operators.production import DistractorEnhancementOperator, DistractorQualityOperator

        self.assertEqual(("generated_samples",), DistractorEnhancementOperator(FakeEnhancer()).input_keys)
        self.assertEqual(("enhanced_samples",), DistractorQualityOperator.input_keys)

    def test_short_temporal_span_is_metadata_not_a_v0_rejection(self):
        from video_rsi.operators.production import TaskComplexityGateOperator

        candidate = {
            "candidate_id": "short_state",
            "task_type": "generic",
            "hop_count": 1,
            "evidence_window": {"start_sec": 3.0, "end_sec": 4.0},
            "selected_evidence": [{"evidence_id": "e1"}],
        }
        result = TaskComplexityGateOperator(min_temporal_span_sec=20).run(
            {"grounded_candidates": [candidate]}, None
        )
        self.assertEqual(result["complexity_checked_candidates"], [candidate])

    def test_non_frontier_and_text_only_match_are_retained_as_signals(self):
        from video_rsi.operators.production import (
            FrontierCandidateSelectionOperator,
            LocalDedupAndEligibilityOperator,
        )

        sample = {
            "task_type": "state_change",
            "question": "What changed?",
            "answer": "the door opened",
            "choices": ["the door opened", "the door closed", "a person sat", "a light turned on"],
            "text_only": {"correct": True},
            "text_only_label": "text_only_match",
            "target_frontier": {
                "bucket": "too_easy", "correct_count": 4, "trial_count": 4,
            },
            "difficulty_label": "too_easy",
            "selected_evidence": [],
            "evidence_window": {"start_sec": 0, "end_sec": 5},
        }
        selected = FrontierCandidateSelectionOperator().run(
            {"frontier_results": [sample]}, None
        )
        result = LocalDedupAndEligibilityOperator().run(
            {**selected, "distractor_feedback": {}}, None
        )
        self.assertEqual(len(result["pool_candidates"]), 1)
        self.assertEqual(result["pool_candidates"][0]["difficulty_label"], "too_easy")
        self.assertEqual(result["feedback_signals"]["text_only_match_signal"], 1)
        self.assertEqual(result["feedback_signals"]["difficulty_too_easy_signal"], 1)


if __name__ == "__main__":
    unittest.main()
