import tempfile
import unittest
from pathlib import Path

from video_rsi.core import RunContext
from video_rsi.operators.events import (
    EventCandidateExtractionOperator,
    SemanticEventProposalOperator,
)
from video_rsi.operators.semantic_refinement import LLMSemanticEventRefinementOperator
from video_rsi.operators.task_mining import TaskOpportunityMiningOperator
from video_rsi.storage import RunStore
from video_rsi.pipelines import build_bootstrap_pipeline


class EventTests(unittest.TestCase):
    def context(self, root):
        return RunContext("test", RunStore(Path(root), "test"))

    def test_cross_minute_event_and_task_opportunities(self):
        observations = []
        for index in range(3):
            observations.append(
                {
                    "segment_index": index,
                    "start_sec": index * 60.0,
                    "end_sec": (index + 1) * 60.0,
                    "duration_sec": 60.0,
                    "caption": "[00:00-01:00] The same man walks toward a door.",
                    "entity_updates": [
                        {
                            "entity_id": "person_01",
                            "evidence_intervals": [{"start_sec": 0, "end_sec": 60}],
                        }
                    ],
                    "observed_events": [
                        {
                            "start_sec": 50,
                            "end_sec": 60,
                            "participants": ["person_01"],
                            "description": "The man walks toward the doorway.",
                            "visible_preconditions": "He is away from the door.",
                            "visible_outcome": "He reaches the door.",
                            "uncertainty": "",
                        }
                    ],
                }
            )
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self.context(tmp)
            atomic = EventCandidateExtractionOperator().run(
                {"observations": observations}, ctx
            )["atomic_evidence"]
            events = SemanticEventProposalOperator(max_event_sec=70).run(
                {"atomic_evidence": atomic}, ctx
            )["semantic_event_proposals"]
            opportunities = TaskOpportunityMiningOperator().run(
                {"semantic_events": events}, ctx
            )["task_opportunities"]
        self.assertGreaterEqual(len(atomic), 6)
        self.assertTrue(all("event_id" in event for event in events))
        task_types = {item["task_type"] for item in opportunities}
        self.assertIn("dynamic_spatial_trajectory", task_types)
        self.assertIn("temporal_relation", task_types)

    def test_model_semantic_refinement(self):
        class FakeClient:
            model = "fake"

            def complete_json(self, *, prompt, schema, max_tokens):
                self.prompt = prompt
                return {
                    "semantic_events": [
                        {
                            "start_sec": 0,
                            "end_sec": 10,
                            "event_type": "movement",
                            "summary": "A man walks to a door.",
                            "entity_ids": ["person_01"],
                            "evidence_ids": ["a_0000_caption_01"],
                            "visible_precondition": "The man is away from the door.",
                            "visible_outcome": "The man reaches the door.",
                            "uncertainty": "",
                        }
                    ]
                }

        evidence = {
            "atomic_id": "a_0000_caption_01",
            "source_kind": "caption_span",
            "segment_index": 0,
            "start_sec": 0,
            "end_sec": 10,
            "description": "A man walks to a door.",
            "participants": ["person_01"],
            "visible_preconditions": "",
            "visible_outcome": "",
            "uncertainty": "",
            "timestamp_quality": "anchored",
        }
        proposal = {
            "start_sec": 0,
            "end_sec": 10,
            "atomic_evidence_ids": [evidence["atomic_id"]],
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = LLMSemanticEventRefinementOperator(FakeClient()).run(
                {
                    "atomic_evidence": [evidence],
                    "semantic_event_proposals": [proposal],
                },
                self.context(tmp),
            )["semantic_events"]
        self.assertEqual("se_0001", output[0]["event_id"])
        self.assertEqual("model", output[0]["refinement"])

    def test_model_semantic_refinement_rejects_unsupported_time(self):
        class FakeClient:
            model = "fake"

            def complete_json(self, *, prompt, schema, max_tokens):
                return {
                    "semantic_events": [
                        {
                            "start_sec": 0,
                            "end_sec": 20,
                            "event_type": "movement",
                            "summary": "A man walks to a door.",
                            "entity_ids": ["person_01"],
                            "evidence_ids": ["a_0000_caption_01"],
                            "visible_precondition": "",
                            "visible_outcome": "",
                            "uncertainty": "",
                        }
                    ]
                }

        evidence = {
            "atomic_id": "a_0000_caption_01",
            "source_kind": "caption_span",
            "segment_index": 0,
            "start_sec": 0,
            "end_sec": 10,
            "description": "A man walks to a door.",
            "participants": ["person_01"],
            "visible_preconditions": "",
            "visible_outcome": "",
            "uncertainty": "",
            "timestamp_quality": "anchored",
        }
        proposal = {
            "start_sec": 0,
            "end_sec": 30,
            "atomic_evidence_ids": [evidence["atomic_id"]],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "not supported"):
                LLMSemanticEventRefinementOperator(FakeClient()).run(
                    {
                        "atomic_evidence": [evidence],
                        "semantic_event_proposals": [proposal],
                    },
                    self.context(tmp),
                )

    def test_bootstrap_pipeline_end_to_end_and_resume(self):
        observations = [
            {
                "segment_index": 0,
                "start_sec": 0.0,
                "end_sec": 60.0,
                "duration_sec": 60.0,
                "caption": "[00:00-01:00] A woman walks across a room.",
                "entity_updates": [
                    {
                        "entity_id": "person_01",
                        "evidence_intervals": [{"start_sec": 0, "end_sec": 60}],
                    }
                ],
                "observed_events": [],
            }
        ]
        state = {
            "video_key": "video_1",
            "video": {"video_id": "video_1", "duration_sec": 60.0},
            "source_result": "caption/result.json",
            "observations": observations,
        }
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self.context(tmp)
            pipeline = build_bootstrap_pipeline()
            first = pipeline.run_record(state, ctx)
            second = pipeline.run_record(state, ctx)
        self.assertEqual("completed", first["status"])
        self.assertEqual(first["semantic_events"], second["semantic_events"])
        self.assertTrue(all(item["resumed"] for item in second["provenance"]))


if __name__ == "__main__":
    unittest.main()
