import unittest
from collections import Counter

from video_rsi.operators.task_candidates import TaskCandidateMiningOperator
from video_rsi.operators.task_mining import TaskOpportunityMiningOperator


class TaskLimitTests(unittest.TestCase):
    """The default policy emits at most one item per task family per video."""

    def _evidence(self):
        return [
            {
                "evidence_id": f"ev_{i}",
                "start_sec": i * 10.0,
                "end_sec": i * 10.0 + 8.0,
                "text": "The person walks toward the door.",
                "entity_ids": ["person_01"],
                "visible_preconditions": "The person is away from the door.",
                "visible_outcome": "The person reaches the door.",
            }
            for i in range(3)
        ]

    def test_candidate_default_is_one_per_task_type(self):
        result = TaskCandidateMiningOperator().run(
            {"evidence_units": self._evidence(), "entity_tracks": []}, None
        )["task_candidates"]
        counts = Counter(item["task_type"] for item in result)
        self.assertTrue(counts)
        self.assertTrue(all(count == 1 for count in counts.values()))

    def test_opportunity_default_is_one_per_task_type(self):
        events = [
            {
                "event_id": f"event_{i}",
                "start_sec": i * 10.0,
                "end_sec": i * 10.0 + 8.0,
                "summary": "The person walks toward the door.",
                "entity_ids": ["person_01"],
                "visible_preconditions": ["The person is away from the door."],
                "visible_outcomes": ["The person reaches the door."],
            }
            for i in range(3)
        ]
        result = TaskOpportunityMiningOperator().run(
            {"semantic_events": events}, None
        )["task_opportunities"]
        counts = Counter(item["task_type"] for item in result)
        self.assertTrue(counts)
        self.assertTrue(all(count == 1 for count in counts.values()))


if __name__ == "__main__":
    unittest.main()
