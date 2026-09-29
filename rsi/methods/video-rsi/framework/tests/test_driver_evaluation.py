import json
import tempfile
import unittest
from pathlib import Path

from drivers.evaluation import EvaluationPolicy, compare_evaluations, evaluate_run


class EvaluationTests(unittest.TestCase):
    def test_primary_metric_is_not_a_weighted_score(self):
        result = compare_evaluations(
            {"accepted_frontier_novel_count": 2},
            {"accepted_frontier_novel_count": 3, "hard_gates": {"ok": True}, "eligible_for_promotion": True},
        )
        self.assertEqual(result["decision"], "accept")
        self.assertEqual(result["reason"], "primary_metric_increased")

    def test_failed_gate_rejects_even_when_primary_increases(self):
        result = compare_evaluations(
            {"accepted_frontier_novel_count": 2},
            {"accepted_frontier_novel_count": 3, "hard_gates": {"target_model_frozen": False}, "eligible_for_promotion": False},
        )
        self.assertEqual(result["decision"], "reject")
        self.assertEqual(result["reason"], "hard_gate_failed")

    def test_too_easy_regression_is_a_signal_not_a_hard_gate(self):
        result = compare_evaluations(
            {
                "accepted_frontier_novel_count": 2,
                "diagnostics": {"too_easy_rate": 0.50},
            },
            {
                "accepted_frontier_novel_count": 3,
                "diagnostics": {"too_easy_rate": 0.75},
                "hard_gates": {"target_model_frozen": True},
                "eligible_for_promotion": True,
            },
        )
        self.assertEqual(result["decision"], "accept")
        self.assertEqual(result["reason"], "primary_metric_increased")
        self.assertEqual(result["difficulty_signal"]["candidate_too_easy_rate"], 0.75)
        self.assertNotIn("difficulty_no_regression", result["candidate_hard_gates"])

    def test_run_evaluation_records_lineage_and_target_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "videos" / "v1").mkdir(parents=True)
            (root / "run_manifest.json").write_text(json.dumps({
                "run_id": "r1", "version": "v0", "pipeline": {"operators": [
                    {"name": "frozen_target_frontier_filter", "config": {"target_model": "qwen3-vl-8b-instruct"}}
                ]}
            }), encoding="utf-8")
            (root / "run_summary.json").write_text(json.dumps({
                "run_id": "r1", "version": "v0", "completed": 1, "failed": 0,
                "accepted_frontier_novel_count": 1,
            }), encoding="utf-8")
            sample = {"producer": {"pipeline_fingerprint": "abc"}}
            (root / "high_quality_pool.jsonl").write_text(json.dumps({"content": sample}) + "\n", encoding="utf-8")
            (root / "videos" / "v1" / "result.json").write_text(json.dumps({"pool_candidates": [sample]}), encoding="utf-8")
            evaluation = evaluate_run(root, EvaluationPolicy())
        self.assertTrue(evaluation["eligible_for_promotion"])
        self.assertTrue(evaluation["hard_gates"]["lineage_complete"])


if __name__ == "__main__":
    unittest.main()
