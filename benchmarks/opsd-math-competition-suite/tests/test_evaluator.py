"""Tests for the OPSD math competition-suite evaluator."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_DIR))

import evaluator  # noqa: E402

FIXTURE = BENCHMARK_DIR / "tests" / "fixture_predictions.jsonl"


class NormaliseAnswerTest(unittest.TestCase):
    def test_strips_boxed_dollar_separators_and_period(self) -> None:
        cases = {
            "\\boxed{2}": "2",
            "$-3$": "-3",
            "1,200": "1200",
            "7.": "7",
            "  42  ": "42",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(evaluator.normalise_answer(raw), expected)

    def test_none_and_empty_return_none(self) -> None:
        for raw in (None, "", "   "):
            with self.subTest(raw=raw):
                self.assertIsNone(evaluator.normalise_answer(raw))


class ScoreBenchmarkTest(unittest.TestCase):
    def test_avg_at_k(self) -> None:
        records = [
            {
                "benchmark": "aime25",
                "question_id": "a",
                "predicted_answer": "1",
                "correct_answer": "1",
            },
            {
                "benchmark": "aime25",
                "question_id": "a",
                "predicted_answer": "9",
                "correct_answer": "1",
            },
            {
                "benchmark": "aime25",
                "question_id": "b",
                "predicted_answer": "2",
                "correct_answer": "2",
            },
            {
                "benchmark": "aime25",
                "question_id": "b",
                "predicted_answer": "2",
                "correct_answer": "2",
            },
        ]
        r = evaluator.score_benchmark(records)
        self.assertEqual(r["metric"], "avg@2")
        self.assertEqual(r["score"], 0.75)

    def test_parse_failure_and_runaway_reported_separately(self) -> None:
        records = [
            {
                "benchmark": "aime26",
                "question_id": "a",
                "predicted_answer": None,
                "correct_answer": "1",
            },
            {
                "benchmark": "aime26",
                "question_id": "b",
                "predicted_answer": "9",
                "correct_answer": "1",
                "finish_reason": "length",
            },
        ]
        r = evaluator.score_benchmark(records)
        self.assertEqual(r["score"], 0.0)
        self.assertEqual(r["parse_failure_count"], 1)
        self.assertEqual(r["runaway_count"], 1)
        self.assertEqual(r["wrong_answer_count"], 1)


class FixtureTest(unittest.TestCase):
    def test_stable_scores(self) -> None:
        metrics = evaluator.evaluate(evaluator.load_predictions(FIXTURE))
        self.assertEqual(metrics["benchmarks"]["aime25"]["score"], 0.75)
        self.assertEqual(metrics["benchmarks"]["hmmt25"]["score"], 0.5)
        self.assertEqual(metrics["benchmarks"]["aime26"]["score"], 0.5)
        self.assertEqual(metrics["benchmarks"]["hmmt26"]["score"], 0.5)
        self.assertEqual(metrics["benchmarks"]["brumo25"]["score"], 1.0)
        self.assertAlmostEqual(metrics["primary_score"], 0.65)
        self.assertEqual(metrics["missing_benchmarks"], [])

    def test_cli_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "metrics.json"
            rc = evaluator.main(["--predictions", str(FIXTURE), "--output", str(out)])
            self.assertEqual(rc, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertAlmostEqual(payload["primary_score"], 0.65)


if __name__ == "__main__":
    unittest.main()
