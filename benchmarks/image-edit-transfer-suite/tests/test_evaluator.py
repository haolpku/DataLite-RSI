"""Tests for the image-edit transfer-suite evaluator."""

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


class ScoreBenchmarkTest(unittest.TestCase):
    def test_mean_ignores_null_and_out_of_range(self) -> None:
        gedit = evaluator.score_benchmark(
            "gedit_bench",
            [
                {"benchmark": "gedit_bench", "sample_id": "a", "score": 8.0},
                {"benchmark": "gedit_bench", "sample_id": "b", "score": 10.0},
                {"benchmark": "gedit_bench", "sample_id": "c", "score": None},
            ],
        )
        self.assertEqual(gedit["score"], 9.0)
        self.assertEqual(gedit["normalised"], 0.9)
        self.assertEqual(gedit["invalid_count"], 1)

        imgedit = evaluator.score_benchmark(
            "imgedit_bench",
            [
                {"benchmark": "imgedit_bench", "sample_id": "a", "score": 4.0},
                {"benchmark": "imgedit_bench", "sample_id": "b", "score": 5.0},
                {"benchmark": "imgedit_bench", "sample_id": "c", "score": 9.0},
            ],
        )
        self.assertEqual(imgedit["score"], 4.5)
        self.assertEqual(imgedit["normalised"], 0.9)
        self.assertEqual(imgedit["invalid_count"], 1)

    def test_rejects_unknown_benchmark(self) -> None:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl") as fh:
            fh.write('{"benchmark": "other", "sample_id": "x", "score": 1}\n')
            fh.flush()
            with self.assertRaises(ValueError):
                evaluator.load_predictions(Path(fh.name))


class FixtureTest(unittest.TestCase):
    def test_primary_is_mean_of_normalised_benches(self) -> None:
        metrics = evaluator.evaluate(evaluator.load_predictions(FIXTURE))
        self.assertEqual(metrics["gedit_bench"], 9.0)
        self.assertEqual(metrics["imgedit_bench"], 4.5)
        self.assertAlmostEqual(metrics["primary_score"], 0.9)
        self.assertEqual(metrics["missing_benchmarks"], [])
        self.assertEqual(metrics["benchmarks"]["gedit_bench"]["invalid_count"], 1)
        self.assertEqual(metrics["benchmarks"]["imgedit_bench"]["invalid_count"], 1)

    def test_cli_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "metrics.json"
            rc = evaluator.main(["--predictions", str(FIXTURE), "--output", str(out)])
            self.assertEqual(rc, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertAlmostEqual(payload["primary_score"], 0.9)


if __name__ == "__main__":
    unittest.main()
