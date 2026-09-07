import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "benchmarks" / "agents-last-exam" / "evaluator.py"
SPEC = importlib.util.spec_from_file_location("agents_last_exam_evaluator", MODULE_PATH)
assert SPEC and SPEC.loader
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def write_run(
    root: Path, name: str, *, score=None, eval_status="success", status="completed"
):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": name,
                "status": status,
                "task": {"path": f"tasks/demo/{name}", "slug": name},
                "agent": {"id": "dummy", "class": "dummy", "model": "test-model"},
                "timings": {"duration_s": 2},
                "usage": {"total_cost_usd": 0.5},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "eval_result.json").write_text(
        json.dumps({"eval_status": eval_status, "score": score}), encoding="utf-8"
    )


class EvaluatorTests(unittest.TestCase):
    def test_aggregates_scores_and_keeps_failure_as_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, "good", score=1.0)
            write_run(root, "partial", score=0.5)
            write_run(
                root, "timeout", score=None, eval_status="timeout", status="timeout"
            )

            result = evaluator.evaluate_predictions(root)

        self.assertEqual(result["counts"]["runs"], 3)
        self.assertEqual(result["counts"]["invalid_runs"], 0)
        self.assertAlmostEqual(result["metrics"]["primary_score"], 0.5)
        self.assertAlmostEqual(result["metrics"]["pass_rate"], 1 / 3)

    def test_malformed_record_is_reported_and_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, "good", score=0.25)
            broken = root / "broken"
            broken.mkdir()
            (broken / "run.json").write_text(
                json.dumps(
                    {
                        "run_id": "broken",
                        "status": "completed",
                        "task": {"path": "tasks/demo/broken"},
                        "agent": {"id": "dummy"},
                    }
                ),
                encoding="utf-8",
            )
            (broken / "eval_result.json").write_text(
                json.dumps({"eval_status": "success", "score": "not-a-number"}),
                encoding="utf-8",
            )

            result = evaluator.evaluate_predictions(root)

        self.assertEqual(result["counts"]["runs"], 2)
        self.assertEqual(result["counts"]["scored_runs"], 1)
        self.assertEqual(result["counts"]["invalid_runs"], 1)
        self.assertAlmostEqual(result["metrics"]["primary_score"], 0.25)

    def test_missing_runs_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                evaluator.evaluate_predictions(Path(directory))


if __name__ == "__main__":
    unittest.main()
