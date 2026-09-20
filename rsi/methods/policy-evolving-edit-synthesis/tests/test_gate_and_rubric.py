"""Unit tests for policy promotion, retirement, and the frozen acceptance rubric.

These tests import only the method package. They do not call a model or an API.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

METHOD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD_DIR / "src"))

from deltasynth.harness.evolve.gate import (  # noqa: E402
    GateConfig,
    evaluate,
    retire_stale_hypotheses,
)
from deltasynth.harness.evolve.state import (  # noqa: E402
    Hypothesis,
    Policy,
    StrategyState,
)
from deltasynth.rubrics.registry import decide_goodcase, load_rubric  # noqa: E402

RUBRIC_PATH = METHOD_DIR / "configs" / "rubrics" / "if_vc_vq.json"


def _claim(*, support: list[str], oppose: list[str] | None = None) -> Hypothesis:
    return Hypothesis(
        hypothesis_id="h1",
        statement="name the destination object explicitly",
        support_batches=support,
        oppose_batches=oppose or [],
    )


class GateTest(unittest.TestCase):
    def test_needs_two_supporting_batches(self) -> None:
        verdict = evaluate(_claim(support=["b1"]), StrategyState())
        self.assertFalse(verdict)
        self.assertTrue(any("needs 2" in reason for reason in verdict.reasons))

    def test_promotes_when_supported_twice_and_not_contradicted(self) -> None:
        verdict = evaluate(_claim(support=["b1", "b2"]), StrategyState())
        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.reasons, [])

    def test_refuses_when_contradicted_at_least_as_often(self) -> None:
        verdict = evaluate(
            _claim(support=["b1", "b2"], oppose=["b3", "b4"]),
            StrategyState(),
        )
        self.assertFalse(verdict)
        self.assertTrue(any("contradicted" in reason for reason in verdict.reasons))

    def test_stops_at_active_policy_ceiling(self) -> None:
        state = StrategyState(
            policies=[
                Policy(policy_id="p1", trigger="t1", procedure="a", active=True),
                Policy(policy_id="p2", trigger="t2", procedure="b", active=True),
                Policy(policy_id="p3", trigger="t3", procedure="c", active=True),
            ]
        )
        verdict = evaluate(
            _claim(support=["b1", "b2"]),
            state,
            GateConfig(max_policies=3),
        )
        self.assertFalse(verdict)
        self.assertTrue(any("already holds 3" in reason for reason in verdict.reasons))

    def test_retires_after_two_untested_batches(self) -> None:
        state = StrategyState(
            hypotheses=[
                Hypothesis(
                    hypothesis_id="stale",
                    statement="unused claim",
                    untested_streak=2,
                    status="open",
                )
            ]
        )
        retired = retire_stale_hypotheses(state)
        self.assertEqual(len(retired), 1)
        self.assertEqual(state.hypotheses[0].status, "retired")


class RubricTest(unittest.TestCase):
    def test_committed_thresholds_match_the_manifest(self) -> None:
        payload = json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))
        minima = {axis["axis_id"]: axis["minimum_score"] for axis in payload["axes"]}
        self.assertEqual(minima["instruction_following"], 4.0)
        self.assertEqual(minima["visual_consistency"], 3.0)
        self.assertEqual(minima["visual_quality"], 3.0)
        self.assertEqual(payload["minimum_average"], 3.5)

    def test_environment_failure_rejects_regardless_of_axis_scores(self) -> None:
        rubric = load_rubric(RUBRIC_PATH)
        ok, average, failures = decide_goodcase(
            rubric,
            {
                "environment_valid": False,
                "axes": {
                    "instruction_following": {"score": 5},
                    "visual_consistency": {"score": 5},
                    "visual_quality": {"score": 5},
                },
            },
        )
        self.assertFalse(ok)
        self.assertIsNone(average)
        self.assertEqual(failures, ["environment_invalid"])

    def test_one_axis_below_minimum_fails_even_if_mean_clears(self) -> None:
        rubric = load_rubric(RUBRIC_PATH)
        ok, average, failures = decide_goodcase(
            rubric,
            {
                "environment_valid": True,
                "axes": {
                    "instruction_following": {"score": 3.0},
                    "visual_consistency": {"score": 5.0},
                    "visual_quality": {"score": 5.0},
                },
            },
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(average, 13.0 / 3.0)
        self.assertEqual(failures, ["below_minimum:instruction_following"])

    def test_mean_below_3_5_fails_when_every_axis_clears_its_floor(self) -> None:
        rubric = load_rubric(RUBRIC_PATH)
        ok, average, failures = decide_goodcase(
            rubric,
            {
                "environment_valid": True,
                "axes": {
                    "instruction_following": {"score": 4.0},
                    "visual_consistency": {"score": 3.0},
                    "visual_quality": {"score": 3.0},
                },
            },
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(average, 10.0 / 3.0)
        self.assertEqual(failures, ["below_minimum_average"])

    def test_accepted_pair(self) -> None:
        rubric = load_rubric(RUBRIC_PATH)
        ok, average, failures = decide_goodcase(
            rubric,
            {
                "environment_valid": True,
                "axes": {
                    "instruction_following": {"score": 4.0},
                    "visual_consistency": {"score": 3.5},
                    "visual_quality": {"score": 3.5},
                },
            },
        )
        self.assertTrue(ok)
        self.assertAlmostEqual(average, 11.0 / 3.0)
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
