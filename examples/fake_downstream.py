#!/usr/bin/env python3
"""Fabricated downstream evaluator for framework validation.

Stands in for real SFT + benchmark evaluation, which needs GPUs this host does
not have free. It reads the framework's request JSON and writes a result JSON
that satisfies the real `downstream_eval` schema, so the validation path --
request serialization, subprocess invocation, result parsing, every count and
bad-case consistency rule, attribution input -- is genuinely exercised. Only the
numbers are invented.

Scores improve slightly per checkpoint so the loop sees a plausible trajectory,
and bad cases are drawn from the candidate dataset so attribution has real text
to reason about.

Invoked by the framework as:
    <this script> --request <path> --result <path>

NOT a measurement. Never present output from this as a real benchmark result.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


# Deterministic per checkpoint so a rerun reproduces the same fabricated trend.
BASE_SCORES = {"gsm8k": 0.42, "math500": 0.27}
PER_CHECKPOINT_GAIN = 0.015


def _load_candidate_rows(dataset_path: str | None, limit: int = 40) -> list[dict]:
    if not dataset_path:
        return []
    path = Path(dataset_path)
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def _bad_cases(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    """Build bad cases from real candidate text so attribution has substance."""
    cases: list[dict] = []
    for index in range(count):
        row = rows[index % len(rows)] if rows else {}
        question = str(row.get("instruction") or row.get("question") or f"synthetic question {index}")
        reference = str(row.get("output") or row.get("answer") or "")
        wrong = str(rng.randint(2, 97))
        correct = ""
        if "####" in reference:
            correct = reference.split("####")[-1].strip()
        elif "\\boxed{" in reference:
            correct = reference.split("\\boxed{")[-1].split("}")[0].strip()
        cases.append(
            {
                "question": question[:600],
                "model_response": (
                    f"Let me work through it. {reference[:200]} "
                    f"So the answer is {wrong}."
                ),
                "parsed_answer": wrong,
                "correct_answer": correct or str(rng.randint(1, 99)),
                "response_token_count": rng.randint(80, 400),
                "finish_reason": "stop",
                "dimension": "arithmetic" if index % 2 == 0 else "multi_step_reasoning",
            }
        )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    checkpoint = int(request.get("checkpoint_iteration", 0))
    stage = str(request.get("stage", "periodic"))
    dataset_path = request.get("dataset_path")
    expected = list(request.get("expected_benchmarks") or ["gsm8k"])
    per_benchmark = int(request.get("bad_cases_per_benchmark", 3))

    rng = random.Random(1000 + checkpoint)
    rows = _load_candidate_rows(dataset_path)

    # A baseline checkpoint evaluates the untrained model: no dataset, no SFT.
    training_performed = stage != "baseline"
    gain = 0.0 if stage == "baseline" else PER_CHECKPOINT_GAIN * checkpoint

    question_count = 32
    n_sampling = 1
    benchmarks: dict[str, dict] = {}
    for name in expected:
        target = min(0.95, BASE_SCORES.get(name, 0.30) + gain)
        total_count = question_count * n_sampling
        correct_count = int(round(target * total_count))
        incorrect_count = total_count - correct_count
        # The framework asserts score == correct_count / total_count exactly, so
        # derive the score from the counts rather than rounding independently.
        score = correct_count / total_count
        # The three category counts must sum to incorrect_count.
        parse_failure_count = 1 if incorrect_count >= 3 else 0
        runaway_count = 1 if incorrect_count >= 5 else 0
        wrong_answer_count = incorrect_count - parse_failure_count - runaway_count
        benchmarks[name] = {
            "score": score,
            "metric": "accuracy",
            "incorrect_count": incorrect_count,
            "wrong_answer_count": wrong_answer_count,
            "parse_failure_count": parse_failure_count,
            "runaway_count": runaway_count,
            "correct_count": correct_count,
            "total_count": total_count,
            "question_count": question_count,
            "n_sampling": n_sampling,
            # Exactly min(bad_cases_per_benchmark, incorrect_count) entries.
            "bad_cases": _bad_cases(rows, min(per_benchmark, incorrect_count), rng),
        }

    result = {
        "status": "ok",
        "checkpoint_iteration": checkpoint,
        "incumbent_iteration": request.get("incumbent_iteration"),
        "dataset_path": dataset_path,
        "stage": stage,
        "training_performed": training_performed,
        "model": "FABRICATED-no-real-training",
        "benchmarks": benchmarks,
        "summary": (
            "FABRICATED downstream feedback for framework validation. "
            "No training or benchmark evaluation was performed; the GPUs on this "
            "host are fully occupied. Scores and bad cases are synthetic."
        ),
        "artifacts": {},
        "wall_time_seconds": 1.0,
    }
    Path(args.result).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[fabricated] checkpoint={checkpoint} stage={stage} benchmarks={list(benchmarks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
