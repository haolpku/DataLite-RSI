#!/usr/bin/env python3
"""Score math SFT transfer predictions.

Input JSONL, one record per generated sample:
  {"benchmark": "gsm8k", "question_id": "0", "predicted_answer": "18",
   "correct_answer": "18", "finish_reason": "stop"}

predicted_answer null/absent = parse failure.
finish_reason other than "stop" = runaway (length-truncated).
Both count as incorrect and are reported separately.

Output: JSON with per-benchmark scores and the aggregate primary_score.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

GREEDY_BENCHMARKS = ("gsm8k", "math", "minerva_math", "gaokao2024_mix", "olympiadbench")
SAMPLED_BENCHMARKS = ("amc23", "aime24", "aime25")
BENCHMARKS = GREEDY_BENCHMARKS + SAMPLED_BENCHMARKS


def normalise_answer(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("\\boxed{") and text.endswith("}"):
        text = text[len("\\boxed{"):-1].strip()
    text = text.strip("$").strip().replace(",", "")
    if text.endswith("."):
        text = text[:-1].strip()
    return text or None


def load_predictions(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{n}: expected a JSON object")
            for field in ("benchmark", "question_id", "correct_answer"):
                if field not in record:
                    raise ValueError(f"{path}:{n}: missing required field {field!r}")
            if record["benchmark"] not in BENCHMARKS:
                raise ValueError(
                    f"{path}:{n}: unknown benchmark {record['benchmark']!r}"
                )
            records.append(record)
    return records


def score_benchmark(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        per_question[str(r["question_id"])].append(r)

    correct = parse_failures = runaways = wrong = 0
    for samples in per_question.values():
        for s in samples:
            pred = normalise_answer(s.get("predicted_answer"))
            ref = normalise_answer(s.get("correct_answer"))
            runaway = s.get("finish_reason", "stop") != "stop"
            if runaway:
                runaways += 1
            if pred is None:
                parse_failures += 1
            elif pred == ref:
                if not runaway:
                    correct += 1
            else:
                wrong += 1

    sample_counts = {len(v) for v in per_question.values()}
    if len(sample_counts) > 1:
        raise ValueError(
            f"inconsistent sample counts per question: {sorted(sample_counts)}; "
            "avg@k requires a uniform k"
        )
    n_sampling = sample_counts.pop() if sample_counts else 0
    total = sum(len(v) for v in per_question.values())

    return {
        "score": correct / total if total else 0.0,
        "metric": "accuracy" if n_sampling <= 1 else f"avg@{n_sampling}",
        "n_sampling": n_sampling,
        "question_count": len(per_question),
        "total_count": total,
        "correct_count": correct,
        "incorrect_count": total - correct,
        "wrong_answer_count": wrong,
        "parse_failure_count": parse_failures,
        "runaway_count": runaways,
    }


def evaluate(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        grouped[r["benchmark"]].append(r)

    benchmarks = {name: score_benchmark(rows) for name, rows in sorted(grouped.items())}
    scores = [v["score"] for v in benchmarks.values()]
    return {
        "primary_score": sum(scores) / len(scores) if scores else 0.0,
        "benchmarks": benchmarks,
        "evaluated_benchmarks": sorted(benchmarks),
        "missing_benchmarks": [b for b in BENCHMARKS if b not in benchmarks],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    try:
        records = load_predictions(args.predictions)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not records:
        print("error: predictions file contains no records", file=sys.stderr)
        return 1
    try:
        metrics = evaluate(records)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
