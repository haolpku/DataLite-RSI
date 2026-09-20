#!/usr/bin/env python3
"""Re-aggregate already-graded image-edit transfer scores.

Input JSONL, one record per judged sample:
  {"benchmark": "gedit_bench", "sample_id": "g0", "score": 8.2}

`score` is the native-scale judge value (GEdit-Bench 0-10, ImgEdit-Bench 0-5).
The evaluator does not call a model. Null or out-of-range scores are invalid
and are excluded from the mean.

primary_score = 0.5 * (gedit_mean / 10 + imgedit_mean / 5)
when both benches are present; otherwise the mean of the available
normalised benches.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

BENCHMARKS = ("gedit_bench", "imgedit_bench")
SCALES = {
    "gedit_bench": (0.0, 10.0),
    "imgedit_bench": (0.0, 5.0),
}


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
            for field in ("benchmark", "sample_id"):
                if field not in record:
                    raise ValueError(f"{path}:{n}: missing required field {field!r}")
            if record["benchmark"] not in BENCHMARKS:
                raise ValueError(
                    f"{path}:{n}: unknown benchmark {record['benchmark']!r}"
                )
            records.append(record)
    return records


def _as_score(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def score_benchmark(name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    lo, hi = SCALES[name]
    valid: list[float] = []
    invalid = 0
    for record in records:
        score = _as_score(record.get("score"))
        if score is None or score < lo or score > hi:
            invalid += 1
            continue
        valid.append(score)
    mean = sum(valid) / len(valid) if valid else None
    return {
        "score": mean,
        "normalised": None if mean is None else mean / hi,
        "scale": [lo, hi],
        "sample_count": len(records),
        "valid_count": len(valid),
        "invalid_count": invalid,
    }


def evaluate(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["benchmark"]].append(record)

    benchmarks = {
        name: score_benchmark(name, rows) for name, rows in sorted(grouped.items())
    }
    normalised = [
        row["normalised"]
        for row in benchmarks.values()
        if row["normalised"] is not None
    ]
    return {
        "primary_score": sum(normalised) / len(normalised) if normalised else 0.0,
        "gedit_bench": (benchmarks.get("gedit_bench") or {}).get("score"),
        "imgedit_bench": (benchmarks.get("imgedit_bench") or {}).get("score"),
        "benchmarks": benchmarks,
        "evaluated_benchmarks": sorted(benchmarks),
        "missing_benchmarks": [name for name in BENCHMARKS if name not in benchmarks],
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
