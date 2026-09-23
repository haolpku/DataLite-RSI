#!/usr/bin/env python3
"""Aggregate single-sample HLE runner records into benchmark metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def aggregate(path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        sample_id = value.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{path}:{line_number} is missing a non-empty id")
        if sample_id in seen:
            raise ValueError(f"duplicate sample id: {sample_id}")
        seen.add(sample_id)
        records.append(value)

    successful = [record for record in records if record.get("status") == "success"]
    scored = [record for record in successful if record.get("score") in {0, 1}]
    correct = sum(record.get("score") == 1 for record in scored)
    return {
        "benchmark": "hle-withtools",
        "num_records": len(records),
        "num_success": len(successful),
        "num_failed": len(records) - len(successful),
        "num_scored": len(scored),
        "accuracy": correct / len(scored) if scored else None,
        "completion_rate": len(successful) / len(records) if records else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = aggregate(args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
