#!/usr/bin/env python3
"""Merge per-container JSONL results in sample order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def records_for(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        records.append(value)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    paths = sorted(
        args.run_dir.glob("batch_*/sample_*/results.jsonl"),
        key=lambda path: int(path.parent.name.removeprefix("sample_")),
    )
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for path in paths:
        for record in records_for(path):
            sample_id = record.get("id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{path}: result is missing a non-empty id")
            if sample_id in seen:
                raise ValueError(f"duplicate sample id: {sample_id}")
            seen.add(sample_id)
            records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    print(f"merged {len(records)} samples into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
