"""Score Video-MME multiple-choice predictions without model dependencies."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def score_rows(rows: list[dict]) -> dict:
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for row in rows:
        category_name = str(row.get("category") or row["task_type"])
        category = CATEGORY_KEYS.get(category_name)
        if category is None:
            raise ValueError(f"unknown Video-MME category: {category_name}")
        prediction = str(row["prediction"]).strip().upper()
        answer = str(row["answer"]).strip().upper()
        totals[category] += 1
        correct[category] += prediction == answer
    if not totals:
        raise ValueError("no prediction rows")
    categories = {name: correct[name] / total for name, total in sorted(totals.items())}
    return {"overall": sum(correct.values()) / sum(totals.values()), "categories": categories}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line]
    result = score_rows(rows)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
CATEGORY_KEYS = {
    "Action Reasoning": "action_reasoning",
    "Object Recognition": "object_recognition",
    "Counting": "counting",
    "Information Synopsis": "information_synopsis",
    "Object Reasoning": "object_reasoning",
    "Temporal Perception": "temporal_perception",
    "Attribute": "attribute",
    "Temporal Reasoning": "temporal_reasoning",
    "Action Recognition": "action_recognition",
    "OCR": "ocr",
    "Spatial Perception": "spatial_perception",
    "Spatial Reasoning": "spatial_reasoning",
}
