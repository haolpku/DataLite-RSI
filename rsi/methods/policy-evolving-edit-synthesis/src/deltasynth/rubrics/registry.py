from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import RubricDefinition


def load_rubric(path: Path) -> RubricDefinition:
    return RubricDefinition.model_validate_json(path.read_text(encoding="utf-8"))


def decide_goodcase(
    rubric: RubricDefinition,
    result: dict[str, Any],
) -> tuple[bool, float | None, list[str]]:
    if result.get("environment_valid") is not True:
        return False, None, ["environment_invalid"]

    axes = result.get("axes") or {}
    failures: list[str] = []
    weighted_sum = 0.0
    total_weight = 0.0
    for axis in rubric.axes:
        value = (axes.get(axis.axis_id) or {}).get("score")
        if not isinstance(value, (int, float)):
            failures.append(f"missing_axis:{axis.axis_id}")
            continue
        score = float(value)
        if score < axis.minimum_score:
            failures.append(f"below_minimum:{axis.axis_id}")
        weighted_sum += score * axis.weight
        total_weight += axis.weight

    if failures or total_weight <= 0:
        average = weighted_sum / total_weight if total_weight > 0 else None
        return False, average, failures
    average = weighted_sum / total_weight
    if average < rubric.minimum_average:
        failures.append("below_minimum_average")
    return not failures, average, failures
