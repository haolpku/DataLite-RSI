"""LSC scoring and candidate-ranking helpers.

The implementation follows the frozen OPSD protocol: compare top-k support
sets for the same student continuation under 80% and 100% privileged
reference contexts.  It deliberately does not infer probabilities outside
the returned top-k support.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def support_jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    """Return Jaccard overlap for two token-id supports."""

    left_set, right_set = set(left), set(right)
    return len(left_set & right_set) / max(1, len(left_set | right_set))


def mean_support_overlap(
    student_supports: Sequence[Iterable[int]],
    privileged_supports: Sequence[Iterable[int]],
) -> float:
    """Average per-token top-k support overlap over the common window."""

    count = min(len(student_supports), len(privileged_supports))
    if count == 0:
        return 0.0
    return sum(
        support_jaccard(student_supports[i], privileged_supports[i])
        for i in range(count)
    ) / count


def lsc(level_80_mean_support_overlap: float, level_100_mean_support_overlap: float) -> float:
    """Compute Late Support Collapse: J_80 - J_100."""

    return float(level_80_mean_support_overlap) - float(level_100_mean_support_overlap)


def lsc_from_row(row: Mapping[str, Any]) -> float:
    """Compute LSC from a feature row using the canonical field names."""

    return lsc(
        row["level_80_mean_support_overlap"],
        row["level_100_mean_support_overlap"],
    )


def cross_lsc(context_rows: Sequence[Mapping[str, Any]]) -> float:
    """Average per-reference LSC for a fixed, preselected context list."""

    if not context_rows:
        return 0.0
    return sum(lsc_from_row(row) for row in context_rows) / len(context_rows)


def rank_low_lsc(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return rows in ascending LSC order (lowest predicted quality first)."""

    return sorted(rows, key=lsc_from_row)


def bottom_fraction(rows: Iterable[Mapping[str, Any]], fraction: float = 0.10) -> list[Mapping[str, Any]]:
    """Select the lowest-LSC fraction, retaining at least one row."""

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in the interval (0, 1]")
    ranked = rank_low_lsc(rows)
    return ranked[: max(1, round(len(ranked) * fraction))]


def p_shape(lsc_value: float, g80: float) -> float:
    """Frozen single-context quality estimate calibrated for transfer."""

    logit = 0.842847 + 113.071472 * lsc_value + 8.183589 * g80
    # Stable sigmoid for extreme transferred scores.
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_logit = math.exp(logit)
    return exp_logit / (1.0 + exp_logit)


__all__ = [
    "bottom_fraction",
    "cross_lsc",
    "lsc",
    "lsc_from_row",
    "mean_support_overlap",
    "p_shape",
    "rank_low_lsc",
    "support_jaccard",
]
