"""Task-owned feedback adapter for evidence-first VideoRSI pipelines.

This module intentionally does not select a model or decode video.  A pipeline
or caller-provided signal provider supplies model-dependent observations; the
adapter validates candidate structure, aggregates route-level evidence, and
returns the stable feedback contract consumed by the common evolution loop.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

from ..feedback import CandidateFeedback
from ..models import TaskSpec


SignalProvider = Callable[[Mapping[str, Any]], Mapping[str, Any] | None]


def _normalise(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _hard_invalid_reason(row: Mapping[str, Any]) -> str | None:
    if not _normalise(row.get("question")):
        return "missing_question"
    choices = row.get("choices")
    if not isinstance(choices, list) or len(choices) != 4:
        return "choices_not_four"
    normalized = [_normalise(choice) for choice in choices]
    if any(not choice for choice in normalized):
        return "empty_choice"
    answer = _normalise(row.get("answer"))
    if normalized.count(answer) != 1:
        return "answer_not_unique_choice"
    evidence = row.get("selected_evidence") or row.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return "missing_evidence"
    if not _normalise(row.get("producer_route") or row.get("video_rsi_route")):
        return "missing_route"
    return None


class VideoRSICandidateEvaluator:
    """Aggregate VideoRSI hard validity and diagnostic feedback.

    ``signal_provider`` is optional. When configured, it may enrich each row
    with frozen-target, text-only, duplicate, or verifier results. Its failure
    is recorded as a signal rather than silently converted into a hard reject.
    """

    def __init__(self, signal_provider: SignalProvider | None = None) -> None:
        self.signal_provider = signal_provider

    def review(
        self, dataset_path: str, task: TaskSpec, **_: Any
    ) -> CandidateFeedback:
        del task
        rows: list[Mapping[str, Any]] = []
        with Path(dataset_path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"candidate row {line_number} must be an object")
                rows.append(value)

        total = len(rows)
        hard_invalid = Counter()
        route = defaultdict(lambda: Counter(total=0, hard_valid=0, accepted=0, reserve=0))
        text_only = Counter()
        difficulty = Counter()
        duplicate = Counter()
        rejections = Counter()
        signal_failures = 0
        reviewed_rows: list[Mapping[str, Any]] = []

        for row in rows:
            enriched = dict(row)
            if self.signal_provider is not None:
                try:
                    supplied = self.signal_provider(row)
                    if supplied:
                        enriched.update(supplied)
                except Exception:
                    signal_failures += 1
            reviewed_rows.append(enriched)
            route_name = str(enriched.get("producer_route") or enriched.get("video_rsi_route") or "unrouted")
            stats = route[route_name]
            stats["total"] += 1
            invalid = _hard_invalid_reason(enriched)
            if invalid:
                hard_invalid[invalid] += 1
                rejections[str(enriched.get("rejection_reason") or invalid)] += 1
                continue
            stats["hard_valid"] += 1
            disposition = str(enriched.get("pool_status") or enriched.get("status") or "candidate")
            if disposition == "accepted":
                stats["accepted"] += 1
            elif disposition in {"reserve", "replayable"}:
                stats["reserve"] += 1
            if enriched.get("rejection_reason"):
                rejections[str(enriched["rejection_reason"])] += 1
            text_only[str(enriched.get("text_only_outcome") or "unknown")] += 1
            difficulty[str(enriched.get("difficulty_label") or "unknown")] += 1
            duplicate[str(enriched.get("duplicate_outcome") or "unknown")] += 1

        valid = total - sum(hard_invalid.values())
        valid_rate = valid / total if total else 0.0
        accepted = sum(item["accepted"] for item in route.values())
        # Retain the original VideoRSI selection semantics: evolution compares
        # only the number of hard-valid, deduplicated frontier records admitted
        # to the data pool.  Evidence validity, routes, text-only behaviour and
        # difficulty are diagnostics for the authoring agent, not substitute
        # optimisation targets.  The fixed probe corpus makes this normalized
        # count order-equivalent to the prior integer proxy.
        accepted_frontier_novel = sum(
            1
            for row in reviewed_rows
            if _hard_invalid_reason(row) is None
            and str(row.get("pool_status") or row.get("status") or "candidate") == "accepted"
            and str(row.get("duplicate_outcome") or "unique") in {"unique", "novel"}
        )
        score = round(accepted_frontier_novel / max(1, total), 6)
        passed = valid > 0
        issues = tuple(f"{name}:{count}" for name, count in sorted(hard_invalid.items()))
        return CandidateFeedback(
            score=score,
            passed=passed,
            issues=issues,
            domain_feedback={
                "evaluator": "video-rsi-router-v1",
                "total_candidates": total,
                "hard_valid_candidates": valid,
                "accepted_candidates": accepted,
                "accepted_frontier_novel_count": accepted_frontier_novel,
                "route_metrics": {name: dict(values) for name, values in sorted(route.items())},
                "hard_invalid_reasons": dict(sorted(hard_invalid.items())),
                "rejection_reasons": dict(sorted(rejections.items())),
                "text_only_outcomes": dict(sorted(text_only.items())),
                "difficulty_labels": dict(sorted(difficulty.items())),
                "duplicate_outcomes": dict(sorted(duplicate.items())),
                "signal_provider_failures": signal_failures,
            },
        )
