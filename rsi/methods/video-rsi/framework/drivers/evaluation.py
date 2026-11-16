"""Fixed-policy evaluation for one VideoRSI run.

The evaluator deliberately keeps one primary comparison quantity:
``accepted_frontier_novel_count``.  Other measurements are hard gates or
diagnostics; they are never collapsed into a weighted score.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationPolicy:
    version: str = "rsi-eval-v1"
    frozen_target_model: str = "qwen3-vl-8b-instruct"
    require_no_text_fallback: bool = True
    require_zero_failures: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def evaluate_run(run_dir: Path, policy: EvaluationPolicy | None = None) -> dict[str, Any]:
    """Evaluate an already completed run without calling any model."""

    policy = policy or EvaluationPolicy()
    run_dir = run_dir.resolve()
    summary = _read_json(run_dir / "run_summary.json")
    manifest = _read_json(run_dir / "run_manifest.json")
    errors = _read_jsonl(run_dir / "errors.jsonl")
    pool_rows = _read_jsonl(run_dir / "high_quality_pool.jsonl")
    results = [
        _read_json(path)
        for path in sorted((run_dir / "videos").glob("*/result.json"))
    ]

    frontier = 0
    verification_pass = 0
    text_only_shortcut = 0
    fallback_used = False
    target_models: set[str] = set()
    task_counts: dict[str, int] = {}
    difficulty_counts = {"too_easy": 0, "hard_review_required": 0, "frontier": 0}
    for result in results:
        for sample in result.get("pool_candidates", []) or []:
            frontier += 1
            task = str(sample.get("task_type") or "unknown")
            task_counts[task] = task_counts.get(task, 0) + 1
        for sample in result.get("pool_candidates", []) or []:
            if sample.get("verification", {}).get("valid"):
                verification_pass += 1
        signals = result.get("feedback_signals", {}) or {}
        text_only_shortcut += int(signals.get("rejected_text_only_shortcut", 0))
        for sample in result.get("frontier_results", []) or []:
            target = sample.get("target_frontier", {}) or {}
            bucket = str(target.get("bucket") or "")
            if bucket in difficulty_counts:
                difficulty_counts[bucket] += 1
            backend = str(target.get("backend") or "")
            if backend:
                target_models.add(backend.removeprefix("api:"))
                fallback_used = fallback_used or backend.endswith(":text-fallback")

    operators = (manifest.get("pipeline") or {}).get("operators", [])
    configured_target = ""
    for operator in operators:
        if operator.get("name") == "frozen_target_frontier_filter":
            configured_target = str((operator.get("config") or {}).get("target_model") or "")
            break
    if configured_target:
        target_models.add(configured_target)

    primary = int(summary.get("accepted_frontier_novel_count", len(pool_rows)))
    difficulty_total = sum(difficulty_counts.values())
    too_easy_rate = (
        difficulty_counts["too_easy"] / difficulty_total
        if difficulty_total else None
    )
    gates = {
        "target_model_frozen": bool(configured_target == policy.frozen_target_model)
        if configured_target else policy.frozen_target_model in target_models,
        "no_text_fallback": not fallback_used if policy.require_no_text_fallback else True,
        "no_run_failures": not errors if policy.require_zero_failures else True,
        "lineage_complete": all(
            bool((row.get("content") or row).get("producer", {}).get("pipeline_fingerprint"))
            for row in pool_rows
        ),
    }
    return {
        "schema_version": "video-rsi-evaluation-v1",
        "policy": policy.as_dict(),
        "run_id": summary.get("run_id") or manifest.get("run_id") or run_dir.name,
        "version": summary.get("version") or manifest.get("version"),
        "primary_metric": "accepted_frontier_novel_count",
        "accepted_frontier_novel_count": primary,
        "diagnostics": {
            "videos_completed": int(summary.get("completed", len(results))),
            "videos_failed": int(summary.get("failed", len(errors))),
            "pool_rows": len(pool_rows),
            "frontier_rows_in_results": frontier,
            "verification_pass_rows": verification_pass,
            "text_only_shortcut_rejections": text_only_shortcut,
            "difficulty_counts": difficulty_counts,
            "difficulty_total": difficulty_total,
            "too_easy_rate": too_easy_rate,
            "target_models_seen": sorted(target_models),
            "accepted_by_task": dict(sorted(task_counts.items())),
        },
        "hard_gates": gates,
        "eligible_for_promotion": all(gates.values()),
    }


def compare_evaluations(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Apply the fixed gates, then require a strict primary-metric increase."""

    baseline_value = int(baseline.get("accepted_frontier_novel_count", 0))
    candidate_value = int(candidate.get("accepted_frontier_novel_count", 0))
    gates = dict(candidate.get("hard_gates") or {})
    baseline_rate = (baseline.get("diagnostics") or {}).get("too_easy_rate")
    candidate_rate = (candidate.get("diagnostics") or {}).get("too_easy_rate")
    # Difficulty is deliberately a feedback signal only.  v0 must not discard
    # or block promotion of a candidate solely because its mix of too-easy or
    # hard-review labels changed; the RSI agent can use these rates when
    # deciding what to edit in a later round.
    eligible = bool(candidate.get("eligible_for_promotion", all(gates.values())))
    improved = candidate_value > baseline_value
    decision = "accept" if eligible and improved else "reject"
    reason = "primary_metric_increased" if decision == "accept" else (
        "hard_gate_failed" if not eligible else "primary_metric_not_strictly_increased"
    )
    return {
        "decision": decision,
        "reason": reason,
        "baseline_primary": baseline_value,
        "candidate_primary": candidate_value,
        "candidate_hard_gates": gates,
        "difficulty_signal": {
            "baseline_too_easy_rate": baseline_rate,
            "candidate_too_easy_rate": candidate_rate,
        },
        "baseline_too_easy_rate": baseline_rate,
        "candidate_too_easy_rate": candidate_rate,
    }
