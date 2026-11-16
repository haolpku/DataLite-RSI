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
from typing import Any, Callable, Mapping, Optional

from ..feedback import CandidateFeedback
from ..models import TaskSpec


SignalProvider = Callable[[Mapping[str, Any]], Optional[Mapping[str, Any]]]


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


def _pool_disposition(row: Mapping[str, Any]) -> str:
    """Return the effective pool disposition for a structurally valid row.

    Early route-specific pipelines frequently omit ``pool_status`` while they
    are still being authored.  Treating that omission as a rejection made a
    fully hard-valid probe score zero, and therefore taught the controller
    nothing.  The VideoRSI policy is that text-only and difficulty signals are
    diagnostic, not hard gates: an otherwise valid, non-duplicate candidate is
    admissible unless the pipeline *explicitly* places it in reserve/reject.
    """
    explicit = _normalise(row.get("pool_status") or row.get("status"))
    if explicit:
        return explicit
    duplicate = _normalise(row.get("duplicate_outcome"))
    if duplicate in {"duplicate", "near_duplicate"}:
        return "rejected"
    if row.get("rejection_reason"):
        return "rejected"
    return "accepted"


def _training_value(row: Mapping[str, Any]) -> tuple[bool, bool]:
    """Return ``(is_training_ready, has_explicit_quality_signal)``.

    VideoRSI historically used acceptance yield as its proxy because early
    pipelines did not emit quality labels.  That is still a useful backwards
    compatible fallback, but once a route has authored auditable grounding,
    shortcut, distractor, or training-value labels, promotion must not reward a
    large pool of syntactically valid caption shortcuts.
    """
    label = _normalise(row.get("training_value_label"))
    grounding = _normalise(row.get("visual_grounding_outcome"))
    shortcut = _normalise(row.get("caption_shortcut_risk"))
    distractor = _normalise(row.get("distractor_quality"))
    explicit = any((label, grounding, shortcut, distractor))
    if not explicit:
        return True, False

    # ``unknown`` is deliberately not a positive result.  A route that has
    # added an audit must not receive promotion credit for silently treating an
    # unavailable audit as grounded/low-risk.  The candidate can still remain
    # in the reserve pool for diagnosis or later replay.
    positive_labels = {"high", "medium", "training_ready", "ready"}
    grounded = {"grounded", "verified", "visible"}
    low_shortcut = {"low", "none", "minimal"}
    strong_distractors = {"strong", "good", "valid", "discriminative"}
    ready = (
        label in positive_labels
        and grounding in grounded
        and shortcut in low_shortcut
        and distractor in strong_distractors
    )
    return ready, True


def _has_complete_quality_audit(row: Mapping[str, Any]) -> bool:
    required = (
        "training_value_label",
        "visual_grounding_outcome",
        "caption_shortcut_risk",
        "distractor_quality",
    )
    unavailable = {"", "unknown", "unavailable", "not_assessed"}
    return all(_normalise(row.get(field)) not in unavailable for field in required)


def _answer_position(row: Mapping[str, Any]) -> str | None:
    """Return the correct A/B/C/D position when the MCQ is well formed."""
    choices = row.get("choices")
    if not isinstance(choices, list) or len(choices) != 4:
        return None
    answer = _normalise(row.get("answer"))
    normalized = [_normalise(choice) for choice in choices]
    if normalized.count(answer) != 1:
        return None
    return "ABCD"[normalized.index(answer)]


def _video_identity(row: Mapping[str, Any]) -> str | None:
    """Best-effort grouping key without inventing a video identity."""
    video = row.get("video")
    if isinstance(video, Mapping):
        for key in ("uri", "path", "id", "video_id"):
            value = _normalise(video.get(key))
            if value:
                return value
    if isinstance(video, str):
        value = _normalise(video)
        if value:
            return value
    for key in ("video_id", "source_video_id"):
        value = _normalise(row.get(key))
        if value:
            return value
    return None


def _distribution_summary(counts: Counter[str], *, expected: tuple[str, ...] = ()) -> dict[str, Any]:
    total = sum(counts.values())
    labels = expected or tuple(sorted(counts))
    rendered = {label: counts.get(label, 0) for label in labels}
    if not total:
        return {"counts": rendered, "total": 0, "coverage": 0, "largest_share": 0.0}
    return {
        "counts": rendered,
        "total": total,
        "coverage": sum(1 for label in labels if counts.get(label, 0)),
        "largest_share": round(max(counts.values()) / total, 6),
    }


class VideoRSICandidateEvaluator:
    """Aggregate VideoRSI hard validity and diagnostic feedback.

    ``signal_provider`` is optional and hardware/model agnostic. It may call a
    remote API, a CPU/GPU model service, or a deterministic auditor to enrich
    each row. Its independent output is kept separate from pipeline-authored
    claims; provider failures are recorded instead of becoming hard rejects.
    """

    def __init__(
        self,
        signal_provider: SignalProvider | None = None,
        *,
        require_independent_quality: bool = False,
        model_role: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.signal_provider = signal_provider
        self.require_independent_quality = require_independent_quality
        self.model_role = model_role or ("reference" if signal_provider else "none")
        if self.model_role not in {"target", "reference", "none"}:
            raise ValueError("model_role must be 'target', 'reference', or 'none'")
        if self.model_role != "none" and signal_provider is None:
            raise ValueError("target/reference model_role requires a signal_provider")
        if self.model_role == "none" and model_name:
            raise ValueError("model_name requires target/reference model_role")
        self.model_name = model_name

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
        training_value = Counter()
        rejections = Counter()
        frame_visibility = Counter()
        signal_failures = 0
        reviewed_rows: list[Mapping[str, Any]] = []
        independent_signals: list[Mapping[str, Any]] = []

        for row in rows:
            enriched = dict(row)
            supplied: Mapping[str, Any] = {}
            if self.signal_provider is not None:
                try:
                    supplied_value = self.signal_provider(row)
                    supplied = supplied_value if isinstance(supplied_value, Mapping) else {}
                    if supplied:
                        enriched.update(supplied)
                except Exception:
                    signal_failures += 1
            reviewed_rows.append(enriched)
            # Keep independent verifier output separate from pipeline-authored
            # labels. Otherwise a generated pipeline could promote itself by
            # writing optimistic quality fields into its own records.
            independent_signals.append(supplied)
            route_name = str(enriched.get("producer_route") or enriched.get("video_rsi_route") or "unrouted")
            stats = route[route_name]
            stats["total"] += 1
            invalid = _hard_invalid_reason(enriched)
            if invalid:
                hard_invalid[invalid] += 1
                rejections[str(enriched.get("rejection_reason") or invalid)] += 1
                continue
            stats["hard_valid"] += 1
            disposition = _pool_disposition(enriched)
            if disposition == "accepted":
                stats["accepted"] += 1
            elif disposition in {"reserve", "replayable"}:
                stats["reserve"] += 1
            if enriched.get("rejection_reason"):
                rejections[str(enriched["rejection_reason"])] += 1
            text_only[str(enriched.get("text_only_outcome") or "unknown")] += 1
            difficulty[str(enriched.get("difficulty_label") or "unknown")] += 1
            duplicate[str(enriched.get("duplicate_outcome") or "unknown")] += 1
            training_value[str(enriched.get("training_value_label") or "unknown")] += 1
            frame_visibility[str(enriched.get("frame_visibility_outcome") or "unknown")] += 1

        valid = total - sum(hard_invalid.values())
        valid_rate = valid / total if total else 0.0
        accepted = sum(item["accepted"] for item in route.values())
        # Retain the original VideoRSI selection semantics: evolution compares
        # only the number of hard-valid, deduplicated frontier records admitted
        # to the data pool.  Evidence validity, routes, text-only behaviour and
        # difficulty are diagnostics for the authoring agent, not substitute
        # optimisation targets.  The fixed probe corpus makes this normalized
        # count order-equivalent to the prior integer proxy.
        frontier_indices = [
            index
            for index, row in enumerate(reviewed_rows)
            if _hard_invalid_reason(row) is None
            and _pool_disposition(row) == "accepted"
            and _normalise(row.get("duplicate_outcome") or "unique")
            in {"unique", "novel", "unique_within_source"}
        ]
        accepted_frontier_novel = len(frontier_indices)
        frontier_rows = [reviewed_rows[index] for index in frontier_indices]
        externally_audited = [independent_signals[index] for index in frontier_indices]
        quality_fields = (
            "training_value_label",
            "visual_grounding_outcome",
            "caption_shortcut_risk",
            "distractor_quality",
        )
        pipeline_quality_claims = sum(
            any(field in row for field in quality_fields) for row in frontier_rows
        )
        external_training_values = [_training_value(signals) for signals in externally_audited]
        complete_audits = [_has_complete_quality_audit(signals) for signals in externally_audited]
        training_ready = sum(
            ready and complete
            for (ready, _), complete in zip(external_training_values, complete_audits)
        )
        labelled_frontier = sum(complete_audits)
        answer_positions = Counter(
            position for row in frontier_rows if (position := _answer_position(row))
        )
        task_types = Counter(
            str(row.get("task_type") or "unknown") for row in frontier_rows
        )
        video_counts = Counter(
            key for row in frontier_rows if (key := _video_identity(row))
        )
        repeated_videos = {key: count for key, count in video_counts.items() if count > 1}
        answer_position_summary = _distribution_summary(
            answer_positions, expected=("A", "B", "C", "D")
        )
        # This is a feedback flag, not a gate.  Small probes may legitimately
        # be imbalanced, but an authoring agent needs to see a strong position
        # prior before it expands that route to a production corpus.
        answer_position_summary["severe_skew"] = bool(
            answer_position_summary["total"] >= 8
            and answer_position_summary["largest_share"] > 0.45
        )
        # Preserve legacy comparability when an older pipeline has no quality
        # labels at all. Once labels are present, use the training-ready subset
        # as the promotion target while retaining all counts for diagnosis.
        if labelled_frontier:
            promotion_count = training_ready
        elif pipeline_quality_claims:
            # Once the pipeline claims to have assessed quality, do not reward
            # those claims as if an independent audit had confirmed them.
            promotion_count = 0
        else:
            promotion_count = accepted_frontier_novel
        score = round(promotion_count / max(1, total), 6)
        passed = valid > 0 and (
            not self.require_independent_quality
            or labelled_frontier == accepted_frontier_novel
        )
        issues = tuple(f"{name}:{count}" for name, count in sorted(hard_invalid.items()))
        return CandidateFeedback(
            score=score,
            passed=passed,
            issues=issues,
            domain_feedback={
                "evaluator": "video-rsi-router-v1",
                "evaluation_model_role": self.model_role,
                "evaluation_model_name": self.model_name,
                "total_candidates": total,
                "hard_valid_candidates": valid,
                "accepted_candidates": accepted,
                "accepted_frontier_novel_count": accepted_frontier_novel,
                "training_ready_frontier_novel_count": training_ready,
                "labelled_frontier_candidate_count": labelled_frontier,
                "promotion_frontier_count": promotion_count,
                "quality_feedback_mode": (
                    "independent_signal_provider"
                    if labelled_frontier
                    else "pipeline_claims_unverified" if pipeline_quality_claims
                    else "yield_only_quality_unverified"
                ),
                "pipeline_quality_claim_count": pipeline_quality_claims,
                "independent_quality_audit_coverage": round(
                    labelled_frontier / max(1, accepted_frontier_novel), 6
                ),
                "route_metrics": {name: dict(values) for name, values in sorted(route.items())},
                "hard_invalid_reasons": dict(sorted(hard_invalid.items())),
                "rejection_reasons": dict(sorted(rejections.items())),
                "text_only_outcomes": dict(sorted(text_only.items())),
                "difficulty_labels": dict(sorted(difficulty.items())),
                "duplicate_outcomes": dict(sorted(duplicate.items())),
                "training_value_labels": dict(sorted(training_value.items())),
                "frame_visibility_outcomes": dict(sorted(frame_visibility.items())),
                "training_pool_diagnostics": {
                    "answer_position": answer_position_summary,
                    "task_type": _distribution_summary(task_types),
                    "same_video": {
                        "identified_videos": len(video_counts),
                        "repeated_video_count": len(repeated_videos),
                        "correlated_variant_count": sum(count - 1 for count in repeated_videos.values()),
                        "max_variants_per_video": max(video_counts.values(), default=0),
                    },
                    "labelled_frontier_coverage": round(
                        labelled_frontier / max(1, accepted_frontier_novel), 6
                    ),
                },
                "signal_provider_failures": signal_failures,
            },
        )
