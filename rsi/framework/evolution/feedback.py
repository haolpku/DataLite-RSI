"""Candidate evaluator interface for the DataFlow incumbent/challenger loop.

The default text implementation is ReviewAgent. Image, video and mixed-task
evaluators implement the same result contract while keeping their own scoring
and evidence rules outside the loop controller.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


from .models import ReviewResult, TaskSpec


@dataclass(frozen=True)
class CandidateFeedback:
    """Comparable task-owned score with independent domain evidence."""

    score: float
    passed: bool
    issues: tuple[str, ...] = ()
    domain_feedback: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("candidate score must be a finite number in [0, 1]")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("candidate score must be a finite number in [0, 1]")
        if not isinstance(self.passed, bool):
            raise TypeError("candidate passed must be a boolean")
        if not isinstance(self.domain_feedback, Mapping):
            raise TypeError("candidate domain_feedback must be a mapping")
        try:
            json.dumps(dict(self.domain_feedback), ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise TypeError("candidate domain_feedback must be JSON-serializable") from exc


class CandidateEvaluator(Protocol):
    def review(self, dataset_path: str, task: TaskSpec, **kwargs: Any) -> CandidateFeedback | ReviewResult:
        """Evaluate one completed candidate and return a comparable result."""
        ...


def normalize_candidate_review(value: CandidateFeedback | ReviewResult) -> ReviewResult:
    if isinstance(value, CandidateFeedback):
        return ReviewResult(
            schema_score=0.0,
            relevance_score=0.0,
            passed=value.passed,
            review_score=float(value.score),
            issues=list(value.issues),
            domain_feedback=dict(value.domain_feedback),
            evaluator_kind="task",
        )
    if not isinstance(value, ReviewResult):
        raise TypeError("candidate evaluator must return CandidateFeedback or ReviewResult")
    if not math.isfinite(value.review_score) or not 0.0 <= value.review_score <= 1.0:
        raise ValueError("candidate review_score must be a finite number in [0, 1]")
    return value
