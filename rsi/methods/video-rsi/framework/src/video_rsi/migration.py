"""Feedback-to-pipeline migration plans for the autonomous evolver.

The planner does not edit code or decide promotion.  It converts structured
failure feedback into explicit, auditable actions that Codex can implement in
a candidate workspace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .pipeline_graph import PipelineGraph


@dataclass(frozen=True)
class FailureSignal:
    task_type: str
    reason: str
    count: int = 1
    operator: str | None = None
    examples: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FailureSignal":
        return cls(
            task_type=str(value.get("task_type", "generic")),
            reason=str(value.get("reason", "unknown")),
            count=int(value.get("count", 1)),
            operator=value.get("operator"),
            examples=tuple(value.get("examples", ())),
        )


@dataclass(frozen=True)
class MigrationAction:
    action: str
    task_type: str
    target: str
    rationale: str
    source: str | None = None
    required_capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class MigrationPlan:
    source_pipeline: str
    actions: tuple[MigrationAction, ...]
    structural_required: bool
    feedback_digest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_pipeline": self.source_pipeline,
            "structural_required": self.structural_required,
            "actions": [
                {
                    "action": item.action,
                    "task_type": item.task_type,
                    "target": item.target,
                    "source": item.source,
                    "rationale": item.rationale,
                    "required_capabilities": list(item.required_capabilities),
                }
                for item in self.actions
            ],
            "feedback_digest": self.feedback_digest,
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target


class PipelineMigrationPlanner:
    """Turn per-task/operator failures into bounded Lego-style actions."""

    _STRUCTURAL_REASONS = {"too_easy", "insufficient_evidence", "identity_failure", "answer_ambiguity"}

    def plan(
        self,
        source: PipelineGraph,
        signals: Iterable[FailureSignal | Mapping[str, Any]],
        *,
        prior_change_scopes: Iterable[str] = (),
    ) -> MigrationPlan:
        normalized = [s if isinstance(s, FailureSignal) else FailureSignal.from_dict(s) for s in signals]
        scopes = set(prior_change_scopes)
        by_task: dict[str, list[FailureSignal]] = {}
        for signal in normalized:
            by_task.setdefault(signal.task_type, []).append(signal)

        actions: list[MigrationAction] = []
        structural_required = "pipeline_graph" in scopes or "operator" in scopes
        for task_type, task_signals in sorted(by_task.items()):
            total = sum(max(1, s.count) for s in task_signals)
            reasons = {s.reason for s in task_signals}
            # A task with recurring failures deserves its own graph rather than
            # another global patch.  The threshold is intentionally modest for
            # probe runs and can be changed by experiment configuration.
            if task_type != "generic" and total >= 2:
                actions.append(MigrationAction(
                    action="fork_pipeline",
                    task_type=task_type,
                    target=f"{task_type}_pipeline_next",
                    source=f"{source.pipeline_id}@{source.version}",
                    rationale=f"{total} failures are concentrated in {task_type}; isolate its evidence and question policy",
                ))
                structural_required = True
            for signal in sorted(task_signals, key=lambda x: (-x.count, x.reason)):
                operator = signal.operator or "task-specific stage"
                if signal.reason in {"too_easy", "question_too_local"}:
                    actions.append(MigrationAction(
                        action="replace_or_insert",
                        task_type=task_type,
                        target="multi_observation_evidence_selector",
                        source=operator,
                        rationale="force a question to depend on explicit cross-observation evidence",
                        required_capabilities=("cross_time_grounding",),
                    ))
                    structural_required = True
                elif signal.reason in {"insufficient_evidence", "cannot_determine", "identity_failure"}:
                    actions.append(MigrationAction(
                        action="replace_or_insert",
                        task_type=task_type,
                        target="task_evidence_validator",
                        source=operator,
                        rationale="validate entity identity and evidence sufficiency before generation",
                        required_capabilities=("evidence_validation",),
                    ))
                    structural_required = True
                elif signal.reason in {"answer_ambiguity", "answer_leak"}:
                    actions.append(MigrationAction(
                        action="update_task_contract",
                        task_type=task_type,
                        target="TaskSpec",
                        source=operator,
                        rationale="tighten answer uniqueness and leakage constraints at the task level",
                    ))
                elif signal.reason in {"distractor_invalid", "distractor_leak"}:
                    actions.append(MigrationAction(
                        action="replace_or_insert",
                        task_type=task_type,
                        target="task_distractor_generator",
                        source=operator,
                        rationale="construct independently plausible alternatives without answer repetition",
                        required_capabilities=("option_independence",),
                    ))

        # If previous rounds only changed wording, force at least one graph or
        # operator action so evolution cannot become an endless prompt loop.
        if not structural_required and ("prompt" in scopes or "task_spec" in scopes):
            actions.append(MigrationAction(
                action="explore_structure",
                task_type=source.task_type,
                target="new_task_specific_pipeline",
                source=f"{source.pipeline_id}@{source.version}",
                rationale="previous changes were non-structural; perform a Lego-level graph exploration",
            ))
            structural_required = True

        digest = {
            "signal_count": len(normalized),
            "failure_count": sum(max(1, s.count) for s in normalized),
            "tasks": sorted(by_task),
            "reasons": sorted({s.reason for s in normalized}),
        }
        return MigrationPlan(source_pipeline=f"{source.pipeline_id}@{source.version}", actions=tuple(actions), structural_required=structural_required, feedback_digest=digest)

