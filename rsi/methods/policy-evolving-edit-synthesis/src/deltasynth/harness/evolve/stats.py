from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

from ...core.schema import Sample
from ...operators.verify.if_vc_vq_verifier import DEFAULT_RUBRIC
from ...rubrics import load_rubric


STATS_NAME = "stats.json"

GATE_REJECT = "gate_reject"
UNCLASSIFIED = "unclassified"

MAX_EXAMPLES = 3


@lru_cache(maxsize=1)
def axis_minimums() -> dict[str, float]:
    try:
        rubric = load_rubric(DEFAULT_RUBRIC)
    except Exception:  # noqa: BLE001 — only used as a fallback
        return {}
    return {axis.axis_id: float(axis.minimum_score) for axis in rubric.axes}


@dataclass(slots=True)
class SampleCard:

    sample_id: str
    batch_id: str
    edit_code: str
    scene_id: str
    category: str = ""
    instruction: str = ""
    target: str = ""
    target_state: str = ""
    judgment: str = "Pending"
    is_goodcase: bool = False
    environment_valid: Optional[bool] = None
    score: Optional[float] = None
    axes: dict[str, float] = field(default_factory=dict)
    axis_reasons: dict[str, str] = field(default_factory=dict)
    axis_failures: list[str] = field(default_factory=list)
    api_calls: int = 0

    @property
    def settled(self) -> bool:
        return self.judgment in ("GoodCase", "BadCase") or (
            self.environment_valid is False
        )

    @property
    def outcomes(self) -> list[str]:
        if self.environment_valid is False:
            return [GATE_REJECT]
        if self.is_goodcase or not self.settled:
            return []
        return list(self.axis_failures) or [UNCLASSIFIED]


def _axis_failures(verification) -> list[str]:
    if verification is None:
        return []
    out: list[str] = []
    for item in getattr(verification, "decision_failures", None) or []:
        text = str(item)
        if text == "environment_invalid":
            continue
        out.append(text.split(":", 1)[1] if ":" in text else text)
    return out


def card_from_sample(sample: Sample) -> SampleCard:
    payload = sample.meta.pair_meta or {}
    step = sample.steps[0] if sample.steps else None
    plan: dict[str, Any] = {}
    environment_valid: Optional[bool] = None
    if step is not None:
        plan = step.raw.get("instruction_plan") or {}
        raw_valid = step.raw.get("environment_valid")
        if isinstance(raw_valid, bool):
            environment_valid = raw_valid

    verification = sample.verification
    axes: dict[str, float] = {}
    reasons: dict[str, str] = {}
    if verification is not None:
        for key, axis in (verification.axes or {}).items():
            axes[key] = axis.score
            if axis.reason:
                reasons[key] = axis.reason
        if environment_valid is None:
            report_valid = getattr(verification, "environment_valid", None)
            if isinstance(report_valid, bool):
                environment_valid = report_valid

    judgment = (
        verification.judgment.value
        if verification is not None and verification.judgment is not None
        else sample.meta.judgment.value
    )

    failures = _axis_failures(verification)
    if not failures and judgment == "BadCase" and axes:
        mins = axis_minimums()
        failures = [a for a, s in axes.items() if a in mins and s < mins[a]]

    score: Optional[float] = None
    if verification is not None:
        raw = getattr(verification, "raw_average", None)
        score = float(raw) if isinstance(raw, (int, float)) else verification.aggregate

    return SampleCard(
        sample_id=sample.sample_id,
        batch_id=sample.meta.batch_id or "",
        edit_code=str((sample.edit_codes or [payload.get("edit_code")])[0] or ""),
        scene_id=str(payload.get("scene_id") or ""),
        category=str(payload.get("category") or ""),
        instruction=(step.instruction.en or "") if step else "",
        target=str(plan.get("target_entity") or ""),
        target_state=str(plan.get("target_state") or ""),
        judgment=judgment,
        is_goodcase=judgment == "GoodCase",
        environment_valid=environment_valid,
        score=score,
        axes=axes,
        axis_reasons=reasons,
        axis_failures=failures,
        api_calls=len(sample.cost.steps),
    )


def read_cards(storage, sample_ids: Iterable[str]) -> list[SampleCard]:
    out: list[SampleCard] = []
    for sid in sample_ids:
        try:
            out.append(card_from_sample(storage.read_sample(sid)))
        except Exception:  # noqa: BLE001 — one unreadable sample is not a batch
            continue
    return out


@dataclass(slots=True)
class AxisBucket:

    axis: str
    count: int = 0
    batches: list[str] = field(default_factory=list)
    score_sum: float = 0.0
    score_n: int = 0
    examples: list[str] = field(default_factory=list)

    @property
    def batch_span(self) -> int:
        return len(self.batches)

    @property
    def mean_score(self) -> Optional[float]:
        return self.score_sum / self.score_n if self.score_n else None

    def observe(self, card: SampleCard, *, batch_id: str) -> None:
        self.count += 1
        if batch_id and batch_id not in self.batches:
            self.batches.append(batch_id)
        if card.score is not None:
            self.score_sum += card.score
            self.score_n += 1
        if len(self.examples) < MAX_EXAMPLES:
            self.examples.append(card.sample_id)
        else:
            self.examples[-1] = card.sample_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "count": self.count,
            "batches": self.batches,
            "score_sum": self.score_sum,
            "score_n": self.score_n,
            "examples": self.examples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AxisBucket":
        return cls(
            axis=str(data.get("axis") or ""),
            count=int(data.get("count") or 0),
            batches=list(data.get("batches") or []),
            score_sum=float(data.get("score_sum") or 0.0),
            score_n=int(data.get("score_n") or 0),
            examples=list(data.get("examples") or []),
        )


@dataclass(slots=True)
class BatchDigest:
    batch_id: str
    samples: int = 0
    settled: int = 0
    accepted: int = 0
    gate_rejects: int = 0
    api_calls: int = 0
    axis_failures: dict[str, int] = field(default_factory=dict)

    @property
    def pass_rate(self) -> Optional[float]:
        return self.accepted / self.settled if self.settled else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "samples": self.samples,
            "settled": self.settled,
            "accepted": self.accepted,
            "gate_rejects": self.gate_rejects,
            "api_calls": self.api_calls,
            "axis_failures": self.axis_failures,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchDigest":
        return cls(
            batch_id=str(data.get("batch_id") or ""),
            samples=int(data.get("samples") or 0),
            settled=int(data.get("settled") or 0),
            accepted=int(data.get("accepted") or 0),
            gate_rejects=int(data.get("gate_rejects") or 0),
            api_calls=int(data.get("api_calls") or 0),
            axis_failures=dict(data.get("axis_failures") or {}),
        )


class RunStats:

    END_STATE_LOG_CAP = 200

    def __init__(self) -> None:
        self.buckets: dict[str, AxisBucket] = {}
        self.digests: list[BatchDigest] = []
        self.end_states: list[str] = []


    def has_folded(self, batch_id: str) -> bool:
        return any(d.batch_id == batch_id for d in self.digests)

    def fold(self, batch_id: str, cards: list[SampleCard]) -> BatchDigest:
        existing = next((d for d in self.digests if d.batch_id == batch_id), None)
        if existing is not None:
            return existing

        digest = BatchDigest(batch_id=batch_id)
        failures: Counter[str] = Counter()
        for card in cards:
            digest.samples += 1
            digest.api_calls += card.api_calls
            if not card.settled:
                continue
            digest.settled += 1
            if card.is_goodcase:
                digest.accepted += 1
            if card.environment_valid is False:
                digest.gate_rejects += 1
            for axis in card.outcomes:
                failures[axis] += 1
                bucket = self.buckets.get(axis)
                if bucket is None:
                    bucket = AxisBucket(axis=axis)
                    self.buckets[axis] = bucket
                bucket.observe(card, batch_id=batch_id)

        digest.axis_failures = dict(failures)
        self.digests.append(digest)
        for card in cards:
            if card.target_state.strip():
                self.end_states.append(card.target_state.strip())
        if len(self.end_states) > self.END_STATE_LOG_CAP:
            self.end_states = self.end_states[-self.END_STATE_LOG_CAP :]
        return digest


    @property
    def batches(self) -> int:
        return len(self.digests)

    @property
    def settled(self) -> int:
        return sum(d.settled for d in self.digests)

    @property
    def accepted(self) -> int:
        return sum(d.accepted for d in self.digests)

    @property
    def api_calls(self) -> int:
        return sum(d.api_calls for d in self.digests)

    @property
    def pass_rate(self) -> Optional[float]:
        return self.accepted / self.settled if self.settled else None

    def failure_table(self) -> list[dict[str, Any]]:
        rows = [
            {
                "axis": b.axis,
                "failures": b.count,
                "batches": b.batch_span,
                "mean_score_when_failing": (
                    round(b.mean_score, 2) if b.mean_score is not None else None
                ),
                "examples": b.examples,
            }
            for b in self.buckets.values()
        ]
        rows.sort(key=lambda r: -r["failures"])
        return rows


    def to_dict(self) -> dict[str, Any]:
        return {
            "buckets": [b.to_dict() for b in self.buckets.values()],
            "digests": [d.to_dict() for d in self.digests],
            "end_states": self.end_states,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunStats":
        stats = cls()
        for raw in data.get("buckets") or []:
            bucket = AxisBucket.from_dict(raw)
            stats.buckets[bucket.axis] = bucket
        stats.digests = [
            BatchDigest.from_dict(raw) for raw in (data.get("digests") or [])
        ]
        stats.end_states = list(data.get("end_states") or [])
        return stats

    def save(self, storage_root: Path) -> Path:
        path = Path(storage_root) / "evolve" / STATS_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, storage_root: Path) -> "RunStats":
        path = Path(storage_root) / "evolve" / STATS_NAME
        if not path.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return cls()
