from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ..core.schema import Judgment, Sample
from ..core.storage import DirStorage


@dataclass(slots=True)
class LedgerRow:

    sample_id: str
    setting_id: Optional[str] = None
    batch_id: Optional[str] = None
    slice_id: Optional[str] = None
    seed_id: Optional[str] = None
    pair_id: Optional[str] = None
    scene_id: Optional[str] = None
    taxonomy: Optional[str] = None
    category: Optional[str] = None
    style: Optional[str] = None
    composition: Optional[str] = None
    generator: Optional[str] = None
    cost_tokens: Optional[int] = None
    api_calls: int = 0
    latency_s: Optional[float] = None
    verifier_aggregate: Optional[float] = None
    verifier_axes: dict[str, float] = field(default_factory=dict)
    failure_tags: list[str] = field(default_factory=list)
    judgment: str = "Pending"
    is_goodcase: bool = False
    environment_valid: Optional[bool] = None
    rewrite_parent_id: Optional[str] = None
    parent_sample_id: Optional[str] = None
    created_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "setting_id": self.setting_id,
            "batch_id": self.batch_id,
            "slice_id": self.slice_id,
            "seed_id": self.seed_id,
            "pair_id": self.pair_id,
            "scene_id": self.scene_id,
            "taxonomy": self.taxonomy,
            "category": self.category,
            "style": self.style,
            "composition": self.composition,
            "generator": self.generator,
            "cost_tokens": self.cost_tokens,
            "api_calls": self.api_calls,
            "latency_s": self.latency_s,
            "verifier_aggregate": self.verifier_aggregate,
            "verifier_axes": self.verifier_axes,
            "failure_tags": self.failure_tags,
            "judgment": self.judgment,
            "is_goodcase": self.is_goodcase,
            "environment_valid": self.environment_valid,
            "rewrite_parent_id": self.rewrite_parent_id,
            "parent_sample_id": self.parent_sample_id,
            "created_at": self.created_at,
        }


def project_sample(sample: Sample) -> LedgerRow:
    payload = sample.meta.pair_meta or {}
    seed_id = payload.get("seed_id")
    scene_id = payload.get("scene_id")
    taxonomy = (
        sample.edit_codes[0] if sample.edit_codes else payload.get("category_hint")
    )
    category = payload.get("category") or payload.get("category_hint")
    style = payload.get("style") or payload.get("style_hint")
    composition = payload.get("composition")
    pair_id = payload.get("pair_id")
    if pair_id is None and taxonomy and scene_id:
        pair_id = f"{taxonomy}::{scene_id}"

    cost_tokens = sample.cost.total_tokens
    api_calls = len(sample.cost.steps)
    latency_s = sample.cost.latency_total_s

    generator = None
    for step in sample.cost.steps:
        if step.step and step.step.lower() in {"b_generations", "c_edits", "image_gen", "edit_apply"}:
            generator = step.model
            break
        if step.model and "image" in (step.model or "").lower():
            generator = step.model
            break
    if generator is None:
        generator = sample.meta.model_edit or sample.meta.model_before or sample.meta.model

    v = sample.verification
    verifier_aggregate = v.aggregate if v else None
    verifier_axes = (
        {k: ax.score for k, ax in (v.axes or {}).items()} if v else {}
    )
    failure_tags = list(v.failure_tags) if v else []
    v_judgment = v.judgment if v else None
    judgment_val = (v_judgment or sample.meta.judgment or Judgment.PENDING).value

    is_goodcase = judgment_val == Judgment.GOOD.value

    environment_valid: Optional[bool] = None
    first_step = sample.steps[0] if sample.steps else None
    if first_step is not None and "environment_valid" in first_step.raw:
        raw_valid = first_step.raw.get("environment_valid")
        if isinstance(raw_valid, bool):
            environment_valid = raw_valid
    if environment_valid is None and v is not None:
        report_valid = getattr(v, "environment_valid", None)
        if isinstance(report_valid, bool):
            environment_valid = report_valid

    created = sample.meta.created_at.isoformat() if sample.meta.created_at else None

    return LedgerRow(
        sample_id=sample.sample_id,
        setting_id=sample.meta.setting_id,
        batch_id=sample.meta.batch_id,
        slice_id=sample.meta.slice_id,
        seed_id=str(seed_id) if seed_id is not None else None,
        pair_id=str(pair_id) if pair_id is not None else None,
        scene_id=str(scene_id) if scene_id is not None else None,
        taxonomy=str(taxonomy) if taxonomy is not None else None,
        category=str(category) if category is not None else None,
        style=str(style) if style is not None else None,
        composition=str(composition) if composition is not None else None,
        generator=generator,
        cost_tokens=cost_tokens,
        api_calls=api_calls,
        latency_s=latency_s,
        verifier_aggregate=verifier_aggregate,
        verifier_axes=verifier_axes,
        failure_tags=failure_tags,
        judgment=judgment_val,
        is_goodcase=is_goodcase,
        environment_valid=environment_valid,
        rewrite_parent_id=sample.rewrite_parent_id,
        parent_sample_id=sample.parent_sample_id,
        created_at=created,
    )


class RunLedger:

    def __init__(self, storage: DirStorage) -> None:
        self.storage = storage


    def scan(self, *, setting_id: Optional[str] = None) -> list[LedgerRow]:
        rows: list[LedgerRow] = []
        for sid in self.storage.list_samples():
            try:
                sample = self.storage.read_sample(sid)
            except Exception:  # noqa: BLE001 — skip broken samples on scan
                continue
            row = project_sample(sample)
            if setting_id is None or row.setting_id == setting_id:
                rows.append(row)
        return rows


    @staticmethod
    def goodcase_count(rows: Iterable[LedgerRow]) -> int:
        return sum(1 for r in rows if r.is_goodcase)

    @staticmethod
    def total_api_calls(rows: Iterable[LedgerRow]) -> int:
        return sum(r.api_calls for r in rows)

    @staticmethod
    def total_tokens(rows: Iterable[LedgerRow]) -> int:
        return sum((r.cost_tokens or 0) for r in rows)

    @classmethod
    def goodcase_per_call(cls, rows: Iterable[LedgerRow]) -> Optional[float]:
        rows = list(rows)
        calls = cls.total_api_calls(rows)
        if calls <= 0:
            return None
        return cls.goodcase_count(rows) / calls


    @staticmethod
    def unique_pairs(rows: Iterable[LedgerRow]) -> int:
        return len({r.pair_id for r in rows if r.pair_id})

    @staticmethod
    def episodes_per_pair(rows: Iterable[LedgerRow]) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for r in rows:
            if r.pair_id:
                counts[r.pair_id] += 1
        return dict(counts)

    @classmethod
    def redundancy(cls, rows: Iterable[LedgerRow]) -> dict[str, Any]:
        counts = cls.episodes_per_pair(rows)
        if not counts:
            return {"unique_pairs": 0, "max": 0, "mean": None}
        values = sorted(counts.values())
        return {
            "unique_pairs": len(counts),
            "max": values[-1],
            "median": values[len(values) // 2],
            "mean": sum(values) / len(values),
        }


    @staticmethod
    def coverage_entropy(
        rows: Iterable[LedgerRow], axis: str = "taxonomy"
    ) -> float:
        counts: Counter[str] = Counter()
        for r in rows:
            v = getattr(r, axis, None)
            if v is None:
                continue
            counts[str(v)] += 1
        n = sum(counts.values())
        if n <= 0 or len(counts) <= 1:
            return 0.0
        h = -sum((c / n) * math.log(c / n) for c in counts.values())
        return h / math.log(len(counts))


    @classmethod
    def repair_rate(cls, rows: Iterable[LedgerRow]) -> Optional[float]:
        rows = list(rows)
        index = {r.sample_id: r for r in rows}
        attempts = 0
        successes = 0
        for r in rows:
            if r.rewrite_parent_id is None:
                continue
            parent = index.get(r.rewrite_parent_id)
            if parent is None:
                continue
            if parent.is_goodcase:
                continue
            attempts += 1
            if r.is_goodcase:
                successes += 1
        if attempts == 0:
            return None
        return successes / attempts


    @staticmethod
    def time_to_n_goodcase(
        rows: Iterable[LedgerRow], n: int
    ) -> Optional[float]:
        sorted_rows = sorted(
            (r for r in rows if r.created_at is not None),
            key=lambda r: r.created_at or "",
        )
        cum = 0.0
        good = 0
        for r in sorted_rows:
            cum += r.latency_s or 0.0
            if r.is_goodcase:
                good += 1
                if good >= n:
                    return cum
        return None


    @staticmethod
    def top_failure_tags(
        rows: Iterable[LedgerRow], top_k: int = 5
    ) -> list[tuple[str, int]]:
        c: Counter[str] = Counter()
        for r in rows:
            for t in r.failure_tags:
                c[t] += 1
        return c.most_common(top_k)


    @classmethod
    def summary(
        cls,
        rows: Iterable[LedgerRow],
        *,
        setting_id: Optional[str] = None,
        coverage_axis: str = "taxonomy",
        target_goodcase: int = 100,
    ) -> dict[str, Any]:
        rows = [r for r in rows if setting_id is None or r.setting_id == setting_id]
        return {
            "setting_id": setting_id,
            "samples": len(rows),
            "goodcase": cls.goodcase_count(rows),
            "api_calls": cls.total_api_calls(rows),
            "tokens": cls.total_tokens(rows),
            "goodcase_per_call": cls.goodcase_per_call(rows),
            "coverage_entropy": cls.coverage_entropy(rows, axis=coverage_axis),
            "redundancy": cls.redundancy(rows),
            "repair_rate": cls.repair_rate(rows),
            "time_to_n_goodcase": cls.time_to_n_goodcase(rows, target_goodcase),
            "top_failure_tags": cls.top_failure_tags(rows, top_k=5),
        }
