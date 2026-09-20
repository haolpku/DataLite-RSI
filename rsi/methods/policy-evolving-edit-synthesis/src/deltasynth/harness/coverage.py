from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .ledger import LedgerRow


SliceKey = tuple[str, ...]


@dataclass(slots=True)
class SliceStats:
    slice_key: SliceKey
    count: int = 0
    goodcase: int = 0
    api_calls: int = 0
    failure_tags: dict[str, int] = field(default_factory=dict)
    repeated_failure: int = 0
    gap: Optional[int] = None

    @property
    def goodcase_rate(self) -> Optional[float]:
        if self.count <= 0:
            return None
        return self.goodcase / self.count

    @property
    def calls_per_goodcase(self) -> Optional[float]:
        if self.goodcase <= 0:
            return None
        return self.api_calls / self.goodcase


@dataclass(slots=True)
class CoverageReport:

    axes: list[str]
    slices: dict[SliceKey, SliceStats] = field(default_factory=dict)
    entropy_by_axis: dict[str, float] = field(default_factory=dict)
    total: int = 0
    total_goodcase: int = 0

    def covered(self) -> set[SliceKey]:
        return {k for k, s in self.slices.items() if s.count > 0}

    def gaps_ranked(self) -> list[tuple[SliceKey, int]]:
        rows = [(k, s.gap) for k, s in self.slices.items() if s.gap is not None]
        rows.sort(key=lambda t: -t[1])
        return rows


class CoverageTracker:

    def __init__(
        self,
        axes: list[str],
        target_per_slice: Optional[int] = None,
        target_distribution: Optional[dict[SliceKey, int]] = None,
    ) -> None:
        if not axes:
            raise ValueError("CoverageTracker needs at least one axis")
        self.axes = axes
        self.target_per_slice = target_per_slice
        self.target_distribution = target_distribution or {}
        self._last_report: Optional[CoverageReport] = None

    @property
    def last_report(self) -> Optional[CoverageReport]:
        return self._last_report


    def update(self, rows: Iterable[LedgerRow]) -> CoverageReport:
        rows = list(rows)
        slices: dict[SliceKey, SliceStats] = {}
        per_slice_failures: dict[SliceKey, int] = defaultdict(int)

        for r in rows:
            key = self._key_for(r)
            stats = slices.setdefault(key, SliceStats(slice_key=key))
            stats.count += 1
            if r.is_goodcase:
                stats.goodcase += 1
            stats.api_calls += r.api_calls
            if not r.is_goodcase and r.judgment in {"BadCase", "Pending"} and r.failure_tags:
                per_slice_failures[key] += 1
                for t in r.failure_tags:
                    stats.failure_tags[t] = stats.failure_tags.get(t, 0) + 1

        for key, stats in slices.items():
            stats.repeated_failure = max(0, per_slice_failures.get(key, 0) - 1)
            target = self.target_distribution.get(key)
            if target is None and self.target_per_slice is not None:
                target = self.target_per_slice
            if target is not None:
                stats.gap = max(0, target - stats.count)

        report = CoverageReport(
            axes=list(self.axes),
            slices=slices,
            entropy_by_axis={
                ax: self._axis_entropy(rows, ax) for ax in self.axes
            },
            total=len(rows),
            total_goodcase=sum(1 for r in rows if r.is_goodcase),
        )
        self._last_report = report
        return report


    def _key_for(self, row: LedgerRow) -> SliceKey:
        parts: list[str] = []
        for ax in self.axes:
            v = getattr(row, ax, None)
            parts.append("∅" if v is None else str(v))
        return tuple(parts)

    @staticmethod
    def _axis_entropy(rows: list[LedgerRow], axis: str) -> float:
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


    def coverage_gap(self, slice_key: SliceKey) -> int:
        if self._last_report is None:
            return 0
        s = self._last_report.slices.get(slice_key)
        if s is None:
            target = self.target_distribution.get(slice_key)
            if target is None and self.target_per_slice is not None:
                target = self.target_per_slice
            return max(0, target or 0)
        return s.gap or 0

    def goodcase_rate(self, slice_key: SliceKey) -> Optional[float]:
        if self._last_report is None:
            return None
        s = self._last_report.slices.get(slice_key)
        return None if s is None else s.goodcase_rate
