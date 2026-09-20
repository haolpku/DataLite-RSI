from __future__ import annotations

from typing import Any, Optional

from ...core.operator import OperatorABC
from ...core.schema import Judgment


class ThresholdFilter(OperatorABC):
    name = "threshold_filter"

    def __init__(
        self,
        *,
        min_score: float = 3.5,
        axis: str = "aggregate",
    ) -> None:
        super().__init__()
        self.min_score = min_score
        self.axis = axis

    def run(
        self,
        storage,
        ctx,
        *,
        sample_ids: Optional[list[str]] = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> list[str]:
        sample_ids = sample_ids or list(storage.list_samples())
        kept: list[str] = []
        dropped: list[str] = []
        skipped: list[str] = []

        for sid in sample_ids:
            sample = storage.read_sample(sid)
            v = sample.verification
            if v is None:
                skipped.append(sid)
                kept.append(sid)
                continue

            score: Optional[float] = None
            if self.axis == "aggregate":
                score = v.aggregate
            else:
                ax = v.axes.get(self.axis)
                if ax is not None:
                    score = ax.score

            if score is None:
                skipped.append(sid)
                kept.append(sid)
                continue

            if score >= self.min_score:
                sample.meta.judgment = Judgment.GOOD
                if v.judgment != Judgment.GOOD:
                    v.judgment = Judgment.GOOD
                storage.write_sample(sample, update_index=False)
                kept.append(sid)
            else:
                sample.meta.judgment = Judgment.BAD
                if v.judgment != Judgment.BAD:
                    v.judgment = Judgment.BAD
                storage.write_sample(sample, update_index=False)
                dropped.append(sid)

        print(
            f"[{self.name}] axis={self.axis} min_score={self.min_score}  "
            f"kept={len(kept)}  dropped={len(dropped)}  skipped(no_score)={len(skipped)}"
        )
        return kept
