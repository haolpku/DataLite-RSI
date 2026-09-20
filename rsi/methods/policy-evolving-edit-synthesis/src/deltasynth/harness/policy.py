from __future__ import annotations

import random
from typing import Any, Optional

from .campaign_spec import AgentPolicyConfig

STRUCTURAL_COMPATIBILITY_TAGS = frozenset(
    {
        "already_satisfied",
        "target_absent",
        "ambiguous_target",
        "edit_not_observable",
    }
)


class DataAgentPolicy:

    def __init__(
        self,
        config: Optional[AgentPolicyConfig] = None,
        *,
        rng: Optional[random.Random] = None,
        seed: int = 42,
    ) -> None:
        self.config = config or AgentPolicyConfig()
        self.rng = rng or random.Random(seed)


    def pick_rewrite_targets(
        self,
        rows,
        *,
        max_targets: int,
    ) -> list:
        if self.config.ablation == "no_rewrite":
            return []

        parent_of = {
            row.sample_id: row.rewrite_parent_id for row in rows
            if row.rewrite_parent_id
        }

        def root_of(sample_id: str) -> str:
            seen = {sample_id}
            while sample_id in parent_of:
                sample_id = parent_of[sample_id]
                if sample_id in seen:
                    break
                seen.add(sample_id)
            return sample_id

        solved_tasks = {
            root_of(row.sample_id) for row in rows if row.is_goodcase
        }

        best_per_task: dict[str, Any] = {}
        for row in rows:
            if row.is_goodcase or not row.failure_tags:
                continue
            if set(row.failure_tags) & STRUCTURAL_COMPATIBILITY_TAGS:
                continue
            task = root_of(row.sample_id)
            if task in solved_tasks:
                continue
            incumbent = best_per_task.get(task)
            if incumbent is None or (row.created_at or "") > (
                incumbent.created_at or ""
            ):
                best_per_task[task] = row

        candidates = sorted(
            best_per_task.values(),
            key=lambda r: (r.created_at or ""),
            reverse=True,
        )
        return candidates[:max_targets]
