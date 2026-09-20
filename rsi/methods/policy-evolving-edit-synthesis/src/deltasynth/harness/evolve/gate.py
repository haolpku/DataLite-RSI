from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .state import Hypothesis, Policy, StrategyState
from .stats import RunStats


@dataclass(slots=True)
class GateConfig:
    min_support_batches: int = 2
    max_policies: int = 3
    retire_after_untested: int = 2


@dataclass(slots=True)
class GateVerdict:
    passed: bool
    reasons: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.passed


def evaluate(
    hypothesis: Hypothesis,
    state: StrategyState,
    config: Optional[GateConfig] = None,
) -> GateVerdict:
    cfg = config or GateConfig()
    reasons: list[str] = []

    if hypothesis.status == "retired":
        reasons.append("the claim behind it has been retired")
    if hypothesis.support < cfg.min_support_batches:
        reasons.append(
            f"supported in {hypothesis.support} batch(es), needs "
            f"{cfg.min_support_batches}"
        )
    if hypothesis.oppose >= hypothesis.support:
        reasons.append(
            f"contradicted in {hypothesis.oppose} batch(es) against "
            f"{hypothesis.support} supporting; not settled"
        )
    non_diversity = [p for p in state.active_policies() if not p.is_diversity]
    if len(non_diversity) >= cfg.max_policies:
        reasons.append(
            f"the library already holds {cfg.max_policies} active policies; "
            f"remove one before adding another"
        )

    return GateVerdict(passed=not reasons, reasons=reasons)


def promote(
    hypothesis: Hypothesis,
    *,
    policy: Policy,
    stats: RunStats,  # noqa: ARG001 — kept so callers need not care what it needs
    batch_id: str,
) -> Policy:
    policy.active = True
    policy.deployed_at_batch = batch_id
    policy.source_hypothesis = hypothesis.hypothesis_id
    policy.evidence_batches = list(hypothesis.support_batches)
    hypothesis.status = "promoted"
    return policy


def retire_stale_hypotheses(
    state: StrategyState, config: Optional[GateConfig] = None
) -> list[str]:
    cfg = config or GateConfig()
    retired: list[str] = []
    for hypothesis in state.hypotheses:
        if hypothesis.status != "open":
            continue
        if hypothesis.untested_streak < cfg.retire_after_untested:
            continue
        hypothesis.status = "retired"
        hypothesis.retired_reason = (
            f"no batch could test it for {hypothesis.untested_streak} batches running"
        )
        retired.append(f"{hypothesis.hypothesis_id}: {hypothesis.retired_reason}")
    return retired
