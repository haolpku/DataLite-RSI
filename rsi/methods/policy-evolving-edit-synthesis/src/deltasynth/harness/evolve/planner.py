from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ...environment import load_environment_snapshot
from ..campaign_spec import CampaignSpec, EvolveConfig
from ..coverage import CoverageReport
from ..ledger import LedgerRow
from ..overhead import OverheadLedger
from ..planners import (
    DEFAULT_ENVIRONMENT_SNAPSHOT,
    BatchJob,
    BatchPlanner,
    EnvironmentPair,
    PlannerServices,
)
from .analyst import AnalysisOutcome, run_analysis
from .committer import CommitOutcome, commit, commit_diversity
from .gate import GateConfig, evaluate, promote, retire_stale_hypotheses
from .render import render_directives, render_memory, strategy_ref
from .state import Policy, StrategyState, StrategyStore
from .stats import RunStats, read_cards


log = logging.getLogger("deltasynth.harness")

EVOLUTION_LOG = "_evolution.jsonl"
ANALYST_LOG = "analyst_calls.jsonl"
MEMORY_NAME = "memory.md"


def _policy_needs_end_states(policy) -> bool:
    text = " ".join(
        [policy.trigger, policy.procedure, policy.boundary]
    ).lower()
    return any(
        marker in text
        for marker in (
            "already used", "already produced", "recent samples", "recent batch",
            "previous batch", "earlier batch", "do not repeat", "avoid repeat",
            "same material", "same end state", "same target",
        )
    )


class EvolvingPlanner(BatchPlanner):

    setting_id = "evolving"

    def __init__(
        self,
        spec: CampaignSpec,
        pairs: list[EnvironmentPair],
        rng: random.Random,
        *,
        services: Optional[PlannerServices] = None,
    ) -> None:
        super().__init__(spec, pairs, rng, services=services)

        if services is None or services.storage_root is None:
            raise ValueError(
                "the evolving arm needs a storage root: its strategy, statistics "
                "and overhead ledger all live under the run's own output "
                "directory so two runs cannot see each other's state"
            )

        edits = {pair.edit_id for pair in pairs}
        if len(edits) != 1:
            raise ValueError(
                f"this arm runs one edit type at a time, but the pool holds "
                f"{len(edits)}: {sorted(edits)}. Narrow the campaign's "
                f"restrict_edits to exactly one edit code."
            )
        self.edit_code = next(iter(edits))

        self.config: EvolveConfig = getattr(spec, "evolve", None) or EvolveConfig()
        self.storage_root = Path(services.storage_root)
        self.llm = services.llm

        self.schedule = sorted(pairs, key=lambda pair: pair.scene_id)
        self.edit_definition = self._load_edit_definition(spec)

        self.store = StrategyStore(self.storage_root)
        self.state: StrategyState = self.store.load(edit_code=self.edit_code)
        self.stats = RunStats.load(self.storage_root)
        self.overhead = OverheadLedger(self.storage_root)
        self.setting_id = spec.setting_id
        self.gate_config = GateConfig(
            min_support_batches=self.config.gate_min_support_batches,
            max_policies=self.config.max_policies,
            retire_after_untested=self.config.retire_after_untested,
        )


    def prepare(self, instruction_planner: Any) -> None:
        directives = render_directives(self.state)
        instruction_planner.set_directives(
            directives,
            strategy_ref=strategy_ref(self.state),
        )
        wants_list = any(
            _policy_needs_end_states(p) for p in self.state.active_policies()
        )
        instruction_planner.set_used_end_states(
            self.stats.end_states if wants_list else []
        )

    def observe(
        self,
        *,
        batch_id: str,
        run_tag: str = "",
        sample_ids: list[str],
        ledger_rows: list[LedgerRow],  # noqa: ARG002 — kept for the hook's contract
        storage: Any,
        ctx: Any,
    ) -> None:
        key = f"{run_tag}:{batch_id}" if run_tag else batch_id
        if self.stats.has_folded(key):
            return

        cards = read_cards(storage, sample_ids)
        cards = [c for c in cards if c.environment_valid is not False]
        self.stats.fold(key, cards)

        next_state = self.state.derive(observed_batch=key)

        analysis = AnalysisOutcome(called=False)
        diversity_outcome = CommitOutcome(called=False)
        commit_outcome = CommitOutcome(called=False)

        try:
            batch_num = int(str(batch_id).rsplit("b", 1)[-1])
        except ValueError:
            batch_num = -1
        is_last_batch = batch_num == self.spec.budget.max_batches - 1

        if self.llm is not None and not is_last_batch:
            analysis = run_analysis(
                llm=self.llm,
                ctx=ctx,
                state=next_state,
                stats=self.stats,
                cards=cards,
                edit_code=self.edit_code,
                edit_definition=self.edit_definition,
                batch_id=key,
                setting_id=self.spec.setting_id,
            )
            self._charge(key, "evolve_analysis", analysis.cost)
            self._log_call("analysis", key, analysis.prompt, analysis.response,
                           analysis.error)

        retired = retire_stale_hypotheses(next_state, self.gate_config)
        promoted, refused = self._apply_gate(next_state, key)

        diversity_policy = next(
            (p for p in next_state.active_policies() if p.is_diversity), None
        )
        if self.llm is not None and next_state.diversity_converging:
            diversity_outcome = commit_diversity(
                llm=self.llm, ctx=ctx, state=next_state, batch_id=key
            )
            self._charge(key, "evolve_commit_diversity", diversity_outcome.cost)
            self._log_call("commit_diversity", key, diversity_outcome.prompt,
                           diversity_outcome.response, diversity_outcome.error)
            if diversity_outcome.text:
                try:
                    payload = json.loads(diversity_outcome.text)
                    if diversity_policy is None:
                        next_state.policies.append(
                            Policy(
                                policy_id="diversity",
                                trigger=str(payload.get("trigger") or ""),
                                procedure=str(payload.get("procedure") or ""),
                                boundary=str(payload.get("boundary") or ""),
                                active=True,
                                is_diversity=True,
                                deployed_at_batch=key,
                            )
                        )
                    else:
                        diversity_policy.version += 1
                        diversity_policy.trigger = str(
                            payload.get("trigger") or diversity_policy.trigger
                        )
                        diversity_policy.procedure = str(
                            payload.get("procedure") or diversity_policy.procedure
                        )
                        diversity_policy.boundary = str(
                            payload.get("boundary") or diversity_policy.boundary
                        )
                        diversity_policy.deployed_at_batch = key
                except (json.JSONDecodeError, KeyError):
                    pass

        fingerprint = next_state.library_fingerprint()
        if (
            self.llm is not None
            and next_state.active_policies()
            and fingerprint != next_state.rendered_from
        ):
            commit_outcome = commit(
                llm=self.llm, ctx=ctx, state=next_state, batch_id=key
            )
            self._charge(key, "evolve_commit", commit_outcome.cost)
            self._log_call("commit", key, commit_outcome.prompt,
                           commit_outcome.response, commit_outcome.error)
            if commit_outcome.text:
                next_state.rendered_prompt = commit_outcome.text
                next_state.rendered_from = fingerprint
        elif not next_state.active_policies():
            next_state.rendered_prompt = ""
            next_state.rendered_from = fingerprint

        self.state = self.store.commit(next_state, reason=f"batch={key}")
        self.stats.save(self.storage_root)
        self._write_memory()
        self._log_evolution(
            batch_id=key,
            analysis=analysis,
            diversity_outcome=diversity_outcome,
            commit_outcome=commit_outcome,
            promoted=promoted,
            refused=refused,
            retired=retired,
        )


    def plan(
        self,
        *,
        batch_id: int,  # noqa: ARG002 — the cursor comes from the ledger
        ledger_rows: list[LedgerRow],
        coverage: Optional[CoverageReport],  # noqa: ARG002 — selection is not scored
        budget_remaining: int,
    ) -> list[BatchJob]:
        n = self._batch_size(budget_remaining)
        if n <= 0:
            return []
        attempted = {row.pair_id for row in ledger_rows if row.pair_id}
        out: list[BatchJob] = []
        for pair in self.schedule:
            if len(out) >= n:
                break
            if pair.pair_id in attempted:
                continue
            out.append(self._job(pair))
        return out

    def is_exhausted(self) -> bool:
        return False


    def _apply_gate(
        self, state: StrategyState, batch_id: str
    ) -> tuple[list[str], list[str]]:
        promoted: list[str] = []
        refused: list[str] = []
        drafted = [
            p
            for p in state.policies
            if not p.active and not p.removed_reason and not p.is_diversity
        ]
        for policy in drafted:
            hypothesis = state.hypothesis(policy.source_hypothesis)
            if hypothesis is None:
                policy.removed_reason = "no hypothesis behind it to gate on"
                refused.append(f"{policy.policy_id}: {policy.removed_reason}")
                continue
            verdict = evaluate(hypothesis, state, self.gate_config)
            if verdict.passed:
                promote(
                    hypothesis, policy=policy, stats=self.stats, batch_id=batch_id
                )
                promoted.append(policy.policy_id)
            else:
                refused.append(f"{policy.policy_id}: {'; '.join(verdict.reasons)}")
        return promoted, refused


    def _load_edit_definition(self, spec: CampaignSpec) -> str:
        snapshot = (
            Path(spec.environment_snapshot)
            if spec.environment_snapshot
            else DEFAULT_ENVIRONMENT_SNAPSHOT
        )
        _, _, edits, _ = load_environment_snapshot(snapshot)
        for edit in edits:
            if edit.edit_id == self.edit_code:
                return f"{edit.subcategory}: {edit.description}"
        return ""

    def _charge(self, batch_id: str, op: str, cost: Any) -> None:
        if cost is None:
            return
        self.overhead.record(
            op=op, cost=cost, setting_id=self.spec.setting_id, batch_id=batch_id
        )

    def _write_memory(self) -> None:
        path = self.storage_root / "evolve" / MEMORY_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_memory(self.state), encoding="utf-8")

    def _log_call(
        self, stage: str, batch_id: str, prompt: str, response: str, error: str
    ) -> None:
        path = self.storage_root / "evolve" / ANALYST_LOG
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "stage": stage,
                        "batch_id": batch_id,
                        "state_version_in": self.state.version,
                        "error": error,
                        "prompt": prompt,
                        "response": response,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def _log_evolution(
        self,
        *,
        batch_id: str,
        analysis: AnalysisOutcome,
        diversity_outcome: CommitOutcome,
        commit_outcome: CommitOutcome,
        promoted: list[str],
        refused: list[str],
        retired: list[str],
    ) -> None:
        digest = self.stats.digests[-1] if self.stats.digests else None
        row = {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "setting_id": self.spec.setting_id,
            "edit_code": self.edit_code,
            "batch_id": batch_id,
            "state_version": self.state.version,
            "batch": digest.to_dict() if digest else {},
            "pass_rate_overall": self.stats.pass_rate,
            "analysis_called": analysis.called,
            "analysis_changes": analysis.changes,
            "analysis_discarded": analysis.discarded,
            "analysis_error": analysis.error,
            "diversity_commit_called": diversity_outcome.called,
            "diversity_commit_error": diversity_outcome.error,
            "commit_called": commit_outcome.called,
            "commit_error": commit_outcome.error,
            "rendered_prompt_chars": len(self.state.rendered_prompt),
            "diversity_converging": self.state.diversity_converging,
            "diversity_reason": self.state.diversity_reason,
            "diversity_categories": self.state.diversity_categories,
            "promoted_policies": promoted,
            "refused_policies": refused,
            "retired_claims": retired,
            "active_policies": [p.policy_id for p in self.state.active_policies()],
            "open_claims": [h.hypothesis_id for h in self.state.open_hypotheses()],
        }
        path = self.storage_root / EVOLUTION_LOG
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        log.info(
            "[evolve] batch=%s v%d accepted=%s promoted=%s active=%d",
            batch_id,
            self.state.version,
            digest.accepted if digest else 0,
            promoted or "-",
            len(self.state.active_policies()),
        )
