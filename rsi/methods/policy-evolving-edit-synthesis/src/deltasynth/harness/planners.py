from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from ..environment import load_environment_snapshot
from .campaign_spec import CampaignSpec
from .coverage import CoverageReport, SliceKey
from .ledger import LedgerRow
from .policy import DataAgentPolicy


log = logging.getLogger("deltasynth.harness")

DEFAULT_ENVIRONMENT_SNAPSHOT = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "configs"
    / "environment"
    / "snapshot.json"
)

_AXIS_FROM_PAIR = {
    "taxonomy": "edit_id",
    "category": "category",
    "style": "style",
    "composition": "composition",
}


@dataclass(frozen=True, slots=True)
class EnvironmentPair:

    scene_id: str
    edit_id: str
    scene_ref: str
    edit_ref: str
    base_prompt: str
    category: str
    style: str
    composition: str
    lighting: str
    subcategory: str

    @property
    def pair_id(self) -> str:
        return f"{self.edit_id}::{self.scene_id}"


def load_environment_pairs(
    snapshot_path: Path = DEFAULT_ENVIRONMENT_SNAPSHOT,
) -> list[EnvironmentPair]:
    _, scenes, edits, compatibility = load_environment_snapshot(snapshot_path)
    scene_by_ref = {scene.ref: scene for scene in scenes}
    edit_by_id = {edit.edit_id: edit for edit in edits}

    pairs: list[EnvironmentPair] = []
    for record in compatibility:
        edit = edit_by_id.get(record.edit_ref)
        if edit is None:
            continue
        for scene_ref in record.allowed_scene_refs:
            scene = scene_by_ref.get(scene_ref)
            if scene is None:
                continue
            pairs.append(
                EnvironmentPair(
                    scene_id=scene.scene_id,
                    edit_id=edit.edit_id,
                    scene_ref=scene.ref,
                    edit_ref=edit.ref,
                    base_prompt=scene.prompt,
                    category=scene.category,
                    style=scene.style,
                    composition=scene.composition or "unspecified",
                    lighting=scene.lighting,
                    subcategory=edit.subcategory,
                )
            )
    pairs.sort(key=lambda pair: (pair.edit_id, pair.scene_id))
    return pairs


@dataclass(slots=True)
class PlannerServices:

    storage_root: Optional[Path] = None
    llm: Optional[Any] = None


@dataclass(slots=True)
class BatchJob:

    pair_id: str
    scene_id: str
    edit_code: str
    slice_id: str
    base_prompt: str
    category: str
    style: str
    composition: str
    lighting: str
    rewrite_parent_id: Optional[str] = None
    rewrite_failure_tags: list[str] = field(default_factory=list)


class BatchPlanner(ABC):
    setting_id: str = "S?"

    needs_generated_before: bool = False

    def __init__(
        self,
        spec: CampaignSpec,
        pairs: list[EnvironmentPair],
        rng: random.Random,
        *,
        services: Optional[PlannerServices] = None,
    ) -> None:
        if not pairs:
            raise ValueError(
                "candidate pair pool is empty; check the environment snapshot"
            )
        self.spec = spec
        self.pairs = pairs
        self.rng = rng
        self.coverage_axes = spec.coverage_axes
        self.services = services

    @abstractmethod
    def plan(
        self,
        *,
        batch_id: int,
        ledger_rows: list[LedgerRow],
        coverage: Optional[CoverageReport],
        budget_remaining: int,
    ) -> list[BatchJob]: ...

    def is_exhausted(self) -> bool:
        return False


    def prepare(self, instruction_planner: Any) -> None:
        pass

    def observe(
        self,
        *,
        batch_id: str,
        run_tag: str,
        sample_ids: list[str],
        ledger_rows: list[LedgerRow],
        storage: Any,
        ctx: Any,
    ) -> None:
        pass


    def _slice_id(self, pair: EnvironmentPair) -> str:
        parts: list[str] = []
        for axis in self.coverage_axes:
            attr = _AXIS_FROM_PAIR.get(axis)
            parts.append(str(getattr(pair, attr)) if attr else "∅")
        return "/".join(parts)

    def _slice_key(self, pair: EnvironmentPair) -> SliceKey:
        return tuple(self._slice_id(pair).split("/"))

    def _job(self, pair: EnvironmentPair) -> BatchJob:
        return BatchJob(
            pair_id=pair.pair_id,
            scene_id=pair.scene_id,
            edit_code=pair.edit_id,
            slice_id=self._slice_id(pair),
            base_prompt=pair.base_prompt,
            category=pair.category,
            style=pair.style,
            composition=pair.composition,
            lighting=pair.lighting,
        )

    def _batch_size(self, budget_remaining: int) -> int:
        return min(self.spec.budget.batch_size, max(0, budget_remaining))


class RandomPlanner(BatchPlanner):

    setting_id = "random"

    def plan(
        self,
        *,
        batch_id: int,  # noqa: ARG002
        ledger_rows: list[LedgerRow],  # noqa: ARG002
        coverage: Optional[CoverageReport],  # noqa: ARG002
        budget_remaining: int,
    ) -> list[BatchJob]:
        n = self._batch_size(budget_remaining)
        return [self._job(self.rng.choice(self.pairs)) for _ in range(n)]


def settled_pair_ids(ledger_rows: list[LedgerRow]) -> set[str]:
    settled: set[str] = set()
    for row in ledger_rows:
        if not row.pair_id:
            continue
        if row.judgment in ("GoodCase", "BadCase"):
            settled.add(row.pair_id)
        elif row.environment_valid is False:
            settled.add(row.pair_id)
    return settled


class StratifiedPlanner(BatchPlanner):

    setting_id = "stratified"

    def __init__(
        self,
        spec: CampaignSpec,
        pairs: list[EnvironmentPair],
        rng: random.Random,
        *,
        allow_repeat_epochs: bool = False,
        skip_pairs: Optional[set[str]] = None,
        services: Optional[PlannerServices] = None,
    ) -> None:
        super().__init__(spec, pairs, rng, services=services)
        self.allow_repeat_epochs = allow_repeat_epochs
        self.skip_pairs: set[str] = set(skip_pairs or ())
        self._reported_exhausted = False
        self._by_edit_category: dict[tuple[str, str], list[EnvironmentPair]] = (
            defaultdict(list)
        )
        for pair in pairs:
            self._by_edit_category[(pair.edit_id, pair.category)].append(pair)
        self._categories_by_edit: dict[str, list[str]] = defaultdict(list)
        for edit_id, category in sorted(self._by_edit_category):
            self._categories_by_edit[edit_id].append(category)
        self._pending: deque[EnvironmentPair] = deque()
        self.epoch = 0

    def _build_epoch(self) -> list[EnvironmentPair]:
        queues = {
            key: self._shuffled(list(items))
            for key, items in self._by_edit_category.items()
        }
        edit_order = sorted(self._categories_by_edit)
        self.rng.shuffle(edit_order)
        cursor = {
            edit_id: self.rng.randrange(len(self._categories_by_edit[edit_id]))
            for edit_id in edit_order
        }
        remaining = sum(len(items) for items in queues.values())
        out: list[EnvironmentPair] = []
        while remaining > 0:
            for edit_id in edit_order:
                categories = self._categories_by_edit[edit_id]
                for _ in range(len(categories)):
                    category = categories[cursor[edit_id] % len(categories)]
                    cursor[edit_id] += 1
                    queue = queues[(edit_id, category)]
                    if queue:
                        out.append(queue.pop())
                        remaining -= 1
                        break
        return out

    def _shuffled(self, items: list[EnvironmentPair]) -> list[EnvironmentPair]:
        copied = list(items)
        self.rng.shuffle(copied)
        return copied

    def next_pairs(self, n: int) -> list[EnvironmentPair]:
        out: list[EnvironmentPair] = []
        while len(out) < n:
            if not self._pending:
                if self.epoch >= 1 and not self.allow_repeat_epochs:
                    self._warn_exhausted()
                    break
                self._pending = deque(self._build_epoch())
                self.epoch += 1
                if not self._pending:
                    self._warn_exhausted()
                    break
            pair = self._pending.popleft()
            if pair.pair_id in self.skip_pairs:
                continue
            out.append(pair)
        return out

    def _warn_exhausted(self) -> None:
        if self._reported_exhausted:
            return
        self._reported_exhausted = True
        log.warning(
            "[%s] the single pass over %d pairs is spent; by design this arm does "
            "not open a second one. If the GoodCase target is still short, the "
            "measured pass rate is below expectation and the pool needs enlarging "
            "rather than resampling across epochs.",
            self.setting_id,
            len(self.pairs),
        )

    def is_exhausted(self) -> bool:
        return (
            not self.allow_repeat_epochs
            and self.epoch >= 1
            and not self._pending
        )

    def plan(
        self,
        *,
        batch_id: int,  # noqa: ARG002
        ledger_rows: list[LedgerRow],
        coverage: Optional[CoverageReport],  # noqa: ARG002
        budget_remaining: int,
    ) -> list[BatchJob]:
        self._adopt_ledger(ledger_rows)
        n = self._batch_size(budget_remaining)
        return [self._job(pair) for pair in self.next_pairs(n)]

    def _adopt_ledger(self, ledger_rows: list[LedgerRow]) -> None:
        if self.epoch > 0 or not ledger_rows:
            return
        settled = settled_pair_ids(ledger_rows)
        if not settled:
            return
        self.skip_pairs |= settled
        log.info(
            "[%s] resuming: %d pairs already settled in the ledger will be "
            "stepped over, %d remain on this pass",
            self.setting_id,
            len(settled),
            sum(
                1 for pair in self.pairs if pair.pair_id not in self.skip_pairs
            ),
        )


PLANNERS: dict[str, type[BatchPlanner]] = {
    "random": RandomPlanner,
    "stratified": StratifiedPlanner,
}

EVOLVING_SETTINGS = frozenset(
    {"evolving", "evolving-no-rewrite", "policy-evolving"}
)


def resolve_planner_class(setting_id: str) -> type[BatchPlanner]:
    planner_cls = PLANNERS.get(setting_id)
    if planner_cls is not None:
        return planner_cls
    if setting_id in EVOLVING_SETTINGS:
        from .evolve.planner import EvolvingPlanner

        return EvolvingPlanner
    raise ValueError(f"unknown setting_id: {setting_id}")


def _narrow(
    pairs: list[EnvironmentPair],
    wanted: Optional[Iterable[str]],
    *,
    key,
    what: str,
) -> list[EnvironmentPair]:
    keep = {value for value in (wanted or []) if value}
    if not keep:
        return pairs
    narrowed = [pair for pair in pairs if key(pair) in keep]
    if not narrowed:
        raise ValueError(
            f"restrict_{what}={sorted(keep)[:8]}... matches no pair in the "
            "current pool"
        )
    missing = keep - {key(pair) for pair in narrowed}
    if missing:
        log.warning(
            "[planner] restrict_%s names %d item(s) absent from the pool; "
            "ignored: %s",
            what,
            len(missing),
            sorted(missing)[:8],
        )
    return narrowed


def make_planner(
    spec: CampaignSpec,
    *,
    snapshot_path: Optional[Path] = None,
    pairs: Optional[Iterable[EnvironmentPair]] = None,
    rng: Optional[random.Random] = None,
    services: Optional[PlannerServices] = None,
) -> BatchPlanner:
    resolved = (
        list(pairs)
        if pairs is not None
        else load_environment_pairs(snapshot_path or DEFAULT_ENVIRONMENT_SNAPSHOT)
    )
    resolved = _narrow(
        resolved, spec.restrict_edits, key=lambda pair: pair.edit_id, what="edits"
    )
    resolved = _narrow(
        resolved, spec.restrict_scenes, key=lambda pair: pair.scene_id, what="scenes"
    )
    planner_cls = resolve_planner_class(spec.setting_id)
    return planner_cls(
        spec,
        resolved,
        rng or random.Random(spec.random_seeds[0]),
        services=services,
    )
