from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
import json
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


STATE_DIRNAME = "evolve"
ACTIVE_NAME = "state_active.json"
HISTORY_NAME = "state_history.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(text: str, *, fallback: str = "item", limit: int = 60) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (cleaned[:limit] or fallback)


Verdict = Literal["support", "oppose", "not_tested"]


class Hypothesis(BaseModel):

    model_config = ConfigDict(extra="allow")

    hypothesis_id: str
    statement: str

    support_batches: list[str] = Field(default_factory=list)
    oppose_batches: list[str] = Field(default_factory=list)
    untested_streak: int = 0

    status: Literal["open", "promoted", "retired"] = "open"
    created_batch: str = ""
    last_update_batch: str = ""
    retired_reason: str = ""

    @property
    def support(self) -> int:
        return len(self.support_batches)

    @property
    def oppose(self) -> int:
        return len(self.oppose_batches)

    @property
    def batches(self) -> list[str]:
        seen = list(self.support_batches)
        for batch in self.oppose_batches:
            if batch not in seen:
                seen.append(batch)
        return seen

    def observe(self, *, batch_id: str, verdict: Verdict) -> None:
        if not batch_id:
            return
        self.support_batches = [b for b in self.support_batches if b != batch_id]
        self.oppose_batches = [b for b in self.oppose_batches if b != batch_id]

        if verdict == "support":
            self.support_batches.append(batch_id)
            self.untested_streak = 0
        elif verdict == "oppose":
            self.oppose_batches.append(batch_id)
            self.untested_streak = 0
        else:
            self.untested_streak += 1
        self.last_update_batch = batch_id


class Policy(BaseModel):

    model_config = ConfigDict(extra="allow")

    policy_id: str
    version: int = 1
    trigger: str
    procedure: str
    boundary: str = ""

    source_hypothesis: str = ""
    evidence_batches: list[str] = Field(default_factory=list)
    active: bool = True
    is_diversity: bool = False

    deployed_at_batch: str = ""
    removed_reason: str = ""


    def render(self) -> str:
        trigger = self.trigger.strip().rstrip(".")
        procedure = self.procedure.strip().rstrip(".")
        out = f"{trigger} -> {procedure}."
        boundary = self.boundary.strip().rstrip(".")
        if boundary:
            out += f" Exception: {boundary}."
        return out


class StrategyState(BaseModel):

    model_config = ConfigDict(extra="allow")

    version: int = 1
    parent_version: Optional[int] = None
    created_at: str = Field(default_factory=_now)
    observed_batch: str = ""
    edit_code: str = ""

    hypotheses: list[Hypothesis] = Field(default_factory=list)
    policies: list[Policy] = Field(default_factory=list)

    diagnosis: str = ""
    rationale: str = ""

    diversity_converging: Optional[bool] = None
    diversity_reason: str = ""
    diversity_categories: dict[str, int] = Field(default_factory=dict)

    rendered_prompt: str = ""
    rendered_from: str = ""


    def active_policies(self) -> list[Policy]:
        return [p for p in self.policies if p.active]

    def library_fingerprint(self) -> str:
        return json.dumps(
            [
                [p.policy_id, p.version, p.trigger, p.procedure, p.boundary]
                for p in self.active_policies()
            ],
            ensure_ascii=False,
            sort_keys=True,
        )

    def policy(self, policy_id: str) -> Optional[Policy]:
        return next((p for p in self.policies if p.policy_id == policy_id), None)

    def hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        return next(
            (h for h in self.hypotheses if h.hypothesis_id == hypothesis_id), None
        )

    def open_hypotheses(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses if h.status == "open"]


    def derive(self, *, observed_batch: str) -> "StrategyState":
        child = StrategyState.model_validate(self.model_dump())
        child.version = self.version + 1
        child.parent_version = self.version
        child.created_at = _now()
        child.observed_batch = observed_batch
        child.diagnosis = ""
        child.rationale = ""
        return child


def seed_state(edit_code: str = "") -> StrategyState:
    return StrategyState(edit_code=edit_code)


class StrategyStore:

    def __init__(self, storage_root: Path) -> None:
        self.dir = Path(storage_root) / STATE_DIRNAME

    @property
    def active_path(self) -> Path:
        return self.dir / ACTIVE_NAME

    @property
    def history_path(self) -> Path:
        return self.dir / HISTORY_NAME


    def exists(self) -> bool:
        return self.active_path.exists() and self.history_path.exists()

    def history(self) -> list[StrategyState]:
        if not self.history_path.exists():
            return []
        out: list[StrategyState] = []
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(StrategyState.model_validate_json(line))
            except Exception:  # noqa: BLE001 — a torn tail loses one version
                continue
        return out

    def load(self, *, edit_code: str = "") -> StrategyState:
        if not self.exists():
            return seed_state(edit_code)
        try:
            pointer: dict[str, Any] = json.loads(
                self.active_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return seed_state(edit_code)
        versions = self.history()
        if not versions:
            return seed_state(edit_code)
        wanted = pointer.get("active_version")
        for state in reversed(versions):
            if state.version == wanted:
                return state
        return versions[-1]


    def commit(self, state: StrategyState, *, reason: str = "") -> StrategyState:
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(state.model_dump_json() + "\n")
        self.active_path.write_text(
            json.dumps(
                {
                    "active_version": state.version,
                    "reason": reason or f"batch={state.observed_batch}",
                    "at": _now(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return state
