from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from ..core.schema import StepCost


FILENAME = "_run_overhead.jsonl"


@dataclass(slots=True)
class OverheadTotals:
    api_calls: int = 0
    tokens: int = 0
    records: int = 0


class OverheadLedger:

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self.root / FILENAME


    def record(
        self,
        *,
        op: str,
        cost: StepCost,
        setting_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        api_calls: int = 1,
        note: str = "",
    ) -> None:
        row: dict[str, Any] = {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "setting_id": setting_id,
            "batch_id": batch_id,
            "op": op,
            "model": cost.model,
            "api_calls": api_calls,
            "prompt_tokens": cost.prompt_tokens,
            "completion_tokens": cost.completion_tokens,
            "total_tokens": cost.total_tokens,
            "latency_s": cost.latency_s,
            "note": note,
        }
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def totals(self, *, setting_id: Optional[str] = None) -> OverheadTotals:
        return summarise(self.rows(), setting_id=setting_id)


def summarise(
    rows: Iterable[dict[str, Any]], *, setting_id: Optional[str] = None
) -> OverheadTotals:
    totals = OverheadTotals()
    for row in rows:
        if setting_id is not None and row.get("setting_id") != setting_id:
            continue
        totals.records += 1
        totals.api_calls += int(row.get("api_calls") or 0)
        totals.tokens += int(row.get("total_tokens") or 0)
    return totals
