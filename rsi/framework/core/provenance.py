"""Append-only provenance for method-neutral execution facts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..io.storage import RecordStore


class ProvenanceStore:
    def __init__(self, root: str | Path) -> None:
        self.records = RecordStore(root)

    def append(self, event: Mapping[str, Any]) -> Path:
        return self.records.append_jsonl("events", dict(event))

    def read(self) -> list[dict[str, Any]]:
        return self.records.read_jsonl("events")
