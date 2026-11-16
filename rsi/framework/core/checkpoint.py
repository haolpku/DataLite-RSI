"""Run-scoped checkpoints with explicit identity validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..io.storage import RecordStore


class CheckpointStore:
    def __init__(self, root: str | Path) -> None:
        self.records = RecordStore(root)

    def save(self, stage: str, identity: str, outputs: Mapping[str, Any]) -> Path:
        return self.records.write_json(
            stage,
            {"schema_version": 1, "identity": identity, "outputs": dict(outputs)},
        )

    def load(self, stage: str, identity: str) -> dict[str, Any] | None:
        value = self.records.read_json(stage)
        if not isinstance(value, dict) or value.get("identity") != identity:
            return None
        outputs = value.get("outputs")
        return outputs if isinstance(outputs, dict) else None
