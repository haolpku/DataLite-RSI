"""Append-only high-quality data-pool ledger with producer lineage."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Iterable

from .storage import atomic_write_json, utc_now


class HighQualityPoolStore:
    """Persist accepted samples without overwriting earlier pipeline outputs."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.root / "pool_ledger.jsonl"
        self._lock = threading.Lock()

    def append(
        self,
        samples: Iterable[dict[str, Any]],
        *,
        rsi_round_id: str,
        evaluation_policy_version: str,
    ) -> int:
        count = 0
        with self._lock:
            with self.ledger_path.open("a", encoding="utf-8") as handle:
                for sample in samples:
                    producer = dict(sample.get("producer") or {})
                    record = {
                        "record_type": "high_quality_sample",
                        "sample_id": sample.get("sample_id"),
                        "content": sample,
                        "rsi_round_id": rsi_round_id,
                        "evaluation_policy_version": evaluation_policy_version,
                        "pipeline": producer.get("pipeline"),
                        "pipeline_version": producer.get("pipeline_version"),
                        "pipeline_fingerprint": producer.get("pipeline_fingerprint"),
                        "operator_provenance": producer.get("operator_provenance", []),
                        "accepted_at": utc_now(),
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                handle.flush()
        return count

    def write_round_manifest(self, round_id: str, payload: dict[str, Any]) -> None:
        atomic_write_json(self.root / "rounds" / f"{round_id}.json", payload)

