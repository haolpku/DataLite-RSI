"""Atomic, provenance-preserving storage for RSI runs."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return result[:160] or "record"


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


class RunStore:
    def __init__(self, root: Path, run_id: str) -> None:
        self.root = root.resolve()
        self.run_id = safe_name(run_id)
        self.run_dir = self.root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._error_lock = threading.Lock()
        self._summary_lock = threading.Lock()

    def write_manifest(self, payload: dict[str, Any]) -> None:
        path = self.run_dir / "run_manifest.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            comparable_existing = dict(existing)
            comparable_existing.pop("created_at", None)
            comparable_payload = dict(payload)
            comparable_payload.pop("created_at", None)
            if comparable_existing != comparable_payload:
                raise ValueError(
                    f"run_id {self.run_id!r} already exists with a different manifest"
                )
            return
        atomic_write_json(path, payload)

    def stage_path(self, video_key: str, stage_index: int, stage_name: str) -> Path:
        return (
            self.run_dir
            / "videos"
            / safe_name(video_key)
            / "stages"
            / f"{stage_index:02d}_{safe_name(stage_name)}.json"
        )

    def load_stage(
        self,
        video_key: str,
        stage_index: int,
        stage_name: str,
        fingerprint: str,
    ) -> dict[str, Any] | None:
        path = self.stage_path(video_key, stage_index, stage_name)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("operator_fingerprint") != fingerprint:
            return None
        outputs = payload.get("outputs")
        return outputs if isinstance(outputs, dict) else None

    def save_stage(
        self,
        *,
        video_key: str,
        stage_index: int,
        stage_name: str,
        fingerprint: str,
        outputs: dict[str, Any],
        stats: dict[str, Any],
    ) -> None:
        atomic_write_json(
            self.stage_path(video_key, stage_index, stage_name),
            {
                "schema_version": "video-rsi-stage-v1",
                "video_key": video_key,
                "stage_index": stage_index,
                "operator": stage_name,
                "operator_fingerprint": fingerprint,
                "stats": stats,
                "outputs": outputs,
                "completed_at": utc_now(),
            },
        )

    def save_final(self, video_key: str, payload: dict[str, Any]) -> None:
        path = self.run_dir / "videos" / safe_name(video_key) / "result.json"
        atomic_write_json(path, payload)

    def append_error(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._error_lock:
            with (self.run_dir / "errors.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()

    def write_summary(self, payload: dict[str, Any]) -> None:
        with self._summary_lock:
            atomic_write_json(self.run_dir / "run_summary.json", payload)

