"""Streaming reader for completed Caption/Entity results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class CorpusRecord:
    video_key: str
    result_path: Path
    state: dict[str, Any]


def _observation_from_segment(item: dict[str, Any]) -> dict[str, Any]:
    segment = item.get("segment") or {}
    entities = item.get("entities") or {}
    return {
        "segment_index": int(segment["index"]),
        "start_sec": float(segment["start_sec"]),
        "end_sec": float(segment["end_sec"]),
        "duration_sec": float(segment["duration_sec"]),
        "caption": str((item.get("caption") or {}).get("minute_caption") or ""),
        "entity_updates": list(entities.get("new_entities") or [])
        + list(entities.get("updated_entities") or []),
        "observed_events": list(entities.get("events") or []),
    }


def load_caption_result(path: Path) -> CorpusRecord:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") != "completed":
        raise ValueError(f"caption result is not completed: {path}")
    segments = payload.get("segments")
    if not isinstance(segments, list) or len(segments) != payload.get(
        "expected_segment_count"
    ):
        raise ValueError(f"caption result has incomplete segments: {path}")
    observations = sorted(
        (_observation_from_segment(item) for item in segments),
        key=lambda item: item["segment_index"],
    )
    expected_indices = list(range(len(observations)))
    actual_indices = [item["segment_index"] for item in observations]
    if actual_indices != expected_indices:
        raise ValueError(f"non-contiguous segment indices in {path}")
    video_key = path.parent.name
    return CorpusRecord(
        video_key=video_key,
        result_path=path.resolve(),
        state={
            "video_key": video_key,
            "video": payload["video"],
            "source_result": str(path.resolve()),
            "observations": observations,
        },
    )


def iter_caption_corpus(root: Path, limit: int | None = None) -> Iterator[CorpusRecord]:
    count = 0
    for path in sorted(root.glob("*/result.json")):
        if limit is not None and count >= limit:
            break
        yield load_caption_result(path)
        count += 1

