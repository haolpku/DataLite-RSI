"""Build a task-neutral evidence index directly from Caption and Entity data."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..core import Operator, RunContext
from ..registry import OPERATOR_REGISTRY
from .events import EventCandidateExtractionOperator


@OPERATOR_REGISTRY.register()
class EvidenceIndexOperator(Operator):
    """Normalize source observations without imposing semantic-event boundaries."""

    name = "evidence_index"
    version = "1"
    input_keys = ("observations",)
    output_keys = ("evidence_units", "entity_tracks")

    def run(self, state: dict[str, Any], context: RunContext) -> dict[str, Any]:
        atomic = EventCandidateExtractionOperator().run(state, context)[
            "atomic_evidence"
        ]
        evidence_units = []
        for item in atomic:
            evidence_units.append(
                {
                    "evidence_id": item["atomic_id"],
                    "source_kind": item["source_kind"],
                    "segment_index": item["segment_index"],
                    "start_sec": item["start_sec"],
                    "end_sec": item["end_sec"],
                    "text": item["description"],
                    "entity_ids": item["participants"],
                    "visible_preconditions": item["visible_preconditions"],
                    "visible_outcome": item["visible_outcome"],
                    "uncertainty": item["uncertainty"],
                    "timestamp_quality": item["timestamp_quality"],
                }
            )

        tracks: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for observation in state["observations"]:
            base = float(observation["start_sec"])
            duration = float(observation["duration_sec"])
            for entity in observation["entity_updates"]:
                entity_id = entity.get("entity_id")
                if not isinstance(entity_id, str) or not entity_id:
                    continue
                intervals = entity.get("evidence_intervals") or [
                    {"start_sec": 0.0, "end_sec": duration}
                ]
                for interval in intervals:
                    start = max(0.0, float(interval.get("start_sec", 0.0)))
                    end = min(duration, float(interval.get("end_sec", duration)))
                    if end < start:
                        continue
                    tracks[entity_id].append(
                        {
                            "segment_index": int(observation["segment_index"]),
                            "start_sec": base + start,
                            "end_sec": base + end,
                            "attributes": {
                                key: value
                                for key, value in entity.items()
                                if key not in {"entity_id", "evidence_intervals"}
                            },
                        }
                    )
        entity_tracks = [
            {
                "entity_id": entity_id,
                "observations": sorted(
                    observations,
                    key=lambda item: (item["start_sec"], item["end_sec"]),
                ),
            }
            for entity_id, observations in sorted(tracks.items())
        ]
        return {
            "evidence_units": evidence_units,
            "entity_tracks": entity_tracks,
        }

