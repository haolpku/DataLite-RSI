"""Grounded atomic evidence extraction and deterministic event fusion."""

from __future__ import annotations

import re
from typing import Any

from ..core import Operator, RunContext
from ..registry import OPERATOR_REGISTRY


TIMESTAMP_RE = re.compile(
    r"\[(?P<sm>\d+):(?P<ss>[0-5]\d)-(?P<em>\d+):(?P<es>[0-5]\d)\]"
)
WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _token_set(text: str) -> set[str]:
    return {item.lower() for item in WORD_RE.findall(text) if len(item) > 2}


def _jaccard(left: str, right: str) -> float:
    a, b = _token_set(left), _token_set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _entity_ids_for_interval(
    updates: list[dict[str, Any]], start_sec: float, end_sec: float
) -> list[str]:
    result = []
    for entity in updates:
        entity_id = entity.get("entity_id")
        if not isinstance(entity_id, str):
            continue
        intervals = entity.get("evidence_intervals") or []
        if any(
            float(interval.get("end_sec", -1)) >= start_sec
            and float(interval.get("start_sec", 1e18)) <= end_sec
            for interval in intervals
            if isinstance(interval, dict)
        ):
            result.append(entity_id)
    return sorted(set(result))


def _caption_spans(observation: dict[str, Any]) -> list[dict[str, Any]]:
    text = observation["caption"]
    matches = list(TIMESTAMP_RE.finditer(text))
    base = float(observation["start_sec"])
    duration = float(observation["duration_sec"])
    output = []
    prefix = text[: matches[0].start()].strip() if matches else text.strip()
    if prefix:
        output.append(
            {
                "local_start_sec": 0.0,
                "local_end_sec": duration,
                "description": prefix,
                "timestamp_quality": "unanchored_prefix",
            }
        )
    for index, match in enumerate(matches):
        sm, ss, em, es = map(int, match.groups())
        start = min(duration, float(sm * 60 + ss))
        end = min(duration, float(em * 60 + es))
        next_pos = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        description = text[match.end() : next_pos].strip()
        if not description or end < start:
            continue
        output.append(
            {
                "local_start_sec": start,
                "local_end_sec": end,
                "description": description,
                "timestamp_quality": "anchored",
            }
        )
    for item in output:
        item["global_start_sec"] = base + item.pop("local_start_sec")
        item["global_end_sec"] = base + item.pop("local_end_sec")
    return output


@OPERATOR_REGISTRY.register()
class EventCandidateExtractionOperator(Operator):
    name = "event_candidate_extraction"
    version = "1"
    input_keys = ("observations",)
    output_keys = ("atomic_evidence",)

    def run(self, state: dict[str, Any], context: RunContext) -> dict[str, Any]:
        del context
        atomic = []
        for observation in state["observations"]:
            segment_index = int(observation["segment_index"])
            base = float(observation["start_sec"])
            duration = float(observation["duration_sec"])
            updates = observation["entity_updates"]
            for index, event in enumerate(observation["observed_events"], start=1):
                local_start = max(0.0, float(event.get("start_sec", 0.0)))
                local_end = min(duration, float(event.get("end_sec", duration)))
                atomic.append(
                    {
                        "atomic_id": f"a_{segment_index:04d}_entity_{index:02d}",
                        "source_kind": "entity_event",
                        "segment_index": segment_index,
                        "start_sec": base + local_start,
                        "end_sec": base + local_end,
                        "description": str(event.get("description") or "").strip(),
                        "participants": sorted(set(event.get("participants") or [])),
                        "visible_preconditions": str(
                            event.get("visible_preconditions") or ""
                        ).strip(),
                        "visible_outcome": str(event.get("visible_outcome") or "").strip(),
                        "uncertainty": str(event.get("uncertainty") or "").strip(),
                        "timestamp_quality": "structured",
                    }
                )
            for index, span in enumerate(_caption_spans(observation), start=1):
                local_start = span["global_start_sec"] - base
                local_end = span["global_end_sec"] - base
                atomic.append(
                    {
                        "atomic_id": f"a_{segment_index:04d}_caption_{index:02d}",
                        "source_kind": "caption_span",
                        "segment_index": segment_index,
                        "start_sec": span["global_start_sec"],
                        "end_sec": span["global_end_sec"],
                        "description": span["description"],
                        "participants": _entity_ids_for_interval(
                            updates, local_start, local_end
                        ),
                        "visible_preconditions": "",
                        "visible_outcome": "",
                        "uncertainty": "",
                        "timestamp_quality": span["timestamp_quality"],
                    }
                )
        atomic.sort(key=lambda item: (item["start_sec"], item["end_sec"], item["atomic_id"]))
        return {"atomic_evidence": atomic}


@OPERATOR_REGISTRY.register()
class SemanticEventProposalOperator(Operator):
    """Propose coarse event blocks before model-based semantic refinement."""

    name = "semantic_event_proposal"
    version = "1"
    input_keys = ("atomic_evidence",)
    output_keys = ("semantic_event_proposals",)

    def __init__(
        self,
        *,
        max_gap_sec: float = 5.0,
        max_event_sec: float = 180.0,
        similarity_threshold: float = 0.32,
    ) -> None:
        self.max_gap_sec = max_gap_sec
        self.max_event_sec = max_event_sec
        self.similarity_threshold = similarity_threshold

    def config(self) -> dict[str, Any]:
        return {
            "max_gap_sec": self.max_gap_sec,
            "max_event_sec": self.max_event_sec,
            "similarity_threshold": self.similarity_threshold,
        }

    def _should_merge(self, current: dict[str, Any], item: dict[str, Any]) -> bool:
        combined_duration = max(current["end_sec"], item["end_sec"]) - current["start_sec"]
        if combined_duration > self.max_event_sec:
            return False
        gap = item["start_sec"] - current["end_sec"]
        if gap > self.max_gap_sec:
            return False
        shared = set(current["entity_ids"]) & set(item["participants"])
        overlap = item["start_sec"] <= current["end_sec"]
        similarity = _jaccard(current["summary"], item["description"])
        same_segment = item["segment_index"] in current["segment_indices"]
        return bool(shared) or similarity >= self.similarity_threshold or (
            overlap and same_segment
        )

    @staticmethod
    def _new_event(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "start_sec": item["start_sec"],
            "end_sec": item["end_sec"],
            "summary_parts": [item["description"]] if item["description"] else [],
            "summary": item["description"],
            "entity_ids": list(item["participants"]),
            "segment_indices": [item["segment_index"]],
            "atomic_evidence_ids": [item["atomic_id"]],
            "visible_preconditions": [item["visible_preconditions"]]
            if item["visible_preconditions"]
            else [],
            "visible_outcomes": [item["visible_outcome"]]
            if item["visible_outcome"]
            else [],
            "timestamp_quality": [item["timestamp_quality"]],
        }

    @staticmethod
    def _merge(current: dict[str, Any], item: dict[str, Any]) -> None:
        current["end_sec"] = max(current["end_sec"], item["end_sec"])
        if item["description"] and item["description"] not in current["summary_parts"]:
            current["summary_parts"].append(item["description"])
        current["summary"] = " ".join(current["summary_parts"])
        current["entity_ids"] = sorted(
            set(current["entity_ids"]) | set(item["participants"])
        )
        current["segment_indices"] = sorted(
            set(current["segment_indices"]) | {item["segment_index"]}
        )
        current["atomic_evidence_ids"].append(item["atomic_id"])
        if item["visible_preconditions"]:
            current["visible_preconditions"].append(item["visible_preconditions"])
        if item["visible_outcome"]:
            current["visible_outcomes"].append(item["visible_outcome"])
        current["timestamp_quality"].append(item["timestamp_quality"])

    def run(self, state: dict[str, Any], context: RunContext) -> dict[str, Any]:
        del context
        fused: list[dict[str, Any]] = []
        current = None
        for item in state["atomic_evidence"]:
            if current is None:
                current = self._new_event(item)
                continue
            if self._should_merge(current, item):
                self._merge(current, item)
            else:
                fused.append(current)
                current = self._new_event(item)
        if current is not None:
            fused.append(current)
        output = []
        for index, event in enumerate(fused, start=1):
            event = dict(event)
            event.pop("summary_parts", None)
            qualities = set(event.pop("timestamp_quality"))
            event["event_id"] = f"se_{index:04d}"
            event["evidence_quality"] = (
                "timestamp_uncertain" if "unanchored_prefix" in qualities else "anchored"
            )
            event["duration_sec"] = event["end_sec"] - event["start_sec"]
            output.append(event)
        return {"semantic_event_proposals": output}


@OPERATOR_REGISTRY.register()
class PromoteEventProposalsOperator(Operator):
    """Bootstrap-only adapter; production uses model semantic refinement."""

    name = "promote_event_proposals"
    version = "1"
    input_keys = ("semantic_event_proposals",)
    output_keys = ("semantic_events",)

    def run(self, state: dict[str, Any], context: RunContext) -> dict[str, Any]:
        del context
        events = []
        for item in state["semantic_event_proposals"]:
            event = dict(item)
            event["refinement"] = "deterministic_bootstrap"
            events.append(event)
        return {"semantic_events": events}
