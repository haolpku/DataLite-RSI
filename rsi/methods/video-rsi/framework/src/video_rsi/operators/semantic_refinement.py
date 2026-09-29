"""Model-backed refinement of coarse proposals into semantic events."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from typing import Any

from ..core import Operator, RunContext
from ..model_client import TextJSONModelClient
from ..registry import OPERATOR_REGISTRY


EVENT_TYPES = [
    "activity",
    "interaction",
    "movement",
    "state_change",
    "presentation",
    "montage",
    "static_state",
    "other",
]


SEMANTIC_EVENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["semantic_events"],
    "properties": {
        "semantic_events": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "start_sec",
                    "end_sec",
                    "event_type",
                    "summary",
                    "entity_ids",
                    "evidence_ids",
                    "visible_precondition",
                    "visible_outcome",
                    "uncertainty",
                ],
                "properties": {
                    "start_sec": {"type": "number", "minimum": 0},
                    "end_sec": {"type": "number", "minimum": 0},
                    "event_type": {"type": "string", "enum": EVENT_TYPES},
                    "summary": {"type": "string", "minLength": 1},
                    "entity_ids": {"type": "array", "items": {"type": "string"}},
                    "evidence_ids": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                    },
                    "visible_precondition": {"type": "string"},
                    "visible_outcome": {"type": "string"},
                    "uncertainty": {"type": "string"},
                },
            },
        }
    },
}


@OPERATOR_REGISTRY.register()
class LLMSemanticEventRefinementOperator(Operator):
    name = "llm_semantic_event_refinement"
    version = "1"
    input_keys = ("atomic_evidence", "semantic_event_proposals")
    output_keys = ("semantic_events",)

    def __init__(
        self,
        client: TextJSONModelClient,
        *,
        max_tokens: int = 5000,
    ) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.prompt_template = (
            files("video_rsi.prompts")
            .joinpath("semantic_event_refinement_v1.txt")
            .read_text(encoding="utf-8")
        )
        self.prompt_sha256 = hashlib.sha256(
            self.prompt_template.encode("utf-8")
        ).hexdigest()

    def config(self) -> dict[str, Any]:
        return {
            "model": self.client.model,
            "max_tokens": self.max_tokens,
            "prompt": "semantic_event_refinement_v1.txt",
            "prompt_sha256": self.prompt_sha256,
        }

    @staticmethod
    def _validate_and_normalize(
        raw: dict[str, Any],
        proposal: dict[str, Any],
        evidence_by_id: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        events = raw.get("semantic_events")
        if not isinstance(events, list) or not events:
            raise ValueError("semantic_events must be a non-empty list")
        if len(events) > 8:
            raise ValueError("semantic_events exceeds maxItems=8")
        normalized = []
        prior_end = float(proposal["start_sec"])
        proposal_start = float(proposal["start_sec"])
        proposal_end = float(proposal["end_sec"])
        for index, event in enumerate(events):
            if not isinstance(event, dict):
                raise ValueError(f"semantic_events[{index}] must be an object")
            start = float(event["start_sec"])
            end = float(event["end_sec"])
            if start < proposal_start - 0.75 or end > proposal_end + 0.75 or end < start:
                raise ValueError(
                    f"semantic event interval [{start}, {end}] outside proposal "
                    f"[{proposal_start}, {proposal_end}]"
                )
            if start < prior_end - 0.75:
                raise ValueError("semantic events must be chronological and non-overlapping")
            evidence_ids = list(dict.fromkeys(event.get("evidence_ids") or []))
            if not evidence_ids or any(item not in evidence_by_id for item in evidence_ids):
                raise ValueError("semantic event contains missing or unknown evidence_ids")
            cited_evidence = [evidence_by_id[item] for item in evidence_ids]
            evidence_start = min(float(item["start_sec"]) for item in cited_evidence)
            evidence_end = max(float(item["end_sec"]) for item in cited_evidence)
            if start < evidence_start - 0.75 or end > evidence_end + 0.75:
                raise ValueError(
                    f"semantic event interval [{start}, {end}] is not supported by "
                    f"its cited evidence envelope [{evidence_start}, {evidence_end}]"
                )
            allowed_entities = {
                entity_id
                for item in cited_evidence
                for entity_id in item["participants"]
            }
            entity_ids = list(dict.fromkeys(event.get("entity_ids") or []))
            if any(item not in allowed_entities for item in entity_ids):
                raise ValueError("semantic event contains an entity absent from cited evidence")
            quality = "anchored"
            if any(
                evidence_by_id[item]["timestamp_quality"] == "unanchored_prefix"
                for item in evidence_ids
            ):
                quality = "timestamp_uncertain"
            normalized.append(
                {
                    "start_sec": start,
                    "end_sec": end,
                    "duration_sec": end - start,
                    "event_type": str(event["event_type"]),
                    "summary": str(event["summary"]).strip(),
                    "entity_ids": sorted(entity_ids),
                    "segment_indices": sorted(
                        {
                            evidence_by_id[item]["segment_index"]
                            for item in evidence_ids
                        }
                    ),
                    "atomic_evidence_ids": evidence_ids,
                    "visible_preconditions": [
                        str(event.get("visible_precondition") or "").strip()
                    ]
                    if str(event.get("visible_precondition") or "").strip()
                    else [],
                    "visible_outcomes": [
                        str(event.get("visible_outcome") or "").strip()
                    ]
                    if str(event.get("visible_outcome") or "").strip()
                    else [],
                    "uncertainty": str(event.get("uncertainty") or "").strip(),
                    "evidence_quality": quality,
                    "refinement": "model",
                }
            )
            prior_end = end
        return normalized

    def run(self, state: dict[str, Any], context: RunContext) -> dict[str, Any]:
        del context
        evidence_by_id = {
            item["atomic_id"]: item for item in state["atomic_evidence"]
        }
        refined = []
        for proposal in state["semantic_event_proposals"]:
            evidence = [
                evidence_by_id[item]
                for item in proposal["atomic_evidence_ids"]
                if item in evidence_by_id
            ]
            prompt = self.prompt_template.format(
                proposal_start_sec=float(proposal["start_sec"]),
                proposal_end_sec=float(proposal["end_sec"]),
                atomic_evidence_json=json.dumps(
                    evidence, ensure_ascii=False, separators=(",", ":")
                ),
            )
            raw = self.client.complete_json(
                prompt=prompt,
                schema=SEMANTIC_EVENT_SCHEMA,
                max_tokens=self.max_tokens,
            )
            refined.extend(
                self._validate_and_normalize(raw, proposal, evidence_by_id)
            )
        for index, event in enumerate(refined, start=1):
            event["event_id"] = f"se_{index:04d}"
        return {"semantic_events": refined}
