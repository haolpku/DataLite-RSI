"""Deterministic mining of concrete task opportunities from event graphs."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from ..core import Operator, RunContext
from ..registry import OPERATOR_REGISTRY
from ..task_catalog import TASK_CATALOG


MOTION_RE = re.compile(
    r"\b(walk|run|move|travel|enter|exit|leave|approach|cross|drive|ride|"
    r"climb|descend|turn|return|arrive|depart|follow|trajectory)\w*\b",
    re.IGNORECASE,
)


@OPERATOR_REGISTRY.register()
class TaskOpportunityMiningOperator(Operator):
    name = "task_opportunity_mining"
    version = "1"
    input_keys = ("semantic_events",)
    output_keys = ("task_opportunities",)

    def __init__(self, max_per_task: int = 1) -> None:
        self.max_per_task = max_per_task

    def config(self) -> dict[str, Any]:
        return {"max_per_task": self.max_per_task}

    @staticmethod
    def _opportunity(
        task_type: str,
        events: list[dict[str, Any]],
        entity_ids: list[str],
        rationale: str,
    ) -> dict[str, Any]:
        spec = TASK_CATALOG[task_type]
        start = min(event["start_sec"] for event in events)
        end = max(event["end_sec"] for event in events)
        return {
            "task_type": task_type,
            "task_subtype": spec["subtype"],
            "event_ids": [event["event_id"] for event in events],
            "entity_ids": sorted(set(entity_ids)),
            "hop_count": max(spec["min_hops"], len(events) - 1),
            "temporal_span_sec": end - start,
            "evidence_topology": spec["evidence_topology"],
            "rationale": rationale,
        }

    def run(self, state: dict[str, Any], context: RunContext) -> dict[str, Any]:
        del context
        events = state["semantic_events"]
        by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            for entity_id in event["entity_ids"]:
                by_entity[entity_id].append(event)
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for entity_id, entity_events in by_entity.items():
            if len(entity_events) >= 2:
                pair = [entity_events[0], entity_events[-1]]
                buckets["entity_tracking"].append(
                    self._opportunity(
                        "entity_tracking",
                        pair,
                        [entity_id],
                        "The same stable entity is grounded in separated events.",
                    )
                )
                buckets["cross_event_comparison"].append(
                    self._opportunity(
                        "cross_event_comparison",
                        pair,
                        [entity_id],
                        "Two separated observations support comparing the entity's actions or state.",
                    )
                )
            if len(entity_events) >= 3:
                selected = [entity_events[0], entity_events[len(entity_events) // 2], entity_events[-1]]
                buckets["conditional_counting"].append(
                    self._opportunity(
                        "conditional_counting",
                        selected,
                        [entity_id],
                        "The entity participates in at least three grounded events that can be conditionally counted.",
                    )
                )
                buckets["multi_hop_reasoning"].append(
                    self._opportunity(
                        "multi_hop_reasoning",
                        selected,
                        [entity_id],
                        "A shared entity connects three temporally separated events.",
                    )
                )

        for event in events:
            if event["visible_preconditions"] and event["visible_outcomes"]:
                buckets["state_change"].append(
                    self._opportunity(
                        "state_change",
                        [event],
                        event["entity_ids"],
                        "The event contains explicit visible precondition and outcome evidence.",
                    )
                )
            if MOTION_RE.search(event["summary"]):
                buckets["dynamic_spatial_trajectory"].append(
                    self._opportunity(
                        "dynamic_spatial_trajectory",
                        [event],
                        event["entity_ids"],
                        "The grounded description contains visible movement or a path transition.",
                    )
                )

        for left, right in zip(events, events[1:]):
            buckets["temporal_relation"].append(
                self._opportunity(
                    "temporal_relation",
                    [left, right],
                    list(set(left["entity_ids"]) | set(right["entity_ids"])),
                    "Two ordered semantic events support before/after or interval reasoning.",
                )
            )
            left_outcomes = " ".join(left["visible_outcomes"]).lower()
            right_preconditions = " ".join(right["visible_preconditions"]).lower()
            shared_entities = set(left["entity_ids"]) & set(right["entity_ids"])
            if shared_entities and left_outcomes and right_preconditions:
                buckets["causal_event_dependency"].append(
                    self._opportunity(
                        "causal_event_dependency",
                        [left, right],
                        sorted(shared_entities),
                        "An earlier visible outcome and later visible precondition share an entity; causality still requires verification.",
                    )
                )

        output = []
        for task_type in TASK_CATALOG:
            for item in buckets[task_type][: self.max_per_task]:
                item = dict(item)
                item["opportunity_id"] = f"{task_type}_{len(output) + 1:04d}"
                output.append(item)
        return {"task_opportunities": output}
