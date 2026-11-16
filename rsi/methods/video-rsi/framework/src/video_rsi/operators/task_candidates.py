"""Task-first candidate mining over evidence units and entity tracks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..core import Operator, RunContext
from ..registry import OPERATOR_REGISTRY
from ..task_catalog import TASK_CATALOG
from .task_mining import MOTION_RE


@OPERATOR_REGISTRY.register()
class TaskCandidateMiningOperator(Operator):
    """Propose task-family candidates without requiring semantic events."""

    name = "task_candidate_mining"
    version = "1"
    input_keys = ("evidence_units", "entity_tracks")
    output_keys = ("task_candidates",)

    def __init__(self, max_per_task: int = 1) -> None:
        self.max_per_task = max_per_task

    def config(self) -> dict[str, Any]:
        return {"max_per_task": self.max_per_task, "representation": "direct_evidence"}

    @staticmethod
    def _make(
        task_type: str,
        evidence: list[dict[str, Any]],
        entity_ids: list[str],
        goal: str,
    ) -> dict[str, Any]:
        spec = TASK_CATALOG[task_type]
        start = min(float(item["start_sec"]) for item in evidence)
        end = max(float(item["end_sec"]) for item in evidence)
        return {
            "task_type": task_type,
            "task_subtype": spec["subtype"],
            "goal": goal,
            "evidence_ids": [item["evidence_id"] for item in evidence],
            "entity_ids": sorted(set(entity_ids)),
            "hop_count": max(spec["min_hops"], len(evidence) - 1),
            "temporal_span_sec": end - start,
            "evidence_topology": spec["evidence_topology"],
            "event_abstraction_recommended": task_type
            in {
                "cross_event_comparison",
                "causal_event_dependency",
                "multi_hop_reasoning",
            },
            "representation": "direct_evidence",
            # Cheap quality signal used by the pre-generation complexity gate;
            # it prevents spending LLM calls on local one-caption questions.
            "complexity_score": float(
                max(spec["min_hops"], len(evidence) - 1)
                + min(3.0, max(0.0, (end - start) / 60.0))
            ),
        }

    def run(self, state: dict[str, Any], context: RunContext) -> dict[str, Any]:
        del context
        evidence = [item for item in state["evidence_units"] if item["text"]]
        by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in evidence:
            for entity_id in item["entity_ids"]:
                by_entity[entity_id].append(item)
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for entity_id, items in by_entity.items():
            ordered = sorted(items, key=lambda item: (item["start_sec"], item["end_sec"]))
            if len(ordered) >= 2:
                pair = [ordered[0], ordered[-1]]
                buckets["entity_tracking"].append(
                    self._make(
                        "entity_tracking",
                        pair,
                        [entity_id],
                        "Resolve the same grounded entity across separated intervals.",
                    )
                )
                buckets["cross_event_comparison"].append(
                    self._make(
                        "cross_event_comparison",
                        pair,
                        [entity_id],
                        "Compare the entity's visible action or state across two intervals.",
                    )
                )
            if len(ordered) >= 3:
                selected = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
                buckets["conditional_counting"].append(
                    self._make(
                        "conditional_counting",
                        selected,
                        [entity_id],
                        "Count distinct grounded occurrences satisfying an entity-conditioned rule.",
                    )
                )
                buckets["multi_hop_reasoning"].append(
                    self._make(
                        "multi_hop_reasoning",
                        selected,
                        [entity_id],
                        "Compose a multi-step inference across three separated observations.",
                    )
                )

        for item in evidence:
            if item["visible_preconditions"] and item["visible_outcome"]:
                # A state-change question should compare observations whenever
                # the entity reappears. The old implementation emitted the
                # current unit alone, which produced caption paraphrases.
                repeated = [
                    other for other in evidence
                    if other["evidence_id"] != item["evidence_id"]
                    and set(item["entity_ids"]) & set(other["entity_ids"])
                    and float(other["start_sec"]) > float(item["end_sec"])
                ]
                state_evidence = [item, min(repeated, key=lambda x: x["start_sec"])] if repeated else [item]
                state_candidate = self._make(
                        "state_change",
                        state_evidence,
                        sorted(set(item["entity_ids"]) | {
                            entity for other in state_evidence for entity in other["entity_ids"]
                        }),
                        "Compare the same grounded entity's visible state before and after a transition.",
                    )
                if len(state_evidence) == 1:
                    state_candidate["single_observation_fallback"] = True
                buckets["state_change"].append(state_candidate)
            if MOTION_RE.search(item["text"]):
                later = [
                    other for other in evidence
                    if float(other["start_sec"]) > float(item["end_sec"])
                    and set(item["entity_ids"]) & set(other["entity_ids"])
                ]
                spatial_evidence = [item, min(later, key=lambda x: x["start_sec"])] if later else [item]
                buckets["dynamic_spatial_trajectory"].append(
                    self._make(
                        "dynamic_spatial_trajectory",
                        spatial_evidence,
                        sorted({entity for other in spatial_evidence for entity in other["entity_ids"]}),
                        "Recover a visible movement path, direction and destination across observations.",
                    )
                )

        ordered_evidence = sorted(
            evidence, key=lambda item: (item["start_sec"], item["end_sec"])
        )
        for left, right in zip(ordered_evidence, ordered_evidence[1:]):
            if float(right["start_sec"]) >= float(left["end_sec"]):
                entities = sorted(set(left["entity_ids"]) | set(right["entity_ids"]))
                buckets["temporal_relation"].append(
                    self._make(
                        "temporal_relation",
                        [left, right],
                        entities,
                        "Determine the order or interval relation of two grounded observations.",
                    )
                )
            shared = sorted(set(left["entity_ids"]) & set(right["entity_ids"]))
            if shared and left["visible_outcome"] and right["visible_preconditions"]:
                buckets["causal_event_dependency"].append(
                    self._make(
                        "causal_event_dependency",
                        [left, right],
                        shared,
                        "Test a visible outcome-to-precondition dependency; causality requires verification.",
                    )
                )

        output = []
        for task_type in TASK_CATALOG:
            # Prefer candidates with more evidence and a wider temporal span;
            # insertion order previously favored the easiest local sentence.
            ranked = sorted(
                buckets[task_type],
                key=lambda item: (-float(item.get("complexity_score", 0.0)),
                                  -float(item.get("temporal_span_sec", 0.0))),
            )
            seen_topologies = set()
            for item in ranked:
                topology = tuple(item["evidence_ids"])
                if topology in seen_topologies:
                    continue
                seen_topologies.add(topology)
                if len(seen_topologies) > self.max_per_task:
                    break
                candidate = dict(item)
                candidate["candidate_id"] = f"tc_{len(output) + 1:05d}"
                output.append(candidate)
        return {"task_candidates": output}
