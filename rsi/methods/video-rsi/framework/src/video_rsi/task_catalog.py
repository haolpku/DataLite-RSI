"""Round-0 task space with concrete evidence requirements."""

from __future__ import annotations

from typing import Any


TASK_CATALOG: dict[str, dict[str, Any]] = {
    "entity_tracking": {
        "subtype": "identity_across_intervals",
        "min_events": 2,
        "min_hops": 1,
        "evidence_topology": "shared_entity_chain",
    },
    "state_change": {
        "subtype": "visible_before_after",
        "min_events": 1,
        "min_hops": 1,
        "evidence_topology": "precondition_outcome",
    },
    "temporal_relation": {
        "subtype": "relative_order_or_interval",
        "min_events": 2,
        "min_hops": 1,
        "evidence_topology": "ordered_disjoint_events",
    },
    "conditional_counting": {
        "subtype": "count_events_satisfying_condition",
        "min_events": 3,
        "min_hops": 2,
        "evidence_topology": "repeated_entity_events",
    },
    "dynamic_spatial_trajectory": {
        "subtype": "movement_path_or_destination",
        "min_events": 1,
        "min_hops": 1,
        "evidence_topology": "motion_with_end_state",
    },
    "cross_event_comparison": {
        "subtype": "compare_actions_or_states",
        "min_events": 2,
        "min_hops": 2,
        "evidence_topology": "shared_entity_parallel_evidence",
    },
    "causal_event_dependency": {
        "subtype": "visible_precondition_to_outcome",
        "min_events": 2,
        "min_hops": 2,
        "evidence_topology": "outcome_precondition_bridge",
    },
    "multi_hop_reasoning": {
        "subtype": "entity_event_chain",
        "min_events": 3,
        "min_hops": 2,
        "evidence_topology": "connected_event_path",
    },
}

