"""Round-0 evidence and task-opportunity bootstrap pipeline."""

from __future__ import annotations

from ..core import Pipeline
from ..operators import (
    EventCandidateExtractionOperator,
    PromoteEventProposalsOperator,
    SemanticEventProposalOperator,
    TaskOpportunityMiningOperator,
)


def build_bootstrap_pipeline() -> Pipeline:
    return Pipeline(
        "round0_event_task_bootstrap",
        [
            EventCandidateExtractionOperator(),
            SemanticEventProposalOperator(),
            PromoteEventProposalsOperator(),
            TaskOpportunityMiningOperator(),
        ],
    )
