"""Model-refined semantic event and task-opportunity pipeline."""

from __future__ import annotations

from ..core import Pipeline
from ..model_client import TextJSONModelClient
from ..operators import (
    EventCandidateExtractionOperator,
    LLMSemanticEventRefinementOperator,
    SemanticEventProposalOperator,
    TaskOpportunityMiningOperator,
)


def build_semantic_fusion_pipeline(
    client: TextJSONModelClient, *, max_tokens: int = 5000
) -> Pipeline:
    return Pipeline(
        "round0_model_semantic_fusion",
        [
            EventCandidateExtractionOperator(),
            SemanticEventProposalOperator(),
            LLMSemanticEventRefinementOperator(client, max_tokens=max_tokens),
            TaskOpportunityMiningOperator(),
        ],
    )

