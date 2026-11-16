"""Complete task-centric generation/evaluation path for one RSI round."""

from __future__ import annotations

from ..backends import (
    DistractorEnhancerBackend,
    EvidenceVerifierBackend,
    FrozenTargetBackend,
    QuestionGeneratorBackend,
    TextOnlyAnswerBackend,
)
from ..core import Pipeline
from ..operators.evidence import EvidenceIndexOperator
from ..operators.production import (
    DistractorEnhancementOperator,
    DistractorQualityOperator,
    FrontierAndDedupOperator,
    FrozenTargetFrontierFilterOperator,
    GroundedTaskBuilderOperator,
    QuestionGenerationOperator,
    TextOnlyFilterOperator,
)
from ..operators.task_candidates import TaskCandidateMiningOperator


def build_task_centric_rsi_pipeline(
    *,
    generator_backend: QuestionGeneratorBackend,
    verifier_backend: EvidenceVerifierBackend | None = None,
    text_only_backend: TextOnlyAnswerBackend,
    frozen_target_backend: FrozenTargetBackend,
    distractor_backend: DistractorEnhancerBackend | None = None,
    max_per_task: int = 1,
    frontier_trials: int = 4,
    candidate_variants: int = 1,
    target_model_name: str = "qwen3-vl-8b-instruct",
) -> Pipeline:
    """Build the deliberately small direct-evidence v0 baseline.

    v0 is intentionally a thin, auditable loop.  Evidence selection, the cheap
    complexity gate and task metadata are one ``GroundedTaskBuilder`` block;
    final frontier selection, local eligibility and dedup are one final block.
    The expensive semantic-event, rewatch and external-verifier branches stay
    optional and are not hidden in the baseline.
    """

    # Kept as a compatibility argument for callers that still pass the old
    # verifier backend.  The default graph deliberately does not instantiate
    # or call it.
    del verifier_backend
    return Pipeline(
        "task_centric_rsi_round",
        [
            EvidenceIndexOperator(),
            TaskCandidateMiningOperator(max_per_task=max_per_task),
            GroundedTaskBuilderOperator(),
            QuestionGenerationOperator(generator_backend, variants_per_candidate=candidate_variants),
            DistractorEnhancementOperator(distractor_backend),
            DistractorQualityOperator(),
            TextOnlyFilterOperator(text_only_backend),
            FrozenTargetFrontierFilterOperator(
                frozen_target_backend,
                trials=frontier_trials,
                target_model_name=target_model_name,
            ),
            FrontierAndDedupOperator(),
        ],
        result_keys=("pool_candidates", "feedback_signals"),
        version="0.5.0-v0-slim-no-gemini-verification",
    )
