"""Core data contracts for the review-only feedback loop."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rsi.framework.core.contracts import InputContract


@dataclass
class TaskSpec:
    task_description: str
    target_schema: dict[str, str]
    quality_criteria: list[str]
    input_contract: InputContract | None = None
    evaluator_kind: str = "review_agent"

    @property
    def modalities(self) -> tuple[str, ...]:
        return self.input_contract.modalities if self.input_contract else ("text",)


@dataclass
class PipelineConfig:
    operators: list[dict[str, Any]]
    field_flow: str
    code: str
    rationale: str


@dataclass
class EmbeddingQualityResult:
    """Deterministic embedding evidence for distribution fit and diversity."""

    enabled: bool = False
    status: str = "disabled"
    metric: str = ""
    enabled_metrics: list[str] = field(default_factory=list)
    proxy_name: str = ""

    candidate_total: int = 0
    candidate_sample_size: int = 0
    proxy_sample_size: int = 0

    embedding_model: str = ""
    embedding_dim: int = 0
    normalized: bool = True
    kernel: str = "RBF"
    sigma: float = 1.0
    estimator: str = "biased"

    mmd: float | None = None
    das: float | None = None
    delta_mmd_vs_parent: float | None = None
    embedding_fail_rate: float = 0.0
    embedding_wall_time_seconds: float = 0.0
    kernel_terms: dict[str, float] = field(default_factory=dict)
    embedding_diversity: dict[str, float] = field(default_factory=dict)
    delta_embedding_diversity_vs_parent: dict[str, float] = field(default_factory=dict)
    embedding_quality_components: dict[str, float] = field(default_factory=dict)
    embedding_quality_score: float | None = None
    embedding_usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ReviewResult:
    """Comparable result, with text-review details or task-owned evidence."""

    schema_score: float
    relevance_score: float
    passed: bool
    issues: list[str] = field(default_factory=list)

    correctness_score: float = 0.0
    difficulty_score: float = 0.0

    dup_rate: float = 0.0
    null_rate: float = 0.0
    parse_valid_rate: float = 1.0
    contamination_rate: float = 0.0

    llm_composite: float = 0.0
    review_score: float = 0.0
    score_components: dict[str, float] = field(default_factory=dict)
    failure_signals: list[str] = field(default_factory=list)
    embedding_quality: EmbeddingQualityResult = field(default_factory=EmbeddingQualityResult)
    # A modality-specific evaluator may attach native IF/VC/VQ, frozen-target,
    # or other evidence without changing the incumbent comparison contract.
    domain_feedback: dict[str, Any] = field(default_factory=dict)
    evaluator_kind: str = "review_agent"
