"""Versioned, deliberately small model assignment policy for RSI."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelAssignment:
    role: str
    model: str
    provider: str
    modality: str
    fixed_within_round: bool
    notes: str = ""


@dataclass(frozen=True)
class ModelPolicy:
    version: str
    assignments: tuple[ModelAssignment, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "assignments": [asdict(item) for item in self.assignments],
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.as_dict(), sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


V0_MODEL_POLICY = ModelPolicy(
    version="v0-model-policy-1",
    assignments=(
        ModelAssignment(
            "coding_agent",
            "gpt-5.6-sol",
            "codex",
            "text",
            False,
            "Use high or medium reasoning effort according to the round budget.",
        ),
        ModelAssignment(
            "question_generation",
            "gpt-5.5",
            "openai",
            "text",
            True,
            "Single high-quality generator for v0; task-specific prompts vary by TaskSpec.",
        ),
        ModelAssignment(
            "task_candidate_mining",
            "none",
            "deterministic",
            "text",
            True,
            "v0 uses rule-based high-recall mining; optional semantic reranking reuses gpt-5.5.",
        ),
        ModelAssignment(
            "text_only_filter",
            "glm-5.3-flash",
            "zhipu",
            "text",
            True,
            "May be replaced by a fixed local or Gemini-family text model in an ablation.",
        ),
        ModelAssignment(
            "distractor_enhancement",
            "glm-5.3-flash",
            "zhipu",
            "text",
            True,
            "Constrained option rewriting; cheaper model is sufficient after deterministic quality gates.",
        ),
        ModelAssignment(
            "deduplication",
            "qwen3-embedding-8b",
            "local",
            "text",
            True,
            "Embedding plus deterministic similarity policy; no generative judge required.",
        ),
        ModelAssignment(
            "frozen_target",
            "qwen3-vl-8b-instruct",
            "local",
            "video-text",
            True,
            "Four trials; 0/4, 4/4 and mixed outcomes are retained with difficulty labels.",
        ),
    ),
)
