"""Backend contracts used by the task-centric RSI pipeline.

Operators own dataflow and validation. Backends own model-specific inference,
which keeps local Qwen, API judges and the frozen target interchangeable.
"""

from __future__ import annotations

from typing import Any, Protocol


class QuestionGeneratorBackend(Protocol):
    name: str

    def generate(
        self,
        *,
        candidate: dict[str, Any],
        evidence: list[dict[str, Any]],
        rewatch: list[dict[str, Any]],
        prompt: str,
    ) -> dict[str, Any]: ...


class DistractorEnhancerBackend(Protocol):
    name: str

    def enhance(
        self,
        *,
        sample: dict[str, Any],
        evidence: list[dict[str, Any]],
        prompt: str,
    ) -> dict[str, Any]: ...


class FocusedRewatchBackend(Protocol):
    name: str

    def inspect(
        self,
        *,
        video: dict[str, Any],
        candidate: dict[str, Any],
        request: dict[str, Any],
    ) -> dict[str, Any]: ...


class EvidenceVerifierBackend(Protocol):
    name: str

    def verify(
        self,
        *,
        sample: dict[str, Any],
        evidence: list[dict[str, Any]],
        rewatch: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


class TextOnlyAnswerBackend(Protocol):
    name: str

    def answer_text(self, *, sample: dict[str, Any]) -> str: ...


class FrozenTargetBackend(Protocol):
    name: str

    def answer_video(
        self,
        *,
        sample: dict[str, Any],
        video: dict[str, Any],
        trial_index: int,
    ) -> str: ...
