"""LLM-backed attribution of downstream benchmark failures.

This agent converts bounded evaluator evidence into a compact, data-side
diagnosis. Raw bad cases stay in the checkpoint audit files and are never
forwarded to the PipelineAgent.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rsi.framework.evolution.agents.base import AgentABC
from rsi.framework.evolution.evaluation.downstream_eval import (
    DownstreamAttributionResult,
    DownstreamEvaluationResult,
)
from rsi.framework.evolution.models import ReviewResult, TaskSpec
from rsi.framework.evolution.prompts import build_downstream_attribution_prompt


_ALLOWED_CATEGORIES = {
    "missing_context",
    "unsupported_answer",
    "reasoning_depth",
    "coverage_gap",
    "data_noise",
    "format_or_parsing",
    "distribution_mismatch",
    "other",
}


class DownstreamAttributionAgent(AgentABC):
    """Turn benchmark failures into bounded, actionable pipeline feedback."""

    name = "downstream_attribution"

    def __init__(
        self,
        serving,
        *,
        max_retries: int = 2,
        max_findings: int = 6,
        max_actions: int = 8,
    ) -> None:
        super().__init__(serving, max_retries=max_retries)
        self.max_findings = max(1, int(max_findings))
        self.max_actions = max(1, int(max_actions))

    def build_prompt(
        self,
        task: TaskSpec,
        feedback: DownstreamEvaluationResult,
        pipeline_operators: list[dict[str, Any]] | None = None,
        parent_review: ReviewResult | None = None,
    ) -> str:
        return build_downstream_attribution_prompt(
            task=task,
            feedback=feedback,
            pipeline_operators=pipeline_operators or [],
            parent_review=parent_review,
            max_cases_per_benchmark=feedback.benchmarks
            and max(
                1,
                max(len(item.bad_cases) for item in feedback.benchmarks.values()),
            )
            or 0,
            max_findings=self.max_findings,
            max_actions=self.max_actions,
        )

    def parse(
        self,
        raw: Any,
        feedback: DownstreamEvaluationResult | None = None,
    ) -> DownstreamAttributionResult:
        if not isinstance(raw, dict):
            return DownstreamAttributionResult(
                status="failed",
                error="attribution response must be a JSON object",
            )

        summary = _clean_text(raw.get("summary"), 1200)
        findings: list[dict[str, Any]] = []
        raw_findings = raw.get("findings", [])
        valid_evidence_ids: set[str] = set()
        if feedback is not None:
            for benchmark_name, benchmark in feedback.benchmarks.items():
                valid_evidence_ids.update(
                    f"{benchmark_name}#{index}"
                    for index in range(1, len(benchmark.bad_cases) + 1)
                )
        if isinstance(raw_findings, list):
            for item in raw_findings[: self.max_findings]:
                if not isinstance(item, dict):
                    continue
                category = _clean_text(item.get("category"), 64).lower()
                if category not in _ALLOWED_CATEGORIES:
                    category = "other"
                diagnosis = _clean_text(item.get("diagnosis"), 900)
                actions = _clean_list(
                    item.get("recommended_actions"), self.max_actions, 500
                )
                evidence_ids = _clean_list(item.get("evidence_ids"), 8, 80)
                if valid_evidence_ids:
                    evidence_ids = [
                        item for item in evidence_ids if item in valid_evidence_ids
                    ]
                if not diagnosis and not actions:
                    continue
                confidence = item.get("confidence", 0.0)
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    confidence = 0.0
                if not math.isfinite(confidence):
                    confidence = 0.0
                confidence = max(0.0, min(1.0, confidence))
                findings.append(
                    {
                        "category": category,
                        "data_side": bool(item.get("data_side", True)),
                        "confidence": round(confidence, 4),
                        "evidence_ids": evidence_ids,
                        "diagnosis": diagnosis,
                        "recommended_actions": actions,
                    }
                )

        non_data_causes = _clean_list(raw.get("non_data_causes"), 6, 500)
        recommendations = _clean_list(
            raw.get("recommended_pipeline_changes"),
            self.max_actions,
            600,
        )
        if not summary and findings:
            summary = "；".join(
                item["diagnosis"] for item in findings[:3] if item["diagnosis"]
            )
        if not summary and not findings and not non_data_causes:
            return DownstreamAttributionResult(
                status="failed",
                error="attribution response contained no usable diagnosis",
            )
        return DownstreamAttributionResult(
            status="ok",
            summary=summary,
            findings=findings,
            non_data_causes=non_data_causes,
            recommended_pipeline_changes=recommendations,
        )

    def attribute(
        self,
        *,
        task: TaskSpec,
        feedback: DownstreamEvaluationResult,
        pipeline_operators: list[dict[str, Any]] | None = None,
        parent_review: ReviewResult | None = None,
        evidence_path: str | Path | None = None,
    ) -> DownstreamAttributionResult:
        prompt = self.build_prompt(
            task,
            feedback,
            pipeline_operators=pipeline_operators,
            parent_review=parent_review,
        )
        raw = self.run_json(prompt)
        result = self.parse(raw, feedback)
        if evidence_path is not None:
            path = Path(evidence_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "checkpoint_iteration": feedback.checkpoint_iteration,
                        "stage": feedback.stage,
                        "prompt": prompt,
                        "raw_response": self.last_raw_response,
                        "parsed": raw,
                        "result": asdict(result),
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        return result


def _clean_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _clean_list(value: Any, limit: int, item_limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    for item in value[:limit]:
        text = _clean_text(item, item_limit)
        if text:
            output.append(text)
    return output
