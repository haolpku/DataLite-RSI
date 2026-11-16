"""Generation, evaluation and pool-eligibility operators for full RSI rounds."""

from __future__ import annotations

import re
import json
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

from ..backends import (
    DistractorEnhancerBackend,
    EvidenceVerifierBackend,
    FocusedRewatchBackend,
    FrozenTargetBackend,
    QuestionGeneratorBackend,
    TextOnlyAnswerBackend,
)
from ..core import Operator, RunContext
from ..registry import OPERATOR_REGISTRY
from ..task_prompts import TASK_PROMPT_VERSION, prompt_fingerprint, render_task_prompt
from ..task_catalog import TASK_CATALOG


def _normalize_answer(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _content_tokens(value: Any) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", _normalize_answer(value))
        if len(token) > 2
    }


@OPERATOR_REGISTRY.register()
class EvidenceSelectionOperator(Operator):
    name = "evidence_selection"
    version = "1"
    input_keys = ("evidence_units", "task_candidates")
    output_keys = ("grounded_candidates",)

    def run(self, state: dict[str, Any], context: RunContext) -> dict[str, Any]:
        del context
        index = {item["evidence_id"]: item for item in state["evidence_units"]}
        output = []
        for candidate in state["task_candidates"]:
            ids = list(dict.fromkeys(candidate["evidence_ids"]))
            if not ids or any(item not in index for item in ids):
                continue
            selected = sorted(
                (index[item] for item in ids),
                key=lambda item: (item["start_sec"], item["end_sec"]),
            )
            enriched = dict(candidate)
            enriched["selected_evidence"] = selected
            enriched["evidence_window"] = {
                "start_sec": min(float(item["start_sec"]) for item in selected),
                "end_sec": max(float(item["end_sec"]) for item in selected),
            }
            # v0 deliberately does not force a second visual pass. An optional
            # FocusedRewatch branch can populate these fields later.
            enriched["rewatch_requests"] = []
            enriched["rewatch_evidence"] = []
            output.append(enriched)
        return {"grounded_candidates": output}


@OPERATOR_REGISTRY.register()
class TaskComplexityGateOperator(Operator):
    """Cheap pre-LLM gate that removes local candidates before token spend."""

    name = "task_complexity_gate"
    version = "1"
    input_keys = ("grounded_candidates",)
    output_keys = ("complexity_checked_candidates", "complexity_feedback")

    def __init__(self, min_temporal_span_sec: float = 20.0) -> None:
        if min_temporal_span_sec < 0:
            raise ValueError("min_temporal_span_sec must be non-negative")
        self.min_temporal_span_sec = min_temporal_span_sec

    def config(self) -> dict[str, Any]:
        return {"min_temporal_span_sec": self.min_temporal_span_sec}

    @staticmethod
    def _reason(candidate: dict[str, Any], min_span: float) -> str | None:
        task_type = str(candidate.get("task_type", ""))
        spec = TASK_CATALOG.get(task_type, {})
        evidence = candidate.get("selected_evidence", [])
        required = int(spec.get("min_events", 1))
        if len(evidence) < required:
            return "insufficient_evidence_units"
        if int(candidate.get("hop_count", 0)) < int(spec.get("min_hops", 1)):
            return "insufficient_reasoning_hops"
        # Temporal span is retained as metadata for analysis, but is not a v0
        # rejection rule.  A fixed global threshold silently removes valid
        # short state changes; task-aware temporal policies belong to later
        # RSI evolution rounds.
        # These task families require a relationship between observations, not
        # a restatement of one caption. State-change is allowed one observation
        # only when no second grounded entity observation exists; the miner
        # marks such fallback candidates explicitly.
        if task_type in {"state_change", "temporal_relation", "cross_event_comparison", "entity_tracking"}:
            if len(evidence) < 2 and not candidate.get("single_observation_fallback", False):
                return "cross_observation_required"
        return None

    def run(self, state: dict[str, Any], context: RunContext) -> dict[str, Any]:
        del context
        accepted: list[dict[str, Any]] = []
        feedback: Counter[str] = Counter()
        for candidate in state["grounded_candidates"]:
            reason = self._reason(candidate, self.min_temporal_span_sec)
            if reason:
                feedback["rejected_" + reason] += 1
                continue
            accepted.append(candidate)
        feedback["input_candidates"] = len(state["grounded_candidates"])
        feedback["accepted_candidates"] = len(accepted)
        return {
            "complexity_checked_candidates": accepted,
            "complexity_feedback": dict(sorted(feedback.items())),
        }


@OPERATOR_REGISTRY.register()
class TaskSkillRoutingOperator(Operator):
    """Attach an explicit task skill/pipeline route to each candidate."""

    name = "task_skill_routing"
    version = "1"
    input_keys = ("complexity_checked_candidates",)
    output_keys = ("routed_candidates",)

    _ROUTES = {
        "entity_tracking": ("entity_tracking_v1", "entity_tracking_pipeline"),
        "state_change": ("state_change_v1", "state_change_pipeline"),
        "temporal_relation": ("temporal_relation_v1", "temporal_relation_pipeline"),
        "conditional_counting": ("conditional_counting_v1", "counting_pipeline"),
        "dynamic_spatial_trajectory": ("spatial_v1", "spatial_pipeline"),
        "cross_event_comparison": ("comparison_v1", "comparison_pipeline"),
        "causal_event_dependency": ("causal_v1", "causal_pipeline"),
        "multi_hop_reasoning": ("multi_hop_v1", "multi_hop_pipeline"),
    }

    def run(self, state: dict[str, Any], context: RunContext) -> dict[str, Any]:
        del context
        routed = []
        for candidate in state["complexity_checked_candidates"]:
            task_type = str(candidate.get("task_type", "generic"))
            skill, pipeline = self._ROUTES.get(task_type, ("generic_v1", "generic_pipeline"))
            enriched = dict(candidate)
            enriched["skill_ref"] = skill
            enriched["task_pipeline_id"] = pipeline
            routed.append(enriched)
        return {"routed_candidates": routed}


@OPERATOR_REGISTRY.register()
class GroundedTaskBuilderOperator(Operator):
    """Build one cheap, grounded task stream before any generative call.

    v0-slim treats evidence selection, the inexpensive complexity gate and
    task metadata attachment as one composable block.  The underlying
    operators remain in the library for later task-specific graph variants,
    but the baseline no longer serializes three separate stages and artifacts.
    """

    name = "grounded_task_builder"
    version = "1"
    input_keys = ("evidence_units", "task_candidates")
    output_keys = ("routed_candidates", "complexity_feedback")

    def __init__(self, min_temporal_span_sec: float = 20.0) -> None:
        self.min_temporal_span_sec = min_temporal_span_sec

    def config(self) -> dict[str, Any]:
        return {"min_temporal_span_sec": self.min_temporal_span_sec}

    def run(self, state: dict[str, Any], context: RunContext) -> dict[str, Any]:
        selected = EvidenceSelectionOperator().run(state, context)
        checked = TaskComplexityGateOperator(
            min_temporal_span_sec=self.min_temporal_span_sec
        ).run(selected, context)
        routed = TaskSkillRoutingOperator().run(checked, context)
        return {
            "routed_candidates": routed["routed_candidates"],
            "complexity_feedback": checked["complexity_feedback"],
        }


@OPERATOR_REGISTRY.register()
class FocusedRewatchPlanningOperator(Operator):
    """Decide whether a later visual backend should re-inspect short windows."""

    name = "focused_rewatch_planning"
    version = "1"
    input_keys = ("grounded_candidates",)
    output_keys = ("rewatch_planned_candidates",)

    def __init__(self, max_window_sec: float = 30.0) -> None:
        self.max_window_sec = max_window_sec

    def config(self) -> dict[str, Any]:
        return {"max_window_sec": self.max_window_sec}

    def run(self, state: dict[str, Any], context: RunContext) -> dict[str, Any]:
        del context
        output = []
        for candidate in state["grounded_candidates"]:
            uncertain = [
                item
                for item in candidate["selected_evidence"]
                if item["timestamp_quality"] == "unanchored_prefix"
                or bool(item["uncertainty"])
            ]
            requests = []
            for item in uncertain:
                start = float(item["start_sec"])
                end = min(float(item["end_sec"]), start + self.max_window_sec)
                requests.append(
                    {
                        "start_sec": start,
                        "end_sec": end,
                        "reason": "timestamp_or_visual_detail_uncertain",
                        "source_evidence_id": item["evidence_id"],
                        "status": "planned",
                    }
                )
            enriched = dict(candidate)
            enriched["rewatch_requests"] = requests
            enriched["rewatch_evidence"] = []
            output.append(enriched)
        return {"rewatch_planned_candidates": output}


@OPERATOR_REGISTRY.register()
class FocusedRewatchOperator(Operator):
    """Execute only the short visual inspections requested by the planner."""

    name = "focused_rewatch"
    version = "1"
    input_keys = ("video", "rewatch_planned_candidates")
    output_keys = ("rewatched_candidates",)

    def __init__(self, backend: FocusedRewatchBackend) -> None:
        self.backend = backend

    def config(self) -> dict[str, Any]:
        return {"backend": self.backend.name}

    def run(self, state: dict[str, Any], context: RunContext) -> dict[str, Any]:
        del context
        output = []
        for candidate in state["rewatch_planned_candidates"]:
            observations = []
            requests = []
            for request in candidate["rewatch_requests"]:
                result = self.backend.inspect(
                    video=state["video"],
                    candidate=candidate,
                    request=request,
                )
                completed = dict(request)
                completed["status"] = "completed"
                requests.append(completed)
                observations.append(
                    {
                        "start_sec": request["start_sec"],
                        "end_sec": request["end_sec"],
                        "description": str(result.get("description") or "").strip(),
                        "uncertainty": str(result.get("uncertainty") or "").strip(),
                        "backend": self.backend.name,
                    }
                )
            enriched = dict(candidate)
            enriched["rewatch_requests"] = requests
            enriched["rewatch_evidence"] = observations
            output.append(enriched)
        return {"rewatched_candidates": output}


@OPERATOR_REGISTRY.register()
class QuestionGenerationOperator(Operator):
    name = "question_generation"
    version = "2"
    input_keys = ("routed_candidates",)
    output_keys = ("generated_samples",)

    def __init__(self, backend: QuestionGeneratorBackend, variants_per_candidate: int = 3) -> None:
        if variants_per_candidate < 1:
            raise ValueError("variants_per_candidate must be positive")
        self.backend = backend
        self.variants_per_candidate = variants_per_candidate

    def config(self) -> dict[str, Any]:
        return {
            "backend": self.backend.name,
            "task_prompt_version": TASK_PROMPT_VERSION,
            "task_prompt_fingerprint": prompt_fingerprint(),
            "variants_per_candidate": self.variants_per_candidate,
        }

    def run(self, state: dict[str, Any], context: RunContext) -> dict[str, Any]:
        del context
        output = []
        for candidate in state["routed_candidates"]:
            for variant_index in range(self.variants_per_candidate):
                variant_candidate = dict(candidate)
                variant_candidate["generation_variant"] = variant_index
                generated = self.backend.generate(
                    candidate=variant_candidate,
                    evidence=candidate["selected_evidence"],
                    rewatch=candidate["rewatch_evidence"],
                    prompt=render_task_prompt(
                        candidate["task_type"],
                        variant_candidate,
                        candidate["selected_evidence"],
                    ),
                )
                question = str(generated.get("question") or "").strip()
                answer = str(generated.get("answer") or "").strip()
                # Gateways/models sometimes return citations under ``evidence``
                # or omit the IDs even though the prompt asked for them.  The
                # candidate itself is already grounded in this exact evidence
                # set, so recover only from that set rather than dropping every
                # otherwise valid question at the hand-off boundary.
                cited = list(dict.fromkeys(generated.get("evidence_ids") or []))
                if not cited:
                    raw_citations = generated.get("evidence") or generated.get("citations") or []
                    if isinstance(raw_citations, list):
                        cited = [
                            item.get("evidence_id") if isinstance(item, dict) else str(item)
                            for item in raw_citations
                        ]
                        cited = [item for item in cited if item]
                if not cited:
                    cited = list(candidate.get("evidence_ids") or [])
                allowed = {item["evidence_id"] for item in candidate["selected_evidence"]}
                if not question or not answer or not cited or any(item not in allowed for item in cited):
                    continue
                choices = generated.get("choices") or generated.get("options") or []
                if isinstance(choices, dict):
                    choices = [choices.get(key, "") for key in "ABCD"]
                # Reject malformed supervision before invoking the distractor
                # model. This saves an expensive call and prevents a later
                # quality gate from being the first place that notices it.
                if len(choices) != 4 or any(not str(item).strip() for item in choices):
                    continue
                if sum(_normalize_answer(item) == _normalize_answer(answer) for item in choices) != 1:
                    continue
                output.append(
                    {
                        "sample_id": f"qa_{len(output) + 1:05d}",
                        "candidate_id": candidate["candidate_id"],
                        "generation_variant": variant_index,
                        "task_type": candidate["task_type"],
                        "question": question,
                        "answer": answer,
                        "choices": [str(item).strip() for item in choices],
                        "rationale": str(generated.get("rationale") or "").strip(),
                        "evidence_ids": cited,
                        "selected_evidence": candidate["selected_evidence"],
                        "rewatch_requests": candidate["rewatch_requests"],
                        "rewatch_evidence": candidate["rewatch_evidence"],
                        "generator": self.backend.name,
                    }
                )
        return {"generated_samples": output}


@OPERATOR_REGISTRY.register()
class DistractorQualityOperator(Operator):
    """Deterministically reject malformed or obvious distractors."""

    name = "distractor_quality"
    version = "1"
    # Validate the latest question/options state.  The enhancement operator
    # runs immediately before this gate; reading generated_samples here would
    # silently bypass all rewritten hard negatives.
    input_keys = ("enhanced_samples",)
    output_keys = ("distractor_checked_samples", "distractor_feedback")

    _GENERIC = {
        "none of the above", "all of the above", "cannot determine",
        "not enough information",
    }

    def _reason(self, sample: dict[str, Any]) -> str | None:
        choices = [str(item).strip() for item in sample.get("choices", [])]
        if len(choices) != 4 or any(not item for item in choices):
            return "must_have_exactly_four_nonempty_choices"
        normalized = [_normalize_answer(item) for item in choices]
        if len(set(normalized)) != 4:
            return "duplicate_choices"
        answer = _normalize_answer(sample.get("answer", ""))
        if not answer or answer not in normalized:
            return "answer_must_match_one_choice"
        question = _normalize_answer(sample.get("question", ""))
        if len(_content_tokens(answer)) >= 4 and SequenceMatcher(None, answer, question).ratio() >= 0.78:
            return "answer_leaked_in_question"
        if any(choice in self._GENERIC for choice in normalized):
            return "generic_meta_distractor"
        # Do not reject lexical near-duplicates here.  A hard negative often
        # shares most of the sentence template with the answer while changing
        # one decisive visual predicate (state, location, count, or relation).
        # Semantic invalidity is handled by the downstream verification and
        # frontier gates; this deterministic character-similarity rule used to
        # discard valid hard negatives.
        lengths = sorted(len(_content_tokens(choice)) for choice in normalized)
        median = lengths[len(lengths) // 2]
        if median and any(length < max(1, int(median * 0.35)) for length in lengths):
            return "option_length_outlier"
        return None

    def run(self, state: dict[str, Any], context: RunContext) -> dict[str, Any]:
        del context
        accepted: list[dict[str, Any]] = []
        feedback: Counter[str] = Counter()
        for sample in state["enhanced_samples"]:
            reason = self._reason(sample)
            if reason:
                feedback["rejected_" + reason] += 1
                continue
            accepted.append(sample)
        feedback["input_samples"] = len(state["enhanced_samples"])
        feedback["accepted_samples"] = len(accepted)
        return {
            "distractor_checked_samples": accepted,
            "distractor_feedback": dict(sorted(feedback.items())),
        }


@OPERATOR_REGISTRY.register()
class DistractorEnhancementOperator(Operator):
    """Use one text-model pass to turn weak options into hard negatives."""

    name = "distractor_enhancement"
    version = "1"
    input_keys = ("generated_samples",)
    output_keys = ("enhanced_samples",)

    def __init__(self, backend: DistractorEnhancerBackend | None = None) -> None:
        self.backend = backend

    def config(self) -> dict[str, Any]:
        return {"backend": self.backend.name if self.backend else "disabled"}

    def run(self, state: dict[str, Any], context: RunContext) -> dict[str, Any]:
        del context
        if self.backend is None:
            return {"enhanced_samples": list(state["generated_samples"])}
        output = []
        for sample in state["generated_samples"]:
            evidence = sample.get("selected_evidence", [])
            prompt = (
                "Rewrite only the three incorrect options of this grounded video "
                "question into plausible hard negatives. Preserve the question, "
                "correct answer, and evidence IDs. Every option must be concrete, "
                "similar in length and specificity, and differ in one visually "
                "plausible detail. Never use none/all of the above, cannot "
                "determine, impossible scenes, or answer paraphrases. Return only "
                "JSON: {\"choices\":[\"A\",\"B\",\"C\",\"D\"]}.\n\n"
                "SAMPLE:\n" + _json(sample) + "\nEVIDENCE:\n" + _json(evidence)
            )
            try:
                result = self.backend.enhance(
                    sample=sample, evidence=evidence, prompt=prompt
                )
            except Exception as exc:
                enriched = dict(sample)
                enriched["distractor_enhancement"] = {
                    "status": "error", "error": str(exc), "backend": self.backend.name
                }
                output.append(enriched)
                continue
            choices = result.get("choices") or result.get("options") or []
            if isinstance(choices, dict):
                choices = [choices.get(key, "") for key in "ABCD"]
            choices = [str(item).strip() for item in choices]
            answer = _normalize_answer(sample.get("answer", ""))
            if len(choices) == 4 and sum(_normalize_answer(item) == answer for item in choices) == 1:
                enriched = dict(sample)
                enriched["choices"] = choices
                enriched["distractor_enhancement"] = {
                    "status": "enhanced", "backend": self.backend.name
                }
            else:
                enriched = dict(sample)
                enriched["distractor_enhancement"] = {
                    "status": "invalid_output", "backend": self.backend.name
                }
            output.append(enriched)
        return {"enhanced_samples": output}


@OPERATOR_REGISTRY.register()
class EvidenceVerificationOperator(Operator):
    name = "evidence_verification"
    version = "1"
    input_keys = ("distractor_checked_samples",)
    output_keys = ("verification_results",)

    def __init__(self, backend: EvidenceVerifierBackend) -> None:
        self.backend = backend

    def config(self) -> dict[str, Any]:
        return {"backend": self.backend.name}

    def run(self, state: dict[str, Any], context: RunContext) -> dict[str, Any]:
        del context
        output = []
        for sample in state["distractor_checked_samples"]:
            verdict = self.backend.verify(
                sample=sample,
                evidence=sample["selected_evidence"],
                rewatch=sample["rewatch_evidence"],
            )
            enriched = dict(sample)
            enriched["verification"] = {
                "valid": bool(verdict.get("valid", False)),
                "answer_supported": bool(verdict.get("answer_supported", False)),
                "unambiguous": bool(verdict.get("unambiguous", False)),
                "video_training_value": bool(verdict.get("video_training_value", False)),
                "reason": str(verdict.get("reason") or ""),
                "verifier": self.backend.name,
            }
            output.append(enriched)
        return {"verification_results": output}


@OPERATOR_REGISTRY.register()
class TextOnlyFilterOperator(Operator):
    name = "text_only_filter"
    version = "1"
    input_keys = ("distractor_checked_samples",)
    output_keys = ("text_only_results",)

    def __init__(self, backend: TextOnlyAnswerBackend) -> None:
        self.backend = backend

    def config(self) -> dict[str, Any]:
        return {"backend": self.backend.name}

    def run(self, state: dict[str, Any], context: RunContext) -> dict[str, Any]:
        del context
        output = []
        for sample in state["distractor_checked_samples"]:
            prediction = self.backend.answer_text(sample=sample)
            enriched = dict(sample)
            normalized_prediction = _normalize_answer(prediction)
            is_correct = normalized_prediction == _normalize_answer(sample["answer"])
            is_uncertain = normalized_prediction in {
                "cannot determine",
                "cannot_determine",
                "uncertain",
                "unknown",
                "insufficient information",
                "not enough information",
            }
            enriched["text_only"] = {
                "prediction": prediction,
                "correct": is_correct,
                "uncertain": is_uncertain and not is_correct,
                "backend": self.backend.name,
            }
            # Text-only agreement is a graded diagnostic signal, not a binary
            # drop in v0.  An abstention is distinct from a confident wrong
            # answer, so the evolution controller can target shortcut-heavy
            # and under-specified questions separately.
            enriched["text_only_label"] = (
                "text_only_match" if is_correct
                else "text_only_uncertain" if is_uncertain
                else "text_only_mismatch"
            )
            output.append(enriched)
        return {"text_only_results": output}


@OPERATOR_REGISTRY.register()
class FrozenTargetFrontierFilterOperator(Operator):
    name = "frozen_target_frontier_filter"
    version = "1"
    input_keys = ("video", "text_only_results")
    output_keys = ("frontier_results",)

    def __init__(
        self,
        backend: FrozenTargetBackend,
        trials: int = 4,
        target_model_name: str = "qwen3-vl-8b-instruct",
    ) -> None:
        if trials < 1:
            raise ValueError("trials must be positive")
        self.backend = backend
        self.trials = trials
        self.target_model_name = target_model_name

    def config(self) -> dict[str, Any]:
        return {
            "backend": self.backend.name,
            "target_model": self.target_model_name,
            "trials": self.trials,
            "frozen": True,
        }

    def run(self, state: dict[str, Any], context: RunContext) -> dict[str, Any]:
        del context
        output = []
        for sample in state["text_only_results"]:
            predictions = [
                self.backend.answer_video(
                    sample=sample,
                    video=state["video"],
                    trial_index=index,
                )
                for index in range(self.trials)
            ]
            correct = sum(
                _normalize_answer(item) == _normalize_answer(sample["answer"])
                for item in predictions
            )
            if correct == self.trials:
                bucket = "too_easy"
            elif correct == 0:
                bucket = "hard_review_required"
            else:
                bucket = "frontier"
            enriched = dict(sample)
            enriched["target_frontier"] = {
                "correct_count": correct,
                "trial_count": self.trials,
                "predictions": predictions,
                "bucket": bucket,
                "backend": self.backend.name,
            }
            # Difficulty is recorded on every candidate and is consumed as a
            # feedback signal; too-easy/too-hard are no longer hard filters.
            enriched["difficulty_label"] = bucket
            output.append(enriched)
        return {"frontier_results": output}


@OPERATOR_REGISTRY.register()
class FrontierCandidateSelectionOperator(Operator):
    """Keep one best frontier candidate per task while retaining diagnostics."""

    name = "frontier_candidate_selection"
    version = "1"
    input_keys = ("frontier_results",)
    output_keys = ("selected_frontier_results",)

    def run(self, state: dict[str, Any], context: RunContext) -> dict[str, Any]:
        del context
        grouped: dict[str, list[dict[str, Any]]] = {}
        for sample in state["frontier_results"]:
            grouped.setdefault(str(sample.get("task_type", "unknown")), []).append(sample)
        selected: list[dict[str, Any]] = []
        for samples in grouped.values():
            # Prefer frontier candidates, then choose the candidate closest to
            # the desired 2/4 signal.  If no frontier exists, retain the best
            # labelled too-easy/too-hard sample for agent feedback instead of
            # dropping the entire task family.
            samples.sort(
                key=lambda s: (
                    0 if s.get("target_frontier", {}).get("bucket") == "frontier" else 1,
                    abs(int(s.get("target_frontier", {}).get("correct_count", 0)) - 2),
                    -float(s.get("evidence_window", {}).get("end_sec", 0.0))
                    + float(s.get("evidence_window", {}).get("start_sec", 0.0)),
                )
            )
            winner = samples[0] if samples else None
            for sample in samples:
                enriched = dict(sample)
                enriched["frontier_selected"] = bool(winner is sample)
                selected.append(enriched)
        return {"selected_frontier_results": selected}


@OPERATOR_REGISTRY.register()
class LocalDedupAndEligibilityOperator(Operator):
    """Apply fixed gates and exact within-record dedup before global pool dedup."""

    name = "local_dedup_and_eligibility"
    version = "1"
    input_keys = ("selected_frontier_results", "distractor_feedback")
    output_keys = ("pool_candidates", "feedback_signals")

    def run(self, state: dict[str, Any], context: RunContext) -> dict[str, Any]:
        del context
        seen = set()
        accepted = []
        feedback: Counter[str] = Counter(state.get("distractor_feedback", {}))
        for sample in state["selected_frontier_results"]:
            # The default v0 graph intentionally omits the unreliable external
            # Gemini verifier.  If an optional verifier was composed, preserve
            # its strict gate; otherwise rely on evidence selection, distractor
            # quality, text-only shortcut filtering and frozen-target evaluation.
            verification = sample.get("verification")
            if verification is not None and not all(
                verification.get(key, False)
                for key in (
                    "valid",
                    "answer_supported",
                    "unambiguous",
                    "video_training_value",
                )
            ):
                feedback["rejected_verification"] += 1
                continue
            if sample["text_only"].get("correct"):
                feedback["text_only_match_signal"] += 1
            elif sample["text_only"].get("uncertain"):
                feedback["text_only_uncertain_signal"] += 1
            else:
                feedback["text_only_mismatch_signal"] += 1
            bucket = sample["target_frontier"]["bucket"]
            feedback[f"difficulty_{bucket}_signal"] += 1
            if not sample.get("frontier_selected", False):
                feedback["rejected_task_variant_not_selected"] += 1
                continue
            key = (_normalize_answer(sample["question"]), _normalize_answer(sample["answer"]))
            if key in seen:
                feedback["rejected_local_duplicate"] += 1
                continue
            seen.add(key)
            accepted.append(sample)
            feedback["accepted_frontier"] += 1
            feedback[f"accepted_task_{sample['task_type']}"] += 1
        return {
            "pool_candidates": accepted,
            "feedback_signals": dict(sorted(feedback.items())),
        }


@OPERATOR_REGISTRY.register()
class FrontierAndDedupOperator(Operator):
    """Collapse final frontier selection and local eligibility into one v0 node."""

    name = "frontier_and_dedup"
    version = "1"
    input_keys = (
        "video",
        "frontier_results",
        "distractor_feedback",
        "complexity_feedback",
    )
    output_keys = ("pool_candidates", "feedback_signals")

    def run(self, state: dict[str, Any], context: RunContext) -> dict[str, Any]:
        selected = FrontierCandidateSelectionOperator().run(state, context)
        result = LocalDedupAndEligibilityOperator().run(selected, context)
        # Preserve the cheap pre-LLM diagnostics in the final round feedback;
        # otherwise the slim wrapper would hide the main early-stage failure
        # signal from the RSI controller.
        feedback = Counter(state.get("complexity_feedback", {}))
        feedback.update(result.get("feedback_signals", {}))
        result["feedback_signals"] = dict(sorted(feedback.items()))
        return result
