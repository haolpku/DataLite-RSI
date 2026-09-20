from __future__ import annotations

import re
from enum import Enum
from typing import Any, Iterable

from ...core.operator import OperatorABC


class CompatibilityVerdict(str, Enum):
    COMPATIBLE = "compatible"
    ALREADY_SATISFIED = "already_satisfied"
    TARGET_ABSENT = "target_absent"
    AMBIGUOUS_TARGET = "ambiguous_target"
    EDIT_NOT_OBSERVABLE = "edit_not_observable"


STRUCTURAL_VERDICTS = frozenset(
    {
        CompatibilityVerdict.ALREADY_SATISFIED.value,
        CompatibilityVerdict.TARGET_ABSENT.value,
        CompatibilityVerdict.AMBIGUOUS_TARGET.value,
        CompatibilityVerdict.EDIT_NOT_OBSERVABLE.value,
    }
)

_QUOTE_RE = re.compile(r"""["'“”‘’]([^"'“”‘’]{1,120})["'“”‘’]""")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_GENERIC_TARGET_RE = re.compile(
    r"\b(?:specified|requested|indicated|target|new)\s+"
    r"(?:text|phrase|content|wording|segment)\b"
    r"|指定(?:文字|文本|短语|内容|字段)|某段文字|另一段",
    flags=re.IGNORECASE,
)


def quoted_literals(text: str) -> list[str]:
    out: list[str] = []
    for match in _QUOTE_RE.finditer(text or ""):
        value = " ".join(match.group(1).split()).strip()
        if len(value) >= 2 and value not in out:
            out.append(value)
    return out


def _change_clause(instruction: str) -> str:
    text = instruction or ""
    markers = ("CRITICAL preserve", "Critical preserve", "关键保留")
    for marker in markers:
        if marker in text:
            return text.split(marker, 1)[0]
    return text


def _report(
    verdict: CompatibilityVerdict,
    reason: str,
    *,
    checker: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "verdict": verdict.value,
        "reason": reason,
        "checker": checker,
        "evidence": evidence or {},
    }


def assess_seed_compatibility(
    *,
    base_prompt: str,
    edit_code: str,
    edit_hint: str,
    diag_slice: str = "",
) -> dict[str, Any]:
    base_literals = quoted_literals(base_prompt)
    hint_literals = quoted_literals(edit_hint)
    code = (edit_code or "").upper()
    slice_name = (diag_slice or "").lower()

    if code == "T4" and not _CJK_RE.search(base_prompt or ""):
        return _report(
            CompatibilityVerdict.TARGET_ABSENT,
            "Translation edit requires source Chinese text, but the base scene "
            "does not declare any Chinese characters.",
            checker="seed_lint_v1",
            evidence={"base_literals": base_literals, "diag_slice": diag_slice},
        )

    if code in {"T1", "T3", "T5"} and _GENERIC_TARGET_RE.search(edit_hint or ""):
        if not hint_literals:
            return _report(
                CompatibilityVerdict.AMBIGUOUS_TARGET,
                "The text edit does not name an exact target string, so the "
                "planner may choose a value already present in the base scene.",
                checker="seed_lint_v1",
                evidence={
                    "base_literals": base_literals,
                    "hint_literals": hint_literals,
                    "diag_slice": diag_slice,
                },
            )

    if code == "T5" and slice_name == "text_misspelled" and not hint_literals:
        return _report(
            CompatibilityVerdict.AMBIGUOUS_TARGET,
            "A misspelling-repair seed must declare both the misspelled source "
            "and the exact corrected target.",
            checker="seed_lint_v1",
            evidence={"base_literals": base_literals, "diag_slice": diag_slice},
        )

    return _report(
        CompatibilityVerdict.COMPATIBLE,
        "No structural seed/edit conflict was found by the static lint.",
        checker="seed_lint_v1",
        evidence={"base_literals": base_literals, "hint_literals": hint_literals},
    )


def assess_recipe_compatibility(
    *,
    base_prompt: str,
    edit_code: str,
    edit_hint: str,
    instruction_en: str,
    change_text: Iterable[str] = (),
    diag_slice: str = "",
    model_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    change_clause = _change_clause(instruction_en)
    base_literals = quoted_literals(base_prompt)
    target_literals = quoted_literals(change_clause)
    overlap = [
        target
        for target in target_literals
        if any(target.casefold() == current.casefold() for current in base_literals)
    ]
    if overlap and (edit_code or "").upper() in {"T1", "T4", "T5"}:
        return _report(
            CompatibilityVerdict.ALREADY_SATISFIED,
            "The requested target text is already declared in the base scene; "
            "the edit would not create an observable state transition.",
            checker="recipe_lint_v1",
            evidence={
                "overlap_literals": overlap,
                "base_literals": base_literals,
                "target_literals": target_literals,
            },
        )

    if isinstance(model_report, dict):
        raw_verdict = str(model_report.get("verdict") or "").strip().lower()
        if raw_verdict in {item.value for item in CompatibilityVerdict}:
            report = {
                "verdict": raw_verdict,
                "reason": str(model_report.get("reason") or ""),
                "checker": "recipe_llm_v1",
                "evidence": model_report.get("evidence") or {},
            }
            return report

    seed_report = assess_seed_compatibility(
        base_prompt=base_prompt,
        edit_code=edit_code,
        edit_hint=edit_hint,
        diag_slice=diag_slice,
    )
    if seed_report["verdict"] != CompatibilityVerdict.COMPATIBLE.value:
        return seed_report

    if (
        (edit_code or "").upper() in {"T1", "T3", "T5"}
        and _GENERIC_TARGET_RE.search(change_clause)
        and not target_literals
    ):
        return _report(
            CompatibilityVerdict.AMBIGUOUS_TARGET,
            "The derived instruction still does not name an exact text target.",
            checker="recipe_lint_v1",
            evidence={"change_text": list(change_text)},
        )

    return _report(
        CompatibilityVerdict.COMPATIBLE,
        "The requested change has a distinct, observable target.",
        checker="recipe_lint_v1",
        evidence={"target_literals": target_literals},
    )


class SeedEditCompatibilityGate(OperatorABC):

    name = "seed_edit_compatibility_gate"

    def run(
        self,
        storage,
        ctx,  # noqa: ARG002
        *,
        sample_ids: list[str] | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> list[str]:
        sample_ids = sample_ids or list(storage.list_samples())
        accepted: list[str] = []
        for sid in sample_ids:
            sample = storage.read_sample(sid)
            payload = sample.meta.pair_meta
            step = sample.steps[0] if sample.steps else None
            environment_valid = (
                step.raw.get("environment_valid") if step else None
            )
            if not isinstance(environment_valid, bool):
                environment_valid = False
                failure_reason = "planner did not return an environment verdict"
            else:
                failure_reason = str(
                    step.raw.get("environment_failure_reason") or ""
                )

            payload["environment_valid"] = environment_valid
            payload["environment_failure_reason"] = failure_reason
            if environment_valid:
                accepted.append(sid)
                sample.meta.tags = [
                    tag for tag in sample.meta.tags if tag != "environment_invalid"
                ]
            elif "environment_invalid" not in sample.meta.tags:
                sample.meta.tags.append("environment_invalid")
            storage.write_sample(sample, update_index=False)
            self._mark_done(storage, sid)
        return accepted
