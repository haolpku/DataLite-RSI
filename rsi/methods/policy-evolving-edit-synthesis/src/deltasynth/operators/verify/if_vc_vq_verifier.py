from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from ...core.operator import OperatorABC
from ...core.parallel import ParallelMixin
from ...core.schema import AxisScore, Judgment, Sample, VerifyReport
from ...rubrics import build_system_prompt, decide_goodcase, load_rubric
from ...serving.base import VLMServingABC


ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_RUBRIC = ROOT.parent / "configs" / "rubrics" / "if_vc_vq.json"


_MISSING = "(not supplied by the planner)"


def build_verify_user_prompt(
    *,
    instruction: str,
    source_state: str,
    target_state: str,
    preserve: list[str],
    change: list[str],
) -> str:

    def render(items: list[str]) -> str:
        return json.dumps(items, ensure_ascii=False) if items else _MISSING

    return (
        f"instruction:\n{instruction}\n\n"
        f"intended initial state: {source_state or _MISSING}\n"
        f"intended end state: {target_state or _MISSING}\n\n"
        f"must preserve:\n{render(preserve)}\n"
        f"must change:\n{render(change)}\n"
    )


def _parse(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict) or not isinstance(
        value.get("environment_valid"), bool
    ):
        raise ValueError("IF/VC/VQ verifier returned an invalid structure")
    return value


class IFVCVQVerifier(ParallelMixin, OperatorABC):
    name = "if_vc_vq_verify"

    def __init__(
        self,
        vlm: VLMServingABC,
        *,
        rubric_path: Path = DEFAULT_RUBRIC,
    ) -> None:
        super().__init__()
        self.vlm = vlm
        self.rubric = load_rubric(rubric_path)
        self.system_prompt = build_system_prompt(self.rubric)

    def run(
        self,
        storage,
        ctx,
        *,
        sample_ids: Optional[list[str]] = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> list[str]:
        sample_ids = sample_ids or list(storage.list_samples())
        pending = self._filter_pending(storage, sample_ids)
        if not pending:
            return sample_ids
        _, failed = self.run_parallel_samples(
            pending,
            self._process_one,
            storage=storage,
            ctx=ctx,
            max_workers=max(1, getattr(ctx, "max_workers", 1)),
            progress_desc=self.name,
        )
        failed_ids = {result.sample_id for result in failed}
        return [sample_id for sample_id in sample_ids if sample_id not in failed_ids]

    def _process_one(self, storage, ctx, sid: str) -> None:
        sample = storage.read_sample(sid)
        if sample.inputs.init_image is None or sample.final_image is None:
            raise ValueError(f"{sid}: missing before/after")
        step = sample.steps[0]
        plan = step.raw.get("instruction_plan") or {}
        user_prompt = build_verify_user_prompt(
            instruction=step.instruction.en or "",
            source_state=str(plan.get("source_state") or ""),
            target_state=str(plan.get("target_state") or ""),
            preserve=step.preserve_text,
            change=step.change_text,
        )
        before = storage.artifact_path(sid, sample.inputs.init_image.path)
        after = storage.artifact_path(sid, sample.final_image.path)
        response = ctx.call_with_retry(
            self.vlm.score,
            [str(before), str(after)],
            op_name=f"if-vc-vq:{sid}",
            prompt=user_prompt,
            system=self.system_prompt,
            response_format_json=True,
            rubric=self.rubric.model_dump(),
        )[0]
        result = _parse(response.content)
        goodcase, average, decision_failures = decide_goodcase(
            self.rubric, result
        )
        axes = {
            axis.axis_id: AxisScore(
                score=float((result.get("axes", {}).get(axis.axis_id) or {}).get("score", 0)),
                reason=str(
                    (result.get("axes", {}).get(axis.axis_id) or {}).get("reason", "")
                ),
                rubric_id=self.rubric.ref,
            )
            for axis in self.rubric.axes
        }
        report = VerifyReport(
            rubric_id=self.rubric.ref,
            axes=axes,
            aggregate=average if goodcase else 0.0,
            judgment=Judgment.GOOD if goodcase else Judgment.BAD,
            verifier_model=getattr(self.vlm, "model", None),
            failure_tags=list(result.get("freeform_failures") or []),
        )
        report.environment_valid = result.get("environment_valid")
        report.environment_failure_reason = result.get(
            "environment_failure_reason"
        )
        report.raw_average = average
        report.decision_failures = decision_failures
        report.rubric_gap = result.get("rubric_gap") or {}
        sample.verification = report
        sample.meta.judgment = report.judgment
        sample.cost.add(response.cost)
        ctx.cost_tracker.add(sid, response.cost)
        sample = Sample.model_validate(sample.model_dump())
        storage.write_sample(sample, update_index=False)
        self._mark_done(storage, sid)
