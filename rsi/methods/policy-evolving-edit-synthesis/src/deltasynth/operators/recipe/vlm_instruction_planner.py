from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from ...core.operator import OperatorABC
from ...core.parallel import ParallelMixin
from ...environment import load_environment_snapshot
from ...serving.base import VLMServingABC


ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_SNAPSHOT = ROOT.parent / "configs" / "environment" / "snapshot.json"

log = logging.getLogger(__name__)


_CONTENT_POOLS_CACHE: Optional[dict[str, dict]] = None


def _load_content_pools() -> dict[str, dict]:
    global _CONTENT_POOLS_CACHE
    if _CONTENT_POOLS_CACHE is not None:
        return _CONTENT_POOLS_CACHE
    pools: dict[str, dict] = {}
    path = os.environ.get("DELTASYNTH_CONTENT_POOLS", "").strip()
    if path:
        try:
            raw = json.loads(Path(path).read_text())
            pools = {k: v for k, v in raw.items() if not k.startswith("_")}
            log.info("content pools loaded for %d edit types from %s", len(pools), path)
        except Exception as exc:
            log.warning(
                "DELTASYNTH_CONTENT_POOLS=%s could not be read, running without "
                "a content menu: %s",
                path,
                exc,
            )
    _CONTENT_POOLS_CACHE = pools
    return pools


REQUIRED_FIELDS: frozenset[str] = frozenset({"environment_valid", "instruction"})

OPTIONAL_FIELDS: dict[str, str] = {
    "environment_failure_reason": "no reason recorded when the gate rejects a pair",
    "target_entity": "sample stops participating in the repeat-pair novelty nudge",
    "source_state": "judge loses the stated starting point",
    "target_state": "judge loses the stated end point, and rewrites lose the intent to keep",
    "instruction_zh": "the Chinese column of the export is empty",
    "preserve": "judge must infer the whole licensed scope from the instruction",
    "change": "judge loses the explicit statement of what had to change",
}

_OUTPUT_SCHEMA = """{
  "environment_valid": true,
  "environment_failure_reason": "",
  "target_entity": "the specific object or attribute you selected",
  "source_state": "the initial state actually observed in the BEFORE image",
  "target_state": "an explicit, verifiable end state different from the initial state",
  "instruction": "the natural edit request, one or two sentences, as a real user would phrase it",
  "instruction_zh": "the Chinese one-to-one rendering of instruction",
  "preserve": ["3-7 things that must stay unchanged"],
  "change": ["1-3 things that must change"]
}"""

_LANGUAGE_RULE = (
    "Language: write every field in English. `instruction_zh` is the only "
    "exception and must be in Chinese."
)

PLAN_SYSTEM_PROMPT = f"""You are an image-edit Instruction Planner. You are
given exactly one real BEFORE image and one Edit type definition. The image is
the only source of truth; never assume anything that is not visible in it.

The Edit definition describes a *type* of edit, not a request that someone has
already filled in. Where it says the target or the new content is "specified",
"given", or "explicitly provided", that means YOU must decide it. Nothing will be
supplied to you beyond the image and the definition, and the absence of a
pre-chosen target is never a reason to reject the image.

Your tasks:
1. Decide whether this image is suitable for performing this Edit. Reject it only
   if the image genuinely lacks anything this Edit could act on.
2. Within the Edit's scope, pick one target in the image that is unambiguous,
   meaningful, observable, and genuinely different from its current state.
3. You may pick a whole object or an independently identifiable local
   attribute. Avoid always reaching for the same kind of target.
4. Write one natural edit instruction.
5. List separately, in `preserve`, the non-target content that must stay
   unchanged.

How to write the instruction: this exact sentence is what gets sent to the image
edit model, so it has to be both natural and sufficient. Phrase it the way a real
user would ask, in one or two sentences, saying directly what to change and what
to change it into. Content that has to be exact, such as target text, numbers, or
colours, must appear in the instruction itself.

Return JSON only:
{_OUTPUT_SCHEMA}

{_LANGUAGE_RULE}

When environment_valid is false you must fill in environment_failure_reason and
source_state; the remaining fields may be empty. Do not emit Markdown or any
text outside the JSON."""

REWRITE_SYSTEM_PROMPT = f"""You are an image-edit Instruction Repair Planner.
The previous edit of this BEFORE image failed. Your job is to revise the
instruction, not to switch to a different task.

You are given the same BEFORE image, the Edit type definition, the previous
instruction, and the failure reasons reported by the judge.

Hard requirements:
1. Keep the same edit target and the same edit intent as the previous attempt.
   Do not switch to another target object and do not change the end state that
   was being aimed at.
2. Revise only the wording, and only in response to the reported failure — for
   example make a vague description concrete, supply exact content that was
   missing, or delimit a scope that was misread.
3. If the failure reasons show that this task cannot be accomplished on this
   image at all, set environment_valid to false and explain why, rather than
   forcing out another instruction.

The instruction stays a natural one or two sentences, and it is sent to the image
edit model exactly as written. `preserve` is a separate record for the reviewer
and is not appended to the instruction, so do not copy it into the instruction as
a checklist; if the failure calls for a narrower scope, narrow it briefly and
naturally inside the sentence.

Return JSON only:
{_OUTPUT_SCHEMA}

Add one further field describing what you changed:
  "revision_note": "what you changed relative to the previous instruction, and
   why that repairs the reported failure"

{_LANGUAGE_RULE}

Do not emit Markdown or any text outside the JSON."""


_DIRECTIVE_HEADER = (
    "Additional standing requirements for this run. They narrow the task; they do "
    "not override anything above."
)


def build_plan_system_prompt(
    directives: Optional[list[str]] = None, *, rewrite: bool = False
) -> str:
    base = REWRITE_SYSTEM_PROMPT if rewrite else PLAN_SYSTEM_PROMPT
    cleaned = [text.strip() for text in (directives or []) if text and text.strip()]
    if not cleaned:
        return base
    lines = "\n".join(f"- {text}" for text in cleaned)
    return f"{base}\n\n{_DIRECTIVE_HEADER}\n{lines}"


def _parse(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("instruction planner did not return a JSON object")
    if not isinstance(value.get("environment_valid"), bool):
        raise ValueError(
            "instruction planner did not return a boolean environment_valid"
        )
    if value["environment_valid"] and not str(value.get("instruction") or "").strip():
        raise ValueError(
            "instruction planner accepted the pair but gave no instruction"
        )
    return value


def build_plan_user_prompt(
    *,
    edit_subcategory: str,
    edit_definition: str,
    rewrite_context: str = "",
    used_end_states: Optional[list[str]] = None,
    content_pool: Optional[dict[str, Any]] = None,
) -> str:
    sections = [
        f"edit_subcategory: {edit_subcategory}",
        f"edit_definition:\n{edit_definition}",
    ]
    if content_pool:
        candidates = [str(c) for c in (content_pool.get("candidates") or []) if str(c).strip()]
        if candidates:
            goal = str(content_pool.get("goal") or "").strip()
            menu = "\n".join(f"- {c}" for c in candidates)
            sections.append(
                "Candidate edit contents for this edit type (benchmark-aligned menu):\n"
                + (goal + "\n" if goal else "")
                + menu
                + "\n\nPick from this menu the entry that best fits what is actually "
                "visible in this BEFORE image — a candidate whose target does not exist "
                "in the image is always the wrong choice. If none of the entries fits "
                "this image naturally, ignore the menu and choose your own content "
                "within the edit's scope. The menu steers WHAT to edit into; it never "
                "changes whether the image is suitable, and it never licenses a "
                "barely-visible edit."
            )
    if used_end_states:
        lines = "\n".join(f"- {s}" for s in used_end_states[-40:])
        sections.append(
            "End states this run has already produced (do not repeat one unless "
            "nothing else fits the target):\n" + lines
        )
    if rewrite_context:
        sections.append(rewrite_context)
    return "\n\n".join(sections)


def build_rewrite_context(
    *,
    prior_instruction: str,
    prior_target_entity: str,
    prior_target_state: str,
    failures: list[str],
    axes: dict[str, Any],
) -> str:
    return (
        f"Previous instruction:\n{prior_instruction}\n\n"
        f"Previously selected target:\n{prior_target_entity}\n\n"
        f"Previously intended end state:\n{prior_target_state}\n\n"
        "Failure reasons reported by the judge:\n"
        f"{json.dumps(failures, ensure_ascii=False, indent=2)}\n\n"
        "Axis scores and reasons:\n"
        f"{json.dumps(axes, ensure_ascii=False, indent=2)}"
    )


class VLMInstructionPlanner(ParallelMixin, OperatorABC):
    name = "vlm_instruction_plan"

    def __init__(
        self,
        vlm: VLMServingABC,
        *,
        environment_snapshot: Path = DEFAULT_SNAPSHOT,
    ) -> None:
        super().__init__()
        self.vlm = vlm
        _, _, edits, _ = load_environment_snapshot(environment_snapshot)
        self.edits = {edit.edit_id: edit for edit in edits}
        self._directives: list[str] = []
        self._custom_system_prompt: str = ""
        self._strategy_ref: str = ""
        self._used_end_states: list[str] = []

    def set_directives(
        self, directives: list[str], *, strategy_ref: str = ""
    ) -> None:
        self._directives = [
            text.strip() for text in directives if text and text.strip()
        ]
        self._strategy_ref = strategy_ref

    def set_system_prompt(
        self, prompt: str, *, strategy_ref: str = ""
    ) -> None:
        self._custom_system_prompt = prompt.strip()
        self._strategy_ref = strategy_ref

    def set_used_end_states(self, end_states: list[str]) -> None:
        self._used_end_states = [s.strip() for s in end_states if s and s.strip()]

    def _resolve_system_prompt(self, rewrite: bool) -> str:
        if self._custom_system_prompt:
            return self._custom_system_prompt
        return build_plan_system_prompt(self._directives, rewrite=rewrite)

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
        if sample.inputs.init_image is None:
            raise ValueError(
                f"{sid}: the before image must exist before planning an instruction"
            )
        payload = sample.meta.pair_meta or {}
        edit_id = str((sample.edit_codes or [payload.get("edit_code")])[0] or "")
        scene_id = str(payload.get("scene_id") or "")
        edit = self.edits.get(edit_id)

        is_rewrite = bool(sample.rewrite_parent_id)
        rewrite_context = ""
        if is_rewrite:
            parent = storage.read_sample(sample.rewrite_parent_id)
            parent_step = parent.steps[0] if parent.steps else None
            parent_plan = (
                (parent_step.raw.get("instruction_plan") or {}) if parent_step else {}
            )
            rewrite_context = build_rewrite_context(
                prior_instruction=(
                    (parent_step.instruction.en if parent_step else "") or ""
                ),
                prior_target_entity=str(parent_plan.get("target_entity") or ""),
                prior_target_state=str(parent_plan.get("target_state") or ""),
                failures=(
                    parent.verification.failure_tags if parent.verification else []
                ),
                axes=(
                    {
                        key: {"score": axis.score, "reason": axis.reason}
                        for key, axis in (parent.verification.axes or {}).items()
                    }
                    if parent.verification
                    else {}
                ),
            )

        user_prompt = build_plan_user_prompt(
            edit_subcategory=(edit.subcategory if edit else ""),
            edit_definition=(edit.description if edit else ""),
            rewrite_context=rewrite_context,
            used_end_states=self._used_end_states,
            content_pool=None if is_rewrite else _load_content_pools().get(edit_id),
        )
        before_path = storage.artifact_path(sid, sample.inputs.init_image.path)
        rubric_id = "instruction_repair_v1" if is_rewrite else "instruction_plan_v3"
        response = ctx.call_with_retry(
            self.vlm.score,
            [str(before_path)],
            op_name=f"instruction-plan:{sid}",
            prompt=user_prompt,
            system=self._resolve_system_prompt(is_rewrite),
            response_format_json=True,
            rubric={"rubric_id": rubric_id},
        )[0]
        plan = _parse(response.content)

        step = sample.steps[0]
        step.raw["instruction_plan"] = plan
        step.raw["environment_valid"] = plan["environment_valid"]
        if self._strategy_ref:
            step.raw["strategy_ref"] = self._strategy_ref
        step.raw["environment_failure_reason"] = (
            plan.get("environment_failure_reason") or ""
        )
        if plan["environment_valid"]:
            step.instruction.en = str(plan.get("instruction") or "")
            step.instruction.zh = str(plan.get("instruction_zh") or "")
            step.preserve_text = list(plan.get("preserve") or [])
            step.change_text = list(plan.get("change") or [])
        if edit is not None:
            sample.subcategory = edit.subcategory

        sample.cost.add(response.cost)
        ctx.cost_tracker.add(sid, response.cost)
        storage.write_sample(sample, update_index=False)
        self._mark_done(storage, sid)
