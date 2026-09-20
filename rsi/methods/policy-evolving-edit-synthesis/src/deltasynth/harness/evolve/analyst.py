from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from ...serving.base import LLMServingABC
from .state import Hypothesis, Policy, StrategyState, slugify
from .stats import RunStats, SampleCard


MAX_REASON_CHARS = 400


SYSTEM_PROMPT = """You are the strategy analyst for an automated
image-editing data factory.

The pipeline is fixed. For each sample it takes one (scene, edit type) pair,
supplies a BEFORE image, asks a vision model to write one natural-language edit
instruction, sends that instruction to an image-editing model, and has a judge
score the result on three axes: instruction following, visual consistency
outside the edited region, and visual quality. A sample is accepted only if it
clears every axis threshold. The judge and its thresholds are frozen for the
whole run and you cannot change them.

This run works on a single edit type and walks the scene list in order, so you
do not choose what gets attempted. You have exactly one lever: **the
instruction**. It is the only control interface to the editing model, so how a
target is chosen and how the request is worded is where the yield lives. You
influence it by maintaining a small library of policies that get appended to the
instruction planner's prompt.

Your job each batch:

1. Read all ten samples. Say what happened.
2. Give a verdict on **every** open claim in the ledger. Three are allowed:
   `supported` if this batch's samples bear it out, `contradicted` if they tell
   against it, `not_tested` if the batch contained nothing that could decide it.
   You must return one entry per open claim — an omission is recorded as
   `not_tested`, and a claim that goes untested for several batches running is
   retired, because nothing that could settle it is coming up.
3. State any new claim this batch suggests, in the same list, with its statement.
4. Draft a policy for a claim you believe is ready, and reword or withdraw
   policies that are not earning their place.
5. Analyse the diversity of what the accepted samples were edited *into*. The
   object being edited is supposed to vary; the destination is where sameness
   would mean the run is teaching one lesson. Say whether they are converging,
   and classify each accepted sample's destination into a category. Categories
   are yours to create — "metal", "ceramic", "wood", "coral", whatever the
   samples actually show. **Return the counts for this batch only**, not the
   running totals — the running totals are kept for you and shown above.

   You do not write policies about diversity. You analyse; another agent turns
   the analysis into a policy when one is needed.

A claim needs support from two separate batches before its policy can go live.
One batch of ten samples is a single observation, and a claim supported in only
one batch is an accident of that batch. You cannot override this; it is checked
in code after you answer.

`supported` is a strong verdict, not a default. It means this batch's samples
genuinely bear the claim out — not merely that nothing contradicts it. A claim
so loosely worded that every batch supports it is not a claim: it accumulates
evidence it has not earned and takes up room in every instruction from then on.
If a batch tells against one of your claims, say so.

The same discipline applies to what you add. You are not expected to produce a
claim or a policy every batch, and an empty `ops` list is a good answer on a
good batch. If this batch accepted 9 or 10 of its 10 samples, the current
setup is working — leave it alone, unless the samples show something genuinely
broken and you can name it. A needless policy is not harmless: it rewrites
every instruction from then on and degrades a run that was already producing
good data. The diversity analysis is the one exception — report convergence
whenever you see it, even in a perfect batch, because a run can pass every
sample and still be teaching one lesson.

Every policy needs a boundary: the situation where it must not fire. If you
cannot say where a strategy stops, you have not understood it yet.

Reply with JSON only, no Markdown."""


ANTI_CONSERVATISM_CONSTRAINT = """\
Standing constraint for this run: never propose, keep, or reword a policy
that harms the quality of the edit in order to raise the pass rate —
choosing tiny, barely noticeable edits is the typical case. If an active
policy has this effect, withdraw it, whatever the pass rate says. This
constraint takes no verdicts and cannot be overridden by any policy."""


def build_system_prompt(setting_id: str = "") -> str:
    if setting_id == "policy-evolving":
        return SYSTEM_PROMPT + "\n\n" + ANTI_CONSERVATISM_CONSTRAINT
    return SYSTEM_PROMPT


RESPONSE_SCHEMA = """{
  "reading": "what this batch's samples say happened, a few sentences",

  "diversity": {
    "converging": true | false,
    "reason": "why you say so, or why not",
    "categories": {}
  },

  `categories` is the count for this batch, not the running total. The running
  total is kept for you and shown in the prompt.

  "review": [
    {"hypothesis_id": "kebab-case-id",
     "verdict": "supported|contradicted|not_tested",
     "note": "what in this batch decided it, or why nothing could"},

    {"hypothesis_id": "a-new-claim",
     "statement": "a falsifiable claim about why samples fail or succeed",
     "verdict": "supported",
     "note": "the evidence from this batch"}
  ],

  "ops": [
    {"op": "add_policy",
     "hypothesis_id": "kebab-case-id, must already exist or be stated in review",
     "policy_id": "kebab-case-id",
     "trigger": "the situation the instruction planner will recognise",
     "procedure": "what to do about it, concretely",
     "boundary": "when this must not be applied"},

    {"op": "update_policy",
     "policy_id": "an existing policy id",
     "trigger": "...", "procedure": "...", "boundary": "...",
     "reason": "what the previous wording got wrong"},

    {"op": "remove_policy",
     "policy_id": "an existing policy id",
     "reason": "why it is not earning its place"},

    {"op": "retire_hypothesis",
     "hypothesis_id": "an existing claim id",
     "reason": "why it is not worth carrying any further"}
  ]
}

`review` must contain one entry for every open claim in the ledger. New claims go
in the same list, with a `statement`.

`ops` may be an empty list, and on a good batch it should be."""


def _clip(text: str, limit: int = MAX_REASON_CHARS) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def build_user_prompt(
    *,
    stats: RunStats,
    state: StrategyState,
    cards: Sequence[SampleCard],
    edit_code: str,
    edit_definition: str,
    batch_id: str,
) -> str:
    sections: list[str] = []

    accepted = sum(1 for c in cards if c.is_goodcase)
    sections.append(
        "## The task\n"
        + json.dumps(
            {
                "edit_type": edit_code,
                "edit_definition": edit_definition,
                "batch_id": batch_id,
                "batches_so_far": stats.batches,
                "accepted_this_batch": f"{accepted}/{len(cards)}",
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    sections.append(
        f"## This batch, all {len(cards)} samples\n"
        + json.dumps(
            [_card_view(c) for c in cards], ensure_ascii=False, indent=2
        )
    )

    if state.diversity_categories:
        sections.append(
            "## What accepted samples have been edited into, so far\n"
            + json.dumps(
                dict(
                    sorted(
                        state.diversity_categories.items(),
                        key=lambda kv: -kv[1],
                    )
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n\nThese counts accumulate across batches. A category that keeps "
            "growing is the run teaching one lesson; a new category is the run "
            "finding a new one."
        )

    open_claims = state.open_hypotheses()
    sections.append(
        "## Claims you must give a verdict on\n"
        + json.dumps(
            [
                {
                    "hypothesis_id": h.hypothesis_id,
                    "statement": h.statement,
                    "supported_in_batches": h.support_batches,
                    "contradicted_in_batches": h.oppose_batches,
                    "untested_streak": h.untested_streak,
                }
                for h in open_claims
            ],
            ensure_ascii=False,
            indent=2,
        )
        + f"\n\nAll {len(open_claims)} of these need an entry in `review`. "
        "Repeating a verdict for the same batch does not add evidence; a claim "
        "left out is recorded as not_tested. Claims already promoted are in the "
        "policy list below and do not need reviewing."
    )

    sections.append(
        "## Policies in the planner's prompt right now\n"
        + json.dumps(
            [
                {
                    "policy_id": p.policy_id,
                    "version": p.version,
                    "trigger": p.trigger,
                    "procedure": p.procedure,
                    "boundary": p.boundary,
                    "deployed_at_batch": p.deployed_at_batch,
                    "from_claim": p.source_hypothesis,
                }
                for p in state.active_policies()
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n\nThese shaped every instruction in this batch. Judge them from the "
        "samples above: a policy whose situation kept coming up and whose samples "
        "still failed the same way is a candidate for rewording or withdrawal."
    )

    sections.append("## Reply with exactly this shape\n" + RESPONSE_SCHEMA)
    return "\n\n".join(sections)


def _card_view(card: SampleCard) -> dict[str, Any]:
    return {
        "sample_id": card.sample_id,
        "scene": card.scene_id,
        "accepted": card.is_goodcase,
        "instruction": card.instruction,
        "target": card.target,
        "end_state": card.target_state,
        "axes": card.axes,
        "axes_below_minimum": card.axis_failures,
        "judge_reasons": {k: _clip(v) for k, v in card.axis_reasons.items()},
    }


@dataclass(slots=True)
class ApplyResult:
    changes: list[str] = field(default_factory=list)
    discarded: list[str] = field(default_factory=list)


def _op_label(raw: dict[str, Any]) -> str:
    name = str(raw.get("op") or "?")
    for key in ("policy_id", "hypothesis_id"):
        value = str(raw.get(key) or "").strip()
        if value:
            return f"{name}({key}={value})"
    return name


def _parse_json(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("analyst did not return a JSON object")
    return value


_VERDICTS = {
    "supported": "support",
    "support": "support",
    "contradicted": "oppose",
    "oppose": "oppose",
    "opposed": "oppose",
    "not_tested": "not_tested",
    "not-tested": "not_tested",
    "untested": "not_tested",
}


def apply_ops(
    state: StrategyState, payload: dict[str, Any], *, batch_id: str
) -> ApplyResult:
    result = ApplyResult()
    state.diagnosis = str(payload.get("reading") or "").strip()

    diversity = payload.get("diversity")
    if isinstance(diversity, dict):
        state.diversity_converging = bool(diversity.get("converging"))
        state.diversity_reason = str(diversity.get("reason") or "").strip()
        categories = diversity.get("categories")
        if isinstance(categories, dict):
            for name, count in categories.items():
                name = str(name).strip().lower()
                if not name:
                    continue
                try:
                    n = int(count)
                except (TypeError, ValueError):
                    continue
                if n > 0:
                    state.diversity_categories[name] = (
                        state.diversity_categories.get(name, 0) + n
                    )

    _apply_review(state, payload.get("review"), result, batch_id)

    ops = payload.get("ops")
    if ops is None:
        return result
    if not isinstance(ops, list):
        result.discarded.append("'ops' was not a list")
        return result

    handlers = {
        "add_policy": _add_policy,
        "update_policy": _update_policy,
        "remove_policy": _remove_policy,
        "retire_hypothesis": _retire_hypothesis,
    }

    for raw in ops:
        if not isinstance(raw, dict):
            result.discarded.append("an op was not an object")
            continue
        handler = handlers.get(str(raw.get("op") or ""))
        if handler is None:
            result.discarded.append(f"unknown op: {_op_label(raw)}")
            continue
        before = len(result.discarded)
        try:
            handler(state, raw, result, batch_id)
        except Exception as exc:  # noqa: BLE001 — one bad op is not a bad batch
            result.discarded.append(f"{_op_label(raw)} failed: {exc}")
            continue
        for i in range(before, len(result.discarded)):
            result.discarded[i] = f"{_op_label(raw)}: {result.discarded[i]}"

    return result


def _apply_review(
    state: StrategyState, review: Any, result: ApplyResult, batch_id: str
) -> None:
    open_before = {h.hypothesis_id for h in state.open_hypotheses()}
    reviewed: set[str] = set()

    if not isinstance(review, list):
        result.discarded.append("the reply carried no 'review' list")
    else:
        for raw in review:
            if not isinstance(raw, dict):
                result.discarded.append("a review entry was not an object")
                continue
            statement = str(raw.get("statement") or "").strip()
            hid = slugify(
                str(raw.get("hypothesis_id") or "") or statement, fallback="claim"
            )
            verdict = _VERDICTS.get(str(raw.get("verdict") or "").strip().lower())
            if verdict is None:
                result.discarded.append(
                    f"review({hid}): verdict {raw.get('verdict')!r} is not one of "
                    f"supported / contradicted / not_tested"
                )
                continue

            hypothesis = state.hypothesis(hid)
            if hypothesis is None:
                if not statement:
                    result.discarded.append(
                        f"review({hid}): a new claim needs a statement"
                    )
                    continue
                hypothesis = Hypothesis(
                    hypothesis_id=hid, statement=statement, created_batch=batch_id
                )
                state.hypotheses.append(hypothesis)
                result.changes.append(f"opened claim {hid}")
            elif statement and statement != hypothesis.statement:
                hypothesis.statement = statement

            hypothesis.observe(batch_id=batch_id, verdict=verdict)
            reviewed.add(hid)
            label = {
                "support": "supported",
                "oppose": "contradicted",
                "not_tested": "not tested by",
            }[verdict]
            result.changes.append(
                f"{label} {hid} (+{hypothesis.support}/-{hypothesis.oppose} "
                f"batches, untested streak {hypothesis.untested_streak})"
            )

            if verdict == "oppose":
                for policy in state.policies:
                    if policy.active and policy.source_hypothesis == hid:
                        policy.active = False
                        policy.removed_reason = (
                            f"the claim behind it ({hid}) was contradicted"
                        )
                        result.changes.append(
                            f"withdrew policy {policy.policy_id} with its claim"
                        )

    for hid in sorted(open_before - reviewed):
        hypothesis = state.hypothesis(hid)
        if hypothesis is None:
            continue
        hypothesis.observe(batch_id=batch_id, verdict="not_tested")
        result.discarded.append(
            f"review: {hid} was left out, recorded as not_tested "
            f"(streak {hypothesis.untested_streak})"
        )


def _add_policy(
    state: StrategyState, raw: dict[str, Any], result: ApplyResult, batch_id: str
) -> None:
    trigger = str(raw.get("trigger") or "").strip()
    procedure = str(raw.get("procedure") or "").strip()
    if not trigger or not procedure:
        result.discarded.append("needs both a trigger and a procedure")
        return

    hid = slugify(str(raw.get("hypothesis_id") or ""), fallback="")
    if not hid or state.hypothesis(hid) is None:
        result.discarded.append(f"references unknown hypothesis {hid!r}")
        return

    pid = slugify(str(raw.get("policy_id") or "") or trigger, fallback="policy")
    if state.policy(pid) is not None:
        result.discarded.append(
            f"policy {pid!r} already exists; use update_policy to change it"
        )
        return

    state.policies.append(
        Policy(
            policy_id=pid,
            trigger=trigger,
            procedure=procedure,
            boundary=str(raw.get("boundary") or "").strip(),
            source_hypothesis=hid,
            active=False,
        )
    )
    result.changes.append(f"drafted policy {pid} (pending the gate)")


def _update_policy(
    state: StrategyState, raw: dict[str, Any], result: ApplyResult, batch_id: str
) -> None:
    pid = slugify(str(raw.get("policy_id") or ""), fallback="")
    policy = state.policy(pid)
    if policy is None:
        result.discarded.append(f"unknown policy {pid!r}")
        return

    trigger = str(raw.get("trigger") or "").strip()
    procedure = str(raw.get("procedure") or "").strip()
    boundary = str(raw.get("boundary") or "").strip()
    if not (trigger or procedure or boundary):
        result.discarded.append("carried no new wording")
        return

    policy.version += 1
    policy.trigger = trigger or policy.trigger
    policy.procedure = procedure or policy.procedure
    policy.boundary = boundary or policy.boundary
    if policy.active:
        policy.deployed_at_batch = batch_id
    result.changes.append(f"updated policy {pid} to v{policy.version}")


def _remove_policy(
    state: StrategyState, raw: dict[str, Any], result: ApplyResult, batch_id: str
) -> None:
    pid = slugify(str(raw.get("policy_id") or ""), fallback="")
    policy = state.policy(pid)
    if policy is None:
        result.discarded.append(f"unknown policy {pid!r}")
        return
    policy.active = False
    policy.removed_reason = str(raw.get("reason") or "removed by the analyst")
    result.changes.append(f"removed policy {pid}")


def _retire_hypothesis(
    state: StrategyState, raw: dict[str, Any], result: ApplyResult, batch_id: str
) -> None:
    hid = slugify(str(raw.get("hypothesis_id") or ""), fallback="")
    hypothesis = state.hypothesis(hid)
    if hypothesis is None:
        result.discarded.append(f"unknown claim {hid!r}")
        return
    if hypothesis.status == "retired":
        result.discarded.append(f"claim {hid!r} was already retired")
        return
    hypothesis.status = "retired"
    hypothesis.retired_reason = str(raw.get("reason") or "retired by the analyst")
    result.changes.append(f"retired claim {hid}")

    for policy in state.policies:
        if policy.active and policy.source_hypothesis == hid:
            policy.active = False
            policy.removed_reason = f"the claim behind it ({hid}) was retired"
            result.changes.append(
                f"withdrew policy {policy.policy_id} with its claim"
            )


@dataclass(slots=True)
class AnalysisOutcome:
    called: bool
    changes: list[str] = field(default_factory=list)
    discarded: list[str] = field(default_factory=list)
    error: str = ""
    cost: Any = None
    prompt: str = ""
    response: str = ""


def run_analysis(
    *,
    llm: LLMServingABC,
    ctx: Any,
    state: StrategyState,
    stats: RunStats,
    cards: Sequence[SampleCard],
    edit_code: str,
    edit_definition: str,
    batch_id: str,
    setting_id: str = "",
) -> AnalysisOutcome:
    prompt = build_user_prompt(
        stats=stats,
        state=state,
        cards=cards,
        edit_code=edit_code,
        edit_definition=edit_definition,
        batch_id=batch_id,
    )
    try:
        response = ctx.call_with_retry(
            llm.generate,
            [prompt],
            system=build_system_prompt(setting_id),
            response_format_json=True,
            op_name=f"evolve-analysis:{batch_id}",
        )[0]
    except Exception as exc:  # noqa: BLE001 — a failed reflection must not end the run
        return AnalysisOutcome(called=True, error=str(exc), prompt=prompt)

    try:
        payload = _parse_json(response.content)
    except Exception as exc:  # noqa: BLE001
        return AnalysisOutcome(
            called=True,
            error=f"unparseable reply: {exc}",
            cost=response.cost,
            prompt=prompt,
            response=response.content,
        )

    applied = apply_ops(state, payload, batch_id=batch_id)
    return AnalysisOutcome(
        called=True,
        changes=applied.changes,
        discarded=applied.discarded,
        cost=response.cost,
        prompt=prompt,
        response=response.content,
    )
