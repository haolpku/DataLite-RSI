from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from ...serving.base import LLMServingABC
from .state import StrategyState


SYSTEM_PROMPT = """You write the standing instructions for an image-editing
instruction planner.

You are given a small set of policies, each with a trigger (when it applies), a
procedure (what to do), and a boundary (when it must not be applied). Turn them
into a single block of clear, compact prose that will be appended to the
planner's system prompt.

Rules:

- One short paragraph per policy, in the order given.
- State the trigger, then the procedure, then the boundary. Do not merge two
  policies into one sentence.
- Do not add new rules, examples, or advice. Do not drop a boundary.
- Do not editorialize about which policy matters more.
- Write in plain imperative English. No Markdown, no headers, no bullets with
  leading dashes — just paragraphs.

Return only the block of prose, nothing else."""


DIVERSITY_SYSTEM_PROMPT = """You write one policy for an image-editing
instruction planner.

The run is converging: the accepted samples keep being edited into the same few
categories. You are given the running counts. Write one policy that tells the
planner to favour the categories the run has produced least, or to propose a new
one, while keeping the instruction appropriate to the target.

The policy has three parts:

- trigger: when it applies (always, or when a specific condition holds)
- procedure: what to do — favour the least-produced categories, or propose a new
  one, without forcing an inappropriate edit
- boundary: when it must not be applied — when the target genuinely only fits a
  common category, or when the instruction would become unnatural

Return JSON only:

{
  "trigger": "...",
  "procedure": "...",
  "boundary": "..."
}"""


@dataclass(slots=True)
class CommitOutcome:
    called: bool
    text: str = ""
    error: str = ""
    cost: Any = None
    prompt: str = ""
    response: str = ""


def build_user_prompt(state: StrategyState) -> str:
    policies = state.active_policies()
    lines = ["The policies to render, in order:", ""]
    for i, policy in enumerate(policies, start=1):
        lines.append(f"{i}. trigger: {policy.trigger.strip()}")
        lines.append(f"   procedure: {policy.procedure.strip()}")
        if policy.boundary.strip():
            lines.append(f"   boundary: {policy.boundary.strip()}")
        lines.append("")
    return "\n".join(lines).strip()


def build_diversity_prompt(state: StrategyState) -> str:
    return (
        "The accepted samples so far have been edited into these categories:\n"
        + json.dumps(
            dict(sorted(state.diversity_categories.items(), key=lambda kv: -kv[1])),
            ensure_ascii=False,
            indent=2,
        )
        + "\n\nWrite one policy that tells the planner to favour the categories "
        "the run has produced least, or to propose a new one, while keeping the "
        "instruction appropriate to the target."
    )


def commit(
    *,
    llm: LLMServingABC,
    ctx: Any,
    state: StrategyState,
    batch_id: str,
) -> CommitOutcome:
    prompt = build_user_prompt(state)
    try:
        response = ctx.call_with_retry(
            llm.generate,
            [prompt],
            system=SYSTEM_PROMPT,
            op_name=f"evolve-commit:{batch_id}",
        )[0]
    except Exception as exc:  # noqa: BLE001 — a failed render must not end the run
        return CommitOutcome(called=True, error=str(exc), prompt=prompt)

    return CommitOutcome(
        called=True,
        text=response.content.strip(),
        cost=response.cost,
        prompt=prompt,
        response=response.content,
    )


def commit_diversity(
    *,
    llm: LLMServingABC,
    ctx: Any,
    state: StrategyState,
    batch_id: str,
) -> CommitOutcome:
    prompt = build_diversity_prompt(state)
    try:
        response = ctx.call_with_retry(
            llm.generate,
            [prompt],
            system=DIVERSITY_SYSTEM_PROMPT,
            response_format_json=True,
            op_name=f"evolve-commit-diversity:{batch_id}",
        )[0]
    except Exception as exc:  # noqa: BLE001
        return CommitOutcome(called=True, error=str(exc), prompt=prompt)

    return CommitOutcome(
        called=True,
        text=response.content.strip(),
        cost=response.cost,
        prompt=prompt,
        response=response.content,
    )
