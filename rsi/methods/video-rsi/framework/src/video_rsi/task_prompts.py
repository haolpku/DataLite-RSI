"""Task-specific, versioned question-generation specifications.

Adapted from TraceAV-Bench's useful principles: explicit task definitions,
necessary evidence, temporal span, entity bridges, answerability and grounded
distractors. It does not require every question to be multi-hop.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


TASK_PROMPT_VERSION = "6"

_COMMON = """
You are the question generator in a grounded Video Data RSI pipeline. Generate
exactly one training question from the supplied evidence. Timestamped Caption
spans and structured Entity observations are the only source of truth. Do not
use world knowledge, likely intentions, unseen events, or facts merely suggested
by appearance.

The question must be video-dependent: every cited evidence unit must contribute
to the answer or remove a material ambiguity. Complexity may come from visual
specificity, a state transition, a temporal relation, a condition, a comparison,
or a genuinely necessary multi-hop chain. Do not force multiple hops when one
well-grounded observation supports a precise, non-trivial question. When the
candidate contains multiple evidence units, prefer a question that requires at
least two independent units; a question answerable from one caption alone is
a weak candidate and should be rejected. For multi-evidence candidates, both
evidence units must be necessary; a decorative second unit is invalid. If the
candidate complexity score is below 2, return reject=true instead of lowering
the standard.

Keep the question self-contained, natural, specific, and answerable by an
attentive viewer. Name entities with a short grounded description rather than
an empty name. Do not expose evidence IDs in the question. Do not ask about
audio unless explicit audio evidence is present.

Use four multiple-choice options A-D. Distractors must be hard negatives: each
should describe a visually plausible alternative involving the same entity,
action, state, or scene type, with similar specificity, length, and grammar as
the correct option. Do not use impossible scenes, generic negations ("none of
the above", "cannot determine"), answer paraphrases, or options that can be
rejected from wording alone. Do not repeat the answer or enumerate the counted
events in the question stem. Exactly one option is correct for v0. The answer
text must state the content, not only an option letter. Cite every necessary
evidence unit and no irrelevant unit. If evidence cannot support a unique answer, return
{\"reject\":true,\"reason\":\"...\"} instead of guessing.
""".strip()

TASK_PROMPTS: dict[str, str] = {
    "entity_tracking": """
TASK DEFINITION — ENTITY TRACKING
Track one stable person, animal, object, or place across at least two separated
observations. Ask about a change or comparison in visible appearance, role,
location, relation, or state. Use concrete identity cues such as face/hair,
clothing plus another cue, object markings, setting, or named role. Do not
inherit an action from an earlier observation into a later one. Both the identity
bridge and the relevant observation must be needed for the answer.
""",
    "state_change": """
TASK DEFINITION — STATE CHANGE
Ask about an observable transition between before/precondition and after/outcome:
an object opens or closes, a person changes location, an item is picked up or
put down, or a visible configuration changes. Both states must be directly
described by evidence. Do not turn intention, emotion, speech, or a presumed
cause into a state. Ask for both endpoints or the exact resulting state.
""",
    "temporal_relation": """
TASK DEFINITION — TEMPORAL RELATION
Ask for an order or interval relation among non-identical observations: before,
after, during, between, first/last, or overlap. Derive the answer from global
timestamps and content, never evidence IDs or listing order. A single interval
may support a precise temporal question; use multiple intervals when comparison
is necessary.
""",
    "conditional_counting": """
TASK DEFINITION — CONDITIONAL COUNTING
Ask for the number of distinct visible occurrences satisfying an explicit
condition, such as an action by a specified entity after a trigger, an appearance
in a location, or a visible state change. Define the counting unit so repeated
camera views are not extra occurrences. Count visible occurrences, not mentions
in prose, and cite every counted occurrence plus the condition evidence.
""",
    "dynamic_spatial_trajectory": """
TASK DEFINITION — DYNAMIC SPATIAL / TRAJECTORY
Ask about a visible path, direction, intermediate location, relative movement,
or destination. Use ordered observations to reconstruct movement; do not infer
an unseen route between frames or a reason for moving. Distinguish a path or
endpoint from a static position and cite the relevant timestamped observations.
""",
    "cross_event_comparison": """
TASK DEFINITION — CROSS-EVENT COMPARISON
Compare two separated observations of the same entity, object, place, or
activity. Ask for one precise similarity or difference in action, state,
appearance, role, relation, or context. Both sides must matter: removing either
side should make the answer unsupported or ambiguous. Avoid generic questions;
require a concrete contrast.
""",
    "causal_event_dependency": """
TASK DEFINITION — VISIBLE EVENT DEPENDENCY
Ask about a dependency between a directly visible precondition/action and a
visible later outcome, or which earlier visible action preceded an outcome. Use
cautious wording such as “what visible change followed” unless evidence directly
establishes causation. Never invent motivation, intention, or an off-screen cause.
""",
    "multi_hop_reasoning": """
TASK DEFINITION — MULTI-HOP ENTITY–EVIDENCE REASONING
Construct a chain of at least three necessary evidence units connected by entity
identity, relation, state, location, or time. This task specifically requires
composition: no single unit or keyword may reveal the answer. Do not add a
decorative third hop. Distractors should break one chain link while remaining
plausible in the video.
""",
}


def prompt_fingerprint() -> str:
    payload = {"version": TASK_PROMPT_VERSION, "prompts": TASK_PROMPTS}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def render_task_prompt(
    task_type: str, candidate: dict[str, Any], evidence: list[dict[str, Any]]
) -> str:
    if task_type not in TASK_PROMPTS:
        raise KeyError(f"no task-specific prompt registered for {task_type!r}")
    variant_index = int(candidate.get("generation_variant", 0))
    variant_instruction = (
        "Produce a cross-evidence comparison or temporal contrast; do not ask for a single local fact."
        if variant_index % 3 == 0
        else "Produce a state, relation, or trajectory question whose answer requires combining the cited intervals."
        if variant_index % 3 == 1
        else "Produce a conditional visual question grounded only in the cited evidence; both evidence sides must matter."
    )
    output_contract = """
Return only one JSON object:
{
  "reject": false,
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer": "...",
  "correct_option": "A",
  "evidence_ids": ["exact evidence_id"],
  "rationale": "why the cited evidence uniquely supports the answer"
}
""".strip()
    return (
        _COMMON
        + "\n\n"
        + TASK_PROMPTS[task_type].strip()
        + "\n\nVARIANT REQUIREMENT:\n"
        + variant_instruction
        + "\n\nSKILL ROUTE:\n"
        + str(candidate.get("skill_ref", "generic_v1"))
        + " / "
        + str(candidate.get("task_pipeline_id", "generic_pipeline"))
        + "\n\n"
        + output_contract
        + "\n\nCANDIDATE METADATA:\n"
        + json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
        + "\n\nGROUNDED EVIDENCE:\n"
        + json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
    )
