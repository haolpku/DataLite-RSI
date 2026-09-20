from __future__ import annotations

import json

from .schema import RubricDefinition


def build_system_prompt(rubric: RubricDefinition) -> str:
    axis_lines = "\n".join(
        f"- {axis.axis_id} ({axis.name_en}): {axis.description}"
        for axis in rubric.axes
    )
    environment_lines = "\n".join(
        f"- {item}" for item in rubric.environment_check
    )
    schema = {
        "environment_valid": True,
        "environment_failure_reason": "",
        "axes": {
            axis.axis_id: {
                "score": 4,
                "reason": "one specific reason",
                "evidence": ["an observable piece of evidence"],
            }
            for axis in rubric.axes
        },
        "freeform_failures": ["a free-form description of a failure"],
        "rubric_gap": {
            "detected": False,
            "reason": "",
            "suggested_axis": "",
        },
    }
    return f"""You are a strict reviewer of image-edit data. You are given the
BEFORE image followed by the AFTER image and the instruction, plus, where
available, the intended initial and end states and the preserve/change lists.
Any of those supporting fields may be marked "(not supplied by the planner)".
When one is, judge without it rather than penalising the sample for its absence,
and fall back on the instruction as the statement of what was licensed to change.

The two images are the only source of truth. Your job is to judge whether AFTER
completes the instruction relative to BEFORE — not whether the picture matches
some original creative intent.

First judge environment validity only:
{environment_lines}

If the environment is invalid, set environment_valid to false and explain why;
still return 0 on all three quality axes, but this sample is not attributed to a
failure of the edit model.

If the environment is valid, score each axis from 1 to 5:
{axis_lines}

Scoring scale:
5 = fully satisfied with no visible problems; 4 = satisfied, only minor issues;
3 = broadly satisfied but with obvious problems; 2 = major failure;
1 = not satisfied at all; 0 = not scored, environment invalid.

For visual consistency you must first establish which target object or attribute
was licensed to change, then check only for changes outside that scope. Visual
quality covers artifacts, blur, seams, melting, human anatomy, physics,
lighting, shadows, reflections, and perspective. Whether text or numeric content
is correct belongs to instruction_following; whether text renders naturally and
legibly belongs to visual_quality.

freeform_failures may describe any failure in your own words; you are not
restricted to a fixed tag vocabulary. If IF/VC/VQ cannot express an important,
recurring evaluation gap, set rubric_gap.detected to true.

Return strict JSON only, no Markdown. JSON structure:
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""
