from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from ..core.operator import OperatorABC
from ..core.schema import Sample, VerifyReport


_TAG_RULES: list[tuple[str, str, Optional[str]]] = [
    (
        "text_too_small",
        r"\b(?:text|font|letters?|caption|title)\b[^.]{0,30}?\btoo\s+small\b"
        r"|\btiny\s+text\b|\billeg(?:ible|ibility)\b"
        r"|\b(?:text|letters?)\b[^.]{0,30}?\bvery\s+small\b",
        "a4",
    ),
    (
        "text_clipped",
        r"\b(?:text|caption|title|word)s?\b[^.]{0,15}?\b(?:clipped|cropped|cut\s+off|truncated)\b",
        "a4",
    ),
    (
        "OCR_mismatch",
        r"\b(?:OCR\s*mismatch|OCR[-_ ]mismatch"
        r"|wrong\s+(?:spelling|word|letters?|text)"
        r"|incorrect\s+text|different\s+text|garbl|misspell"
        r"|target\s+text\s+(?:wrong|incorrect|differs))",
        "a4",
    ),
    ("text_misspelled", r"\b(?:misspell|spelling\s+error|typo|garbled)\b", "a4"),
    (
        "text_not_legible",
        r"\b(?:not\s+legible|hard\s+to\s+read|blur(?:r|ry|red)?\s+text|text\s+is\s+blur)",
        "a4",
    ),
    (
        "text_added_unrequested",
        r"\b(?:extra\s+text|unrequested\s+text|added\s+text\s+not\s+requested|spurious\s+text)\b",
        "a4",
    ),
    (
        "preservation_drift",
        r"\b(?:preservation|preserved)\b[^.]{0,30}?(?:drift|shift|fail|broken|altered|changed)"
        r"|\b(?:subject|object)\b[^.]{0,15}?(?:shifted|shift|drift)"
        r"|\bnot\s+pixel[\s-]?identical\b"
        r"|\b(?:shape|outline|silhouette|structure)s?\b[^.]{0,15}?(?:changed|altered|different|moved)"
        r"|\b(?:steam|smoke|vapor|foam|bokeh|reflection|hair|fabric)\b[^.]{0,15}?(?:changed|altered|different|deformed)"
        r"|\b(?:deviates?|differs?|diverges?)\s+from\s+(?:the\s+)?(?:original|source|input)\b"
        r"|\b(?:lost|missing)\s+detail\b"
        r"|\bno\s+longer\s+(?:identical|the\s+same)\b",
        "a2",
    ),
    (
        "identity_swap",
        r"\b(?:identity\s+(?:loss|swap|changed|drift|altered)"
        r"|different\s+subject"
        r"|the\s+(?:subject|person|object)\s+(?:looks|is)\s+different"
        r"|features\b[^.]{0,15}?\baltered)",
        "a2",
    ),
    (
        "color_drift",
        r"\b(?:colou?r\s+(?:shift|drift|changed|altered|wrong|different))",
        "a2",
    ),
    (
        "layout_changed",
        r"\b(?:layout\s+(?:changed|moved|altered|shifted|broken)|composition\s+(?:changed|altered))",
        "a2",
    ),
    ("background_changed", r"\bbackground\s+(?:changed|altered|different|replaced)\b", "a2"),
    ("background_lost", r"\bbackground\s+(?:removed|missing|lost)\b", "a2"),
    (
        "physics_violation",
        r"\b(?:physics|gravity|collision|impossible|unphysical|defies\s+physics|floating)\b",
        "a5",
    ),
    (
        "anatomy_error",
        r"\b(?:extra\s+(?:fingers?|limbs?|hands?|arms?|legs?|toes?)"
        r"|missing\s+(?:fingers?|limbs?)"
        r"|distorted\s+(?:hand|face|body)"
        r"|melted\s+(?:fingers?|hand))",
        "a5",
    ),
    (
        "lighting_inconsistent",
        r"\b(?:lighting\s+(?:mismatch|inconsistent|wrong)"
        r"|shadows?\s+(?:wrong|missing|inconsistent|don'?t\s+match))",
        "a5",
    ),
    (
        "perspective_broken",
        r"\b(?:perspective\s+(?:wrong|broken|inconsistent)|skewed\s+perspective)\b",
        "a5",
    ),
    (
        "already_satisfied",
        r"\b(?:already\s+(?:satisfied|correct|present|matches?)"
        r"|target\s+(?:state|text)\s+(?:already|is\s+already)"
        r"|before\s+image\s+already"
        r"|end\s+state\s+(?:equals|matches)\s+(?:the\s+)?initial)\b",
        "a1",
    ),
    (
        "target_absent",
        r"\b(?:target\s+(?:is\s+)?absent|target\s+(?:not\s+found|does\s+not\s+exist)"
        r"|source\s+(?:text|object)\s+(?:is\s+)?(?:absent|missing)"
        r"|no\s+(?:such|matching)\s+(?:text|object|element)\s+(?:is\s+)?present)\b",
        "a1",
    ),
    (
        "ambiguous_target",
        r"\b(?:ambiguous\s+target|target\s+(?:is\s+)?(?:ambiguous|unspecified)"
        r"|exact\s+(?:target|replacement)\s+(?:is\s+)?not\s+specified"
        r"|unclear\s+which\s+(?:text|object|element))\b",
        "a1",
    ),
    (
        "edit_not_observable",
        r"\b(?:edit\s+(?:is\s+)?not\s+observable"
        r"|cannot\s+be\s+verified\s+from\s+(?:the\s+)?(?:images|before)"
        r"|no\s+observable\s+(?:state\s+)?change)\b",
        "a1",
    ),
    (
        "target_not_changed",
        r"\b(?:not\s+(?:visibly\s+)?(?:added|changed|applied|executed)"
        r"|change\b[^.]{0,15}?\bnot\s+visible"
        r"|change\b[^.]{0,15}?\bmissing"
        r"|edit\b[^.]{0,15}?\bnot\s+performed"
        r"|unchanged\s+(?:target|image)"
        r"|target\s+is\s+unchanged)",
        "a1",
    ),
    (
        "partial_change",
        r"\b(?:partial(?:ly)?\s+(?:applied|changed|done)|incomplete\s+(?:change|edit)|only\s+partly)",
        "a1",
    ),
    (
        "change_too_strong",
        r"\b(?:too\s+(?:much|strong|aggressive)|over[-_]?(?:edit|applied)|exaggerated)\b",
        "a1",
    ),
    (
        "unrequested_addition",
        r"\b(?:unrequested\s+(?:addition|element|object|decoration)"
        r"|added\s+(?:object|element)\b[^.]{0,20}?\bnot\s+requested"
        r"|extra\s+(?:object|decor))",
        "a1",
    ),
    (
        "multi_object_confusion",
        r"\b(?:wrong\s+(?:object|subject)|object\s+confusion|swapped\s+objects?)",
        "a1",
    ),
    (
        "count_wrong",
        r"\b(?:wrong\s+count|wrong\s+number|too\s+many|too\s+few|extra\s+copies?|duplicate)\b",
        "a1",
    ),
    (
        "spatial_relation_wrong",
        r"\b(?:wrong\s+side|wrong\s+position|left/right\s+(?:swap|wrong)"
        r"|spatial\s+(?:relation|order)\s+wrong|on\s+wrong\s+side)",
        "a1",
    ),
    (
        "data_value_wrong",
        r"\b(?:wrong\s+(?:number|value|figure|amount|percentage|price)|altered\s+(?:number|value))",
        "a3",
    ),
    (
        "chart_distorted",
        r"\b(?:chart\s+(?:distorted|broken|wrong)|axis\s+(?:wrong|missing)|bars?\s+(?:wrong|swapped))",
        "a3",
    ),
    (
        "background_too_busy",
        r"\b(?:busy\s+background|cluttered\s+background|too\s+many\s+elements)",
        "a4",
    ),
    (
        "artifact_visible",
        r"\b(?:visible\s+artifact|seam|halo|edge\s+artifact|bleeding|smudge)\b",
        "a5",
    ),
    (
        "temporal_inconsistency",
        r"\b(?:flicker(?:s|ing)?"
        r"|frame[\s-]?to[\s-]?frame\s+(?:drift|change|inconsistency)"
        r"|temporal\s+(?:drift|inconsistency|incoherence|jitter)"
        r"|inconsistent\s+(?:between|across)\s+frames"
        r"|jitter(?:y|ing)?\s+(?:identity|appearance)"
        r"|identity\s+drift)\b",
        "a6",
    ),
    (
        "motion_artifact",
        r"\b(?:double\s+exposure"
        r"|motion\s+(?:blur|artifact|smear|glitch)"
        r"|limb(?:s)?\s+(?:morph|duplicat|blend|merge)"
        r"|fingers?\s+(?:morph|swap|blend)"
        r"|warping\s+(?:body|limb)"
        r"|smear(?:ed|y)?\s+motion)\b",
        "a6",
    ),
    (
        "action_not_executed",
        r"\b(?:action\s+(?:not\s+executed|missing|not\s+performed|never\s+(?:happens|occurs))"
        r"|never\s+(?:opens?|closes?|moves?|jumps?|falls?|rolls?)"
        r"|object\s+stays?\s+still\s+despite"
        r"|did\s+not\s+(?:move|open|close|fall|roll|jump|pick(?:\s+up)?))\b",
        "a7",
    ),
    (
        "object_pop_in",
        r"\b(?:pops?\s+(?:in|up|out)\b"
        r"|object\s+(?:pops?|appears?\s+suddenly|materialis|spawns?)"
        r"|new\s+object\s+(?:appears?|introduced)\s+(?:mid|between)\s+(?:clip|frames?)"
        r"|pop[\s-]in\b"
        r"|sudden\s+appearance"
        r"|disappear(?:s|ing|ed)?\s+(?:between|across|mid)[\s-]?(?:clip|frames?))\b",
        "a6",
    ),
    (
        "camera_motion_glitch",
        r"\b(?:camera\s+(?:jumps?|stutters?|warps?|teleports?)"
        r"|sudden\s+camera\s+cut"
        r"|view(?:point)?\s+(?:jump|jitter)"
        r"|parallax\s+(?:wrong|broken))\b",
        "a6",
    ),
]


FAILURE_TAG_VOCAB: list[str] = sorted({tag for tag, _, _ in _TAG_RULES})


_COMPILED: list[tuple[str, re.Pattern[str], Optional[str]]] = [
    (tag, re.compile(pat, flags=re.IGNORECASE), axis) for tag, pat, axis in _TAG_RULES
]


@dataclass(slots=True)
class DiagnosisHit:
    tag: str
    axis: Optional[str]
    matched_text: str


def diagnose_text(text: str, *, axis: Optional[str] = None) -> list[DiagnosisHit]:
    if not text:
        return []
    hits: list[DiagnosisHit] = []
    for tag, pat, ax in _COMPILED:
        m = pat.search(text)
        if m:
            hits.append(
                DiagnosisHit(tag=tag, axis=axis or ax, matched_text=m.group(0))
            )
    return hits


def diagnose_axes_text(report: VerifyReport) -> list[str]:
    if report is None:  # type: ignore[unreachable]
        return []
    seen: dict[str, None] = {}
    for axis_key, axis_score in (report.axes or {}).items():
        score = axis_score.score
        if score is None:
            continue
        if score >= 4.5:
            continue
        text = axis_score.reason or ""
        ax_id = axis_key.split("_", 1)[0]
        for hit in diagnose_text(text, axis=ax_id):
            seen.setdefault(hit.tag, None)
    return list(seen.keys())


class FailureDiagnoser(OperatorABC):

    name = "failure_diagnose"

    def __init__(self, *, force: bool = False, only_badcase: bool = False) -> None:
        super().__init__()
        self.force = force
        self.only_badcase = only_badcase

    def run(  # type: ignore[override]
        self,
        storage,
        ctx,  # noqa: ARG002
        *,
        sample_ids: Optional[list[str]] = None,
        **kwargs,  # noqa: ARG002
    ) -> list[str]:
        sample_ids = sample_ids or list(storage.list_samples())
        touched: list[str] = []
        for sid in sample_ids:
            sample = storage.read_sample(sid)
            if sample.verification is None:
                continue
            if self.only_badcase and sample.meta.judgment.value != "BadCase":
                continue
            if not self.force and sample.verification.failure_tags:
                continue
            tags = diagnose_axes_text(sample.verification)
            if not tags:
                touched.append(sid)
                continue
            sample.verification.failure_tags = tags
            sample = Sample.model_validate(sample.model_dump())
            storage.write_sample(sample, update_index=False)
            self._mark_done(storage, sid)
            touched.append(sid)
        return touched
