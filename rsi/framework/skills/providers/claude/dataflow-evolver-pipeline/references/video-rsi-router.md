# VideoRSI router contract

Apply this reference only when the task metadata sets
`pipeline_skill_profile` to `video-rsi-router`. The input corpus is fixed. A
record must carry a valid `video` artifact reference; it may additionally carry
timestamped captions, entity tracks, observations, events, or an event graph.

## Route records by available evidence

Implement a small deterministic `VideoEvidenceRouterOperator` before any
generative step. It must preserve every input field and attach
`video_rsi_route` plus `route_reason`:

| Evidence present | Route | Required behavior |
| --- | --- | --- |
| timestamped captions and entity/observation records | `caption_entity` | mine evidence units and cross-time task opportunities |
| timestamped captions or observations only | `caption_only` | build direct evidence units; do not invent entity identity |
| event graph with provenance intervals | `event_graph` | use event nodes only when each cited event resolves to source intervals |
| only a video reference | `video_only` | create a bounded visual-evidence request; do not claim unsupported captions |
| malformed or missing provenance | `invalid` | preserve the record with a structured rejection reason; never fabricate evidence |

Do not force all routes through one old baseline graph. Each route may use a
different ordered subset of evidence indexing, task mining, evidence selection,
question generation, distractor construction, and quality checks. Share only
operators whose inputs and semantics are genuinely unchanged.

## Generated record contract

For each candidate, retain `video`, `question`, `choices`, `answer`,
`task_type`, `selected_evidence`, `producer_route`, and a stable `sample_id`.
`selected_evidence` contains source-relative time intervals and provenance IDs.
Generate exactly four nonempty choices; the answer matches exactly one choice.
Construct negatives that are plausible but contradicted by the selected evidence,
not merely lexical paraphrases of the answer. Preserve hard-invalid candidates
with `rejection_reason`; retain hard-valid reserve candidates when a soft quota
or diversity rule excludes them.

## Training-value prior

Schema validity is a feasibility check, **not** a reason to place a record in
the training pool. Build a lightweight, auditable training-value assessment
from the evidence already available to the route. For every hard-valid
candidate, record `visual_grounding_outcome`, `caption_shortcut_risk`,
`distractor_quality`, and `training_value_label` (`high`, `medium`, `low`, or
`reserve`) with a short rationale. These are reviewable signals rather than a
single opaque model verdict.

- Prefer questions whose answer requires an observable event, state, relation,
  count, or temporal ordering in the cited video interval. A caption, OCR
  string, narration, subtitle, or generic world-knowledge statement must not
  be the only evidence needed for the answer.
- Treat a question as high shortcut risk when its answer can be recovered from
  the question wording, an isolated caption/subtitle sentence, or generic
  knowledge without inspecting the video. Keep such records in a reserve pool
  for diagnosis; do not let them dominate the training pool. Do not discard a
  record solely because one noisy text-only probe says so.
- Cited evidence must be local enough to audit. Whole-video or whole-minute
  evidence is appropriate only for genuine synopsis questions; otherwise
  select the minimal supporting interval(s) and make the required relation
  explicit.
- A task lens is a candidate generator, not a quota. Do not emit one generic
  factual question per lens by default. Select lenses and task opportunities
  according to visible evidence and diversify across object/action recognition,
  object/action reasoning, counting, temporal perception/reasoning, spatial
  reasoning, attributes, OCR, and synopsis when the route can support them.
- Questions from the same video must not be near paraphrases or reworded views
  of the same caption. Prefer distinct evidence intervals or distinct visual
  relations; cap correlated same-video variants in the training pool.
- Distractors must be visually plausible alternatives that differ in a
  checkable event, object, attribute, count, state, location, or temporal
  relation. Reject answer restatements, lexical edits, and irrelevant fantasy
  choices even when all four choices are syntactically valid.

## Quality-operator palette

The following are reusable **operator patterns**, not a mandatory fixed graph.
Choose, split, merge, reorder, or omit them only when the route and its probe
evidence justify doing so. Their purpose is to make a pipeline's quality claim
auditable instead of relying on a generator's self-report.

| Operator pattern | Input → output contract | When it is needed |
| --- | --- | --- |
| `EvidenceWindowOperator` | provenance/event candidates → minimal `selected_evidence` intervals plus `evidence_span_sec` | Before generating a local event, relation, count, or state-change question. It must not silently substitute a whole caption chunk for a short event. |
| `FrameVisibilityOperator` | selected intervals + the training/evaluation frame policy → `visible_frame_count`, `frame_visibility_outcome` | When the learner sees sampled frames rather than a decoded full video. Reserve candidates whose decisive evidence is absent from the declared sampling policy. |
| `TaskOpportunityMiner` | evidence windows → typed task plans with capability, entities, relation, and required intervals | Before MCQ generation when a route has many possible tasks. It selects evidence-supported opportunities instead of filling a fixed task-lens quota. |
| `VisualGroundingAuditOperator` | video/frame evidence + MCQ → `visual_grounding_outcome`, rationale | Use a model-agnostic auditor: a remote VLM API, CPU/GPU-served endpoint, or deterministic visual check. Never assume local CUDA or a specific model. It must distinguish direct visual support from caption-only support; unavailable audit is `unknown`, never `grounded`. |
| `CaptionShortcutAuditOperator` | question/options + captions/subtitles → `caption_shortcut_risk`, rationale | Use before training-pool admission. Detect answer-bearing caption/subtitle sentences, question leakage, and generic-knowledge questions; retain high-risk cases as reserve rather than silently deleting them. |
| `DistractorContrastOperator` | MCQ + selected evidence → `distractor_quality`, pairwise contrast reasons | Use when choices are generated by an LLM. Each wrong choice must differ on a checkable visual fact, not merely wording. |
| `SameVideoDiversityOperator` | candidates grouped by video → correlation clusters and per-video selection | Use whenever more than one candidate is emitted per video. Prefer different evidence windows/capabilities and reserve paraphrases or repeated caption views. |
| `TrainingPoolSelector` | audited candidates → `accepted`, `reserve`, and a capability histogram | Use after quality assessment. Maintain yield, but cap shortcut/weak/correlated items and prevent one easy task family from dominating the train set. |

Each operator should preserve its decision fields and rationale in the output
row. A failed or unavailable soft audit routes a record to `reserve` or labels
it `unknown`; it must never be rewritten as successful evidence. Keep schema,
provenance, unique-answer, and exact-duplicate guards deterministic and
separate from these soft quality decisions.

## Training-distribution feedback

Treat the training pool as supervision for an A/B/C/D benchmark, not just a
bag of individually valid MCQs. The evaluator reports answer-position counts,
task-type coverage, frame-visibility outcomes, and same-video correlation for
the fixed probe. These are soft diagnostics: do not discard valid records only
because a small probe is imbalanced, but use a severe position skew, missing
capability coverage, or many correlated variants to change the next route.

- When an answer-position skew is observed, prefer deterministic option
  permutations that update the answer label and preserve option text. Do not
  fabricate new distractors merely to fill an A/B/C/D quota.
- A record whose grounding/shortcut/distractor audit is `unknown` may remain a
  reserve candidate, but is not evidence of training-ready quality for
  promotion. Never map an unavailable audit to a positive outcome.
- Use the declared Video-MME-compatible frame policy in
  `FrameVisibilityOperator`; a decisive event absent from that policy belongs
  in reserve or requires a changed evidence window, not a confident training
  label.
- If one source video contributes several candidates, explicitly cluster them
  by evidence window and visual relation. Retain diverse clusters and reserve
  correlated restatements instead of treating raw sample count as diversity.

## Feedback and evolution

The task-owned evaluator reports route-level counts and rates for evidence
validity, schema failures, answer/option consistency, text-only outcome,
difficulty label, duplicate outcome, accepted samples, reserve samples, and
rejection reasons. Text-only solvability and frozen-target difficulty are
signals, not automatic hard rejections unless the task explicitly declares
otherwise. Keep the evaluator model and its role fixed within one campaign.
Declare `target` only when the endpoint serves the actual downstream training
model; a general-purpose remote API is a `reference`, not a target. Reference
judgments may guide generic quality but must not be reported as target-model
gains. With no model auditor, keep model-dependent labels `unknown`.
Promotion must not treat raw schema-validity or accepted count as training
quality. Preserve a viable yield floor, then compare the fixed probe using its
training-ready frontier: novel records with grounded evidence, non-dominant
shortcut risk, and discriminative distractors. Route coverage, text-only rate,
difficulty mix, and raw schema-validity remain diagnostics, but a proposal that
increases yield by filling the pool with low-value caption-derived items must
not be promoted. When the runtime has no reliable signal provider, keep these
labels as `unknown` and request a small reviewed sample before claiming a
quality improvement.

Use these signals to modify any of four peer surfaces: route policy, task
contract, operator/prompt, or task-specific pipeline graph. A candidate may add,
remove, split, merge, or replace operators when route-level evidence justifies
it. Do not weaken schema, source-provenance, media-integrity, unique-answer, or
deduplication invariants to increase yield.

`decision.json` must name the route policy revision and each route's operator
sequence in `params`. This prevents incompatible cached prefixes from being
reused after route behavior changes.
