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

## Feedback and evolution

The task-owned evaluator reports route-level counts and rates for evidence
validity, schema failures, answer/option consistency, text-only outcome,
difficulty label, duplicate outcome, accepted samples, reserve samples, and
rejection reasons. Text-only solvability and frozen-target difficulty are
signals, not automatic hard rejections unless the task explicitly declares
otherwise. Keep the evaluator and frozen target fixed within one campaign.
Promotion preserves the prior VideoRSI policy: compare the fixed-probe
`accepted_frontier_novel_count` only. Do not substitute route coverage,
text-only rate, difficulty mix, or raw schema-validity for that primary metric;
they are diagnostic evidence used to decide what to change next.

Use these signals to modify any of four peer surfaces: route policy, task
contract, operator/prompt, or task-specific pipeline graph. A candidate may add,
remove, split, merge, or replace operators when route-level evidence justifies
it. Do not weaken schema, source-provenance, media-integrity, unique-answer, or
deduplication invariants to increase yield.

`decision.json` must name the route policy revision and each route's operator
sequence in `params`. This prevents incompatible cached prefixes from being
reused after route behavior changes.
