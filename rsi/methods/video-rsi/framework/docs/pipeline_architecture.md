# Task-centric Video Data RSI Pipeline

## Production path (v0-slim)

```text
Completed Caption + Entity corpus
  -> EvidenceIndexOperator
  -> TaskCandidateMiningOperator
  -> GroundedTaskBuilderOperator (selection + cheap complexity gate + task metadata)
  -> QuestionGenerationOperator (one variant by default)
  -> deterministic question/schema validation
  -> DistractorEnhancementOperator (one-shot hard-negative rewrite)
  -> DistractorQualityOperator (deterministic hard-negative gate)
  -> TextOnlyFilterOperator
  -> FrozenTargetFrontierFilterOperator     (frozen target, four trials)
  -> FrontierAndDedupOperator (frontier selection + local eligibility + dedup)
  -> global dedup
  -> permanent High-quality Data Pool
  -> round feedback and Task/Operator/Pipeline evolution
```

The default v0 graph is intentionally small.  `EvidenceVerificationOperator`,
semantic-event refinement, focused rewatch and task-specific graph routing are
library/experimental components, not mandatory v0 stages.  They can be added
as Lego blocks by a later task pipeline when diagnostics demonstrate a benefit.
The old longer graph remains reproducible through the historical v0-v5
snapshots; new experiments use the `0.5.0-v0-slim-no-gemini-verification`
baseline.

The v0 gates are intentionally permissive about the two previously dominant
failure signals.  A short temporal span is retained as metadata; any
task-aware span policy is left to a later evolution round.  Text-only
evaluation does not reject a sample: it records `text_only_match`,
`text_only_uncertain` (abstention/insufficient text), or `text_only_mismatch`.
Likewise, frozen-target outcomes receive `difficulty_label` values
`too_easy`, `frontier`, or `hard_review_required`; all three can enter the
pool after structural validation and are exposed as feedback signals.

The unit shared by all task families is a grounded `evidence_unit`, not a
semantic event. An evidence unit is a timestamped Caption span or structured
Entity observation/event with exact source provenance. Entity tracks provide
stable cross-minute anchors for identity, state, relation and trajectory
reasoning.

Semantic-event abstraction is an optional representation provider. A task
pipeline may request it when it improves candidate construction, but no task
is required to pass through it.

`FocusedRewatchPlanningOperator` and `FocusedRewatchOperator` are retained as
an optional future branch. They are not in v0 because the existing Caption and
Entity corpus is already the primary visual annotation, and a second visual
pass adds substantial decode/VLM cost without a demonstrated quality gain.

## Is semantic-event extraction necessary?

| Task family | Mandatory? | Default evidence path | Event abstraction hypothesis |
|---|---:|---|---|
| Entity Tracking | No | Entity occurrences across intervals | Can obscure identity evidence by merging participants |
| State Change | No | Entity state snapshots or visible precondition/outcome | May summarize away the exact before/after state |
| Temporal Relation | No | Timestamped evidence intervals | Useful only when relations refer to higher-level activities |
| Conditional Counting | No | Task-specific occurrence grouping | Generic event boundaries can create count leakage or duplicates |
| Dynamic Spatial / Trajectory | No | Ordered entity/location observations | A trajectory should remain continuous rather than be event-cut |
| Cross-event Comparison | No | Two selected evidence groups | Event summaries may reduce prompt length and improve coherence |
| Causal / Event Dependency | No | Visible outcome/precondition evidence | Event abstraction may help define cause/outcome units |
| Multi-hop Reasoning | No | Entity-evidence graph | Event nodes may help some graph topologies but add inference error |

## Event-representation ablation

Run three representation arms on the same stratified videos, task prior,
generator, judges and frozen target:

1. `direct_evidence`: Caption/Entity evidence and entity tracks only;
2. `deterministic_event`: bounded rule-based event proposals;
3. `llm_event`: semantic events refined by a caller-selected language model.

Report results separately for every task family and for duration/source bins:

- representation latency, model tokens and GPU cost;
- candidate yield and FocusedRewatch rate;
- Evidence Verification pass rate and ambiguity rate;
- Text-only rejection rate;
- frontier acceptance rate;
- globally novel accepted samples;
- the number of newly accepted frontier-and-novel samples under the same fixed
  generation budget.

The decision is per task family, not global. An event arm is retained only when
it increases the accepted frontier-and-novel sample count under the same fixed
budget without lowering grounding quality. Do not combine unrelated metrics
into a weighted score.
Until this ablation is complete, `direct_evidence` is the production baseline
and event refinement remains an experimental branch.

## Fixed and evolving components

The evaluation policy and target model remain frozen within an RSI experiment.
The reference implementation calls this role `frozen-video-target`; replacing
the selected model creates a new evaluation campaign rather than a comparable
pipeline iteration.
Round feedback may modify task priors, task definitions, task-specific prompts,
operator prompts,
FocusedRewatch policy, or the pipeline graph. Previously accepted data remains
in the permanent pool when a later pipeline replaces its producer.

## v0 boundary and autonomous evolution

The v0 *data graph* ends after per-video generation, fixed-policy evaluation,
local eligibility, and lineage-preserving output.  The outer RSI controller may
then evolve that graph autonomously; the controller and its model policy are
kept outside the data graph so every candidate remains comparable.

After v0 is stable, a Python round controller invokes a headless Codex CLI in a
stateful multi-turn dialogue:

```text
collect logs/examples/feedback
  -> Codex turn 1: diagnose per-operator failures and propose surfaces
  -> codex exec --json --cd <workspace> --model <coding-model> "..."
  -> Codex turn 2: resume the same session and implement the justified change
  -> Codex turn 3: resume again, run tests and repair the candidate
  -> run smoke data and fixed probe evaluation
  -> accept/reject the candidate Task/Prompt/Operator/Pipeline version
  -> start the next round (or codex exec resume <session_id> on CLI failure)
```

The agent receives both aggregate indicators and concrete failed examples/logs.
It may propose changes at four peer levels: TaskSpec, task-specific PromptSpec,
Operator, and Pipeline graph. The controller, not the agent's free-form
judgement, enforces tests, fixed evaluation policy, target-model immutability,
lineage, and the single primary proxy: accepted frontier-and-novel sample count
under a fixed budget.
The v0 model policy is intentionally small. Concrete model identifiers are
runtime configuration, not part of this public method contract:

| Role | v0 model |
|---|---|
| Coding Agent | configured Codex-compatible coding model |
| Question Generation | configured structured-output language model |
| Task Candidate Mining | deterministic rules; optional configured reranker |
| Text-only Filter | one fixed text model within an experiment |
| General Verification | optional configured verifier |
| Deduplication | configured embedding backend plus deterministic similarity rules |
| Frozen Target | one fixed video-language model within an experiment |

These assignments are recorded in `model_policy.py`; the controller does not
silently switch models between rounds.

`TaskCandidateMiningOperator` is not an LLM call. It is a high-recall,
deterministic operator. Keeping candidate mining deterministic in v0 avoids
spending a model call on every evidence unit; an optional reranker is added only
when diagnostics show that recall alone produces too many weak candidates.

## Operator library, task graphs and skill packs

The next framework layer treats v0 as an immutable baseline graph rather than
the one pipeline that every task must keep patching. `OperatorLibrary` exposes
versioned operator manifests (input/output keys, capabilities, supported task
types and fingerprints). `PipelineGraph` and `PipelineRegistry` store
serializable ordered graphs whose nodes reference library operators as
`OperatorName@version`; backend-bearing operators can still be injected at
runtime. A graph can therefore be forked for one task without changing v0.

`SkillSpec` is the task-facing analogue of a coding skill. A skill pack stores
the task definition, evidence instructions, preferred operators, required
capabilities, evaluation checks and forbidden shortcuts. `SkillRegistry.route`
selects the appropriate skill before a task graph is assembled. Skills contain
guidance and contracts, not hidden executable behavior.

`PipelineMigrationPlanner` converts structured per-task/operator failures into
auditable Lego-style actions such as `fork_pipeline`, `replace_or_insert` and
`update_task_contract`. It does not edit code or decide promotion. The Codex
agent implements the plan in a candidate workspace, adds tests, and the fixed
controller evaluates and promotes or rolls back the candidate. If previous
rounds only changed wording, the planner marks structural exploration as
required so evolution cannot remain a prompt-only loop.
