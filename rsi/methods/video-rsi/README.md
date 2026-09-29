# VideoRSI

## Motivation

Improving video understanding through supervised fine-tuning depends strongly on
the quality, grounding, and diversity of the training data. For video tasks,
constructing such data is difficult because a useful training example must
connect a question and answer to visual evidence distributed across time.

VideoRSI studies recursive self-improvement at the **data-pipeline level**. The
pipeline, rather than model weights, is iteratively improved to construct better
video-grounded SFT data. This design avoids repeatedly training a target model
while exploring pipeline changes, reducing the cost of recursive improvement for
multimodal video tasks.

## Method

VideoRSI transforms a video corpus into a high-quality SFT pool through five
stages.

### 1. Temporal visual evidence

Each source video is divided into chronological 60-second chunks.
`CaptionOperator` produces detailed captions with relative timestamps.
`EntityExtractionOperator` extracts entities, entity updates, and events from
the current chunk and its caption, while maintaining stable entity identities
across chunks. The resulting captions, entity tracks, and events form a
timestamped evidence store for every video.

### 2. Evidence-based task construction

The pipeline indexes caption spans, entity observations, and events as evidence
units. Task candidate mining retrieves evidence units appropriate for video
understanding tasks, including action recognition, object reasoning, counting,
temporal reasoning, OCR, and spatial reasoning.

For each candidate, the grounded task builder creates a task record with the
selected evidence, task type, and answer target. The question generator then
produces a multiple-choice video question with evidence references and source
video provenance.

### 3. Question and option refinement

Generated questions are checked for schema validity, answer consistency, and
evidence references. A distractor-enhancement stage constructs more challenging
incorrect options, followed by a distractor-quality check. This stage removes
duplicate, malformed, and weak question-option sets.

### 4. Selection and data aggregation

VideoRSI evaluates whether a question can be solved from text alone and measures
its difficulty with a frozen target model. The frontier filter combines these
signals with task eligibility and duplicate detection. Local and global
deduplication produce a diverse SFT data pool while preserving evidence lineage
for each accepted example.

### 5. Recursive pipeline improvement

Feedback is collected from candidate yield, evidence and schema validation,
question filtering, text-only checks, frozen-target difficulty, duplicate
rejection, and accepted frontier-and-novel samples under a fixed budget. These
signals are used to update task definitions, prompts, operators, and pipeline
routing in the next iteration. The target model is not repeatedly trained during
this pipeline-search process.

## What recurses

| Component | Recursively modified? |
| --- | --- |
| Pipeline operators, prompts, and routing | **Yes** — updated from per-iteration diagnostics |
| Output SFT dataset | **Yes** — as a consequence of running the improved pipeline |
| Source video corpus | No — fixed for the whole run |
| Frozen target used for difficulty filtering | No — not trained during pipeline search |
| Downstream benchmark labels | No — never used for in-loop selection |

## Public implementation

This contribution includes a portable implementation under `framework/`:

```text
rsi/methods/video-rsi/
|-- method.json
|-- README.md
|-- configs/video-rsi-reference.yaml
`-- framework/
    |-- src/video_rsi/       # evidence, task, operator and pipeline runtime
    |-- drivers/             # versioned runs, fixed evaluation, Codex evolution
    |-- configs/             # task priors and task-specific skill packs
    |-- tests/               # deterministic unit/integration tests
    `-- docs/                # pipeline and evolution design
```

The framework contains no videos, API credentials, model weights, checkpoints,
or generated data pools. Those must be supplied at runtime through explicit
paths and environment variables. See [`framework/README.md`](framework/README.md)
and [`framework/drivers/README.md`](framework/drivers/README.md) for the
execution boundary and commands.

## External services

Captioning, entity extraction, question generation, and distractor refinement
need a vision-language serving endpoint. Frozen-target difficulty filtering
needs a local copy of the target video-language model. Training recipes and
model-specific benchmark adapters are intentionally outside this first public
framework contribution.

## Safety, rollback, and budget

- **Pipeline search does not train the target.** Any downstream training is a
  separate experiment, outside the search loop.
- **Text-only and schema filters** drop questions that are ungrounded or
  solvable without the video.
- **Budget** is a fixed five iterations. There is no convergence criterion.
- **Human intervention** is reported as none.

## Known limitations

- **External assets are required.** Caption/entity corpora, video files,
  serving endpoints and model weights are not redistributed.
- **Downstream benchmarks are out of loop.** Pipeline updates use yield and
  filter diagnostics rather than held-out benchmark labels.
- **No end-to-end GPU smoke test** ships with this method contribution; the
  included deterministic tests cover pipeline contracts, evolution decisions,
  and pool lineage without paid APIs.
