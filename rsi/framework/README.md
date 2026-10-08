# Pipeline evolution framework

`rsi.framework.run(task_config, resources=...)` is the active application entry;
the CLI is `python -m rsi.framework --task <task.json>`.
It runs the fixed-corpus incumbent/challenger evolution loop for text,
image, video, or mixed structured records. Every active task declares
`method_id: "dataflow-evolver"`; modality
never selects a different loop. Policy and Video source trees remain in
`rsi/methods/` as research references, outside the active runtime import path.

```text
rsi/framework/
├── __init__.py              public Python API
├── cli.py / __main__.py     framework CLI
├── method.json              active evolution manifest
├── core/                    Operator, Pipeline, contracts, checkpoint, provenance
├── schemas/                 active task-envelope JSON Schema
├── io/                      records, blobs, artifacts, serving
├── runtime/                 task loading, run lifecycle, agent contract, skills
├── evolution/
│   ├── controller.py        incumbent/challenger iteration and acceptance
│   ├── runner.py            TaskEnvelope -> evolution Pipeline adapter
│   ├── corpus.py            fixed input discovery
│   ├── models.py            candidate/review data contracts
│   ├── feedback.py          task evaluator interface
│   ├── media.py             local source artifact resolution
│   ├── prompts.py           role-specific prompts and bounded repair
│   ├── agents/              Pipeline, Review, Diagnostic, Attribution agents
│   ├── evaluation/          hard metrics, DAS/Vendi, downstream protocol
│   ├── execution/           generated Pipeline, preflight, cache, observation
│   ├── providers/           lazy Codex/Claude/OpenCode runtimes and sessions
│   ├── telemetry/           pipeline token observations
│   └── utils/               configuration, parsing, logging
├── configs/                 evolution configuration templates
├── skills/                  provider pipeline-authoring packs
└── tasks/                   text and mixed-modality task templates
```

Use the public contracts from `rsi.framework`:

```python
from rsi.framework import (
    BatchedFileStorage, BatchedPipelineABC, FileStorage, OperatorABC, PipelineABC,
    PipelineLLMServing, StreamBatchedFileStorage, StreamBatchedPipelineABC, run,
)
```

Generated operators and pipelines use the step-based contract: an operator
subclasses `OperatorABC` and implements `run(self, storage)`; a pipeline
subclasses `PipelineABC`, builds one `FileStorage`, and calls its operators in
`forward()`. `compile()` records the graph and validates key integrity, then
`forward()` executes it. This is the same contract as `open-dataflow` 1.0.10,
reimplemented here so no `open-dataflow` installation or `dataflow` import is
needed; `tests/parity` pins the observable behavior against the real package.
For a large corpus, pair `BatchedPipelineABC` with `BatchedFileStorage` and call
`forward(batch_size=N, resume_from_last=True)`; pair
`StreamBatchedPipelineABC` with `StreamBatchedFileStorage` when each step should
be streamed in chunks instead of buffered in memory. These classes are active
runtime contracts and are covered by the batched/streaming parity cases. Do not
mix the pipeline and storage variants, and do not use data batches to control
LLM HTTP concurrency.
`LLMServingABC` is the DataFlow graph-discovery contract in
`core/llm_serving.py`; `PipelineLLMServing` is its OpenAI-compatible
implementation in `io/serving.py`. The two are intentionally separate from the
outer evolution-agent serving interface in `evolution/providers/serving.py`.
The earlier key-based `Operator`/`Pipeline` contract in `core/` still drives the
outer evolution loop and remains available to existing artifacts through
`evolution/execution/generated_pipeline.py`.

The outer checkpoint fingerprints the fixed input and resolved config
environment without writing credential values to the manifest. If a service
changes behavior behind unchanged settings, bump the affected operator version.

A text task uses ReviewAgent by default. A task declaring image or video must
pass a `candidate_evaluator` through `resources`; its `review(dataset_path,
task, **kwargs)` returns `CandidateFeedback(score, passed, domain_feedback)`.
The framework owns comparison of `(passed, score)`, checkpoint, provenance,
execution observation, and bounded repair. The evaluator owns visual evidence
and domain gates. Policy IF/VC/VQ and VideoRSI frozen-target/frontier rules have
not yet been integrated into this loop.

Provider authoring skills are stored in `skills/providers/` and copied into a
run-local provider-native path only for PipelineAgent. PipelineDiagnosticAgent
uses a separate built-in system prompt, isolated cwd, disabled project skill
discovery, and read-only behavior. It loads no project skill; the run manifest
records the diagnostic prompt fingerprint when diagnosis is enabled. Its artifact
does not enter ReviewAgent score or best-so-far acceptance. Provider SDKs are
loaded only when their provider runs.
`requirements/core.txt` covers config, the step-based runtime (pandas drives
storage type semantics) and offline evaluation imports;
`requirements/live-review.txt` adds the Review/embedding API clients, and
`requirements/claude-agent.txt` adds Claude's SDK. Codex and OpenCode use
separately installed CLIs. None of these lists requires `open-dataflow`.
For which behaviors are verified equivalent to the reference runtime and which
are not, see [the parity audit](../../docs/dataflow-serving-parity.md).

The task and private configuration templates are under `tasks/` and `configs/`.
The task's `metadata.method_config_ref` names a file in `configs/`. Real inputs,
model services, and credentials are supplied by the caller or environment and
are not stored in this tree. See [the extension guide](../../docs/dataflow-multimodal-extension.md)
for the current multimodal boundary.

## Where a run puts its files

Given a task whose `workspace` is `W` and a run id `R`, the outer run root is
`W/runs/R/`. The evolution loop runs underneath it:

```text
W/runs/R/                              outer run root (RunStore)
├── run_manifest.json                  status, acceptance, fingerprints
├── execution_observation.json         per-stage observation
├── provenance/events.jsonl            append-only provenance
├── records/dataflow/
│   ├── final_dataset.jsonl            exported incumbent dataset
│   └── candidates.jsonl               per-candidate trace
├── diagnostics/                       PipelineDiagnosticAgent artifacts
├── blobs/                             content-addressed media
└── native/                            ← the coding agent's workspace
    ├── .agents/skills/…               run-local authoring skill (codex)
    ├── .claude/skills/…               (claude)
    ├── .opencode/skills/…             (opencode)
    └── runs/evolution/
        ├── iterations.jsonl
        ├── iteration_001/             ← one candidate per directory
        │   ├── pipeline.py            agent-authored entry
        │   ├── operators/             agent-authored operator modules
        │   ├── decision.json          agent-declared operator order
        │   ├── cache/                 pipeline_step_stepN.jsonl artifacts
        │   ├── step_reuse_plan.json   framework's reuse decision
        │   ├── step_manifest.json     per-step identity and hashes
        │   ├── execution_observation.json
        │   ├── review.jsonl           ReviewAgent evidence
        │   ├── sessions/              agent session and tool logs
        │   └── attempt_NN_*.log       compile/execution stdout and stderr
        └── iteration_002/ …
```

Two directories are easy to confuse:

- **The agent's cwd is `W/runs/R/native/`** — the whole run, not one iteration.
  It is set once (`PipelineAgent.workspace_dir`) and reused for every call, which
  is why the run-local authoring skill is mirrored there and why a backend's
  project-skill discovery finds it.
- **The agent writes its artifacts into `…/native/runs/evolution/iteration_NNN/`**,
  the path supplied per call. The framework validates, compiles and executes
  `pipeline.py` with that iteration directory as the subprocess cwd.

`PipelineDiagnosticAgent` is the exception: it runs in a throwaway empty
temporary directory (`isolated_agent_cwd`) so project-skill discovery cannot
reach either the repository or the authoring skill, and it only reads paths it
is explicitly given.

Agent session history lives wherever the backend puts it, which is controlled by
its own home variable (`CODEX_HOME` for codex, for example), not by the
framework. Point that at a per-run directory to keep sessions auditable.

## Provenance

This framework is a refactor of the DataFlow-Evolver research prototype at
commit `ba6e489524bbf5835a06cccad555c5182e4ce4f4` ("isolate diagnostic agents
from pipeline skills"), authorized by the repository owner. It replaces that
prototype; the prototype's source is retained read-only under
`rsi/methods/dataflow-evolver/` as a research reference and is never imported by
the active runtime.

The prototype depended on `open-dataflow==1.0.10`. This runtime implements the
required execution behavior itself, without copying that package's source, and
verifies the result against the real package in an isolated environment — see
[the parity audit](../../docs/dataflow-serving-parity.md). The prototype
repository has no LICENSE file, so `method.json` records `NOASSERTION` pending a
public redistribution decision. No dataset, image, video, weight, credential,
workspace cache, session log, or run result was copied from it.
