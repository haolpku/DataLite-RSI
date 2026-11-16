# Named feedback sources

The evolution loop keeps feedback sources independent. A primary review runs
after every successful candidate. Any number of command sources may evaluate
the selected dataset at their own intervals. Their scores are never averaged
into the primary `review_score`; a Python selection policy may inspect each
source explicitly. The next PipelineAgent receives the latest result of every
configured command source, labeled by name, plus the selected candidate's
primary review evidence and issues.

Configure additional command sources in the task's private YAML:

```yaml
feedback_sources:
  api-quality:
    kind: command
    command: [python, /absolute/path/to/quality_eval.py]
    benchmarks: [quality]
    interval: 1
    trains_model: false
    baseline: true
    attribution: false
    timeout_sec: 3600
    resources: API only
  sft-math:
    kind: command
    command: [/absolute/path/to/python, /absolute/path/to/sft_eval.py]
    benchmarks: [math500]
    interval: 2
    trains_model: true
    baseline: false
    attribution: true
    resources: 2 GPUs
```

No `feedback_sources.criteria` entry is needed or accepted. With no injected
evaluator, the primary review is the built-in `ReviewAgent`. Its quality
criteria come from `task.quality_criteria`, and sampling, validation and
optional embedding settings come from `review`. `candidate_evaluator` is an
optional Python object supplied by the caller, not another YAML source. When
present, it replaces `ReviewAgent` for each candidate. Its `passed`,
`review_score`, `issues`, and
`domain_feedback` become the primary review result, so no separate criteria
review is run. This is how image/video tasks provide task-specific scoring.
Named command sources are additive: they evaluate the selected incumbent at
their configured intervals and do not replace this primary review.
The primary review's `(passed, review_score)` comparison remains the default
selection rule; use `selection_policy.kind: python` to select using other
sources. Source names must be unique. Disabled sources may set `enabled: false`.

Each `command` source uses the existing external evaluation protocol:
`<command> --request <request.json> --result <result.json>`. The request now
includes `source`, `stage` (`baseline` or `periodic`), the evaluated
iteration, dataset and pipeline paths, expected benchmark names, and
`training_performed`. The result must report `status: ok` and a finite score
for each configured benchmark or metric. For named sources, `model`, count
fields, and `bad_cases` are optional; `metric` defaults to `score`, and the
source name labels results with no model. The `downstream_eval` shortcut keeps
its stricter legacy result contract. Scores, optional summary, and optional
bad-case attribution remain separate by source. A no-training
source's baseline receives the raw input corpus; a training source's baseline
receives no candidate dataset. A failed external evaluation is recorded as an
infrastructure failure, not a negative quality score. Current loop execution
starts all sources due at the same checkpoint concurrently, including enabled
baselines. It waits for all results before selection or the next PipelineAgent
call, then records them in configuration order. There are no cross-iteration
background jobs. `resources` documents requirements but does not allocate
GPUs or prevent competing sources from using the same device.

Within a run, `delta_previous` and `delta_baseline` are computed only against
results from the **same source**. Audit files live under
`native/runs/<run_name>/feedback_sources/<name>/checkpoint_NNN/`. The native
manifest lists source definitions in `feedback_sources` and all results in
`feedback_evaluations`; `feedback_evaluations.jsonl` records each checkpoint.
The Python selection
context's `feedback` entries include `source` and sanitized scores, never raw
bad cases.

## Existing downstream shortcut

The existing `downstream_eval` block remains a quick way to configure one
training/evaluation command. When enabled, it is translated into a named
`command` source (default name `downstream`, overridable with
`downstream_eval.source_name`) and retains its existing audit directory
`native/runs/<run_name>/downstream_eval/`. It can be enabled alongside
additional `feedback_sources` entries. Its legacy `downstream_evaluation`
manifest section and `downstream_evaluations.jsonl` record only this shortcut;
other sources stay in the generic result list. A duplicate name is rejected.
