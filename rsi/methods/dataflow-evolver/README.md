# DataFlow-Evolver

**Evolve the pipeline, not the dataset.**

Most data-centric self-improvement loops mutate a dataset directly.
DataFlow-Evolver instead treats the *data-processing program* as the object under
recursion: a coding agent writes Python operators, the framework executes them
against a fixed raw corpus, and an independent reviewer scores the dataset that
comes out. What survives an iteration is code, so an improvement is inspectable,
re-runnable, and attributable to a specific operator change rather than to a diff
over rows.

```text
      incumbent pipeline --> agent writes challenger --> execute on fixed corpus
             ^                                                    |
             |                                                    v
      accept if better <-- review_score <-- ReviewAgent + embedding statistics
             |
             +-- rejected: condensed summary enters history, code is discarded
```

## What recurses

| Component | Recursively modified? |
| --- | --- |
| Pipeline operator code | **Yes** -- rewritten every iteration |
| Output dataset | **Yes** -- as a consequence of running the pipeline |
| Raw input corpus | No -- fixed for the whole run |
| Model weights | No -- the base model is never updated in-loop |
| Reviewer, prompts, thresholds | No -- fixed by configuration |

Three separate model roles, easy to conflate:

1. **Pipeline agent** -- a coding agent that authors the operators (Codex CLI
   driving `gpt-5.6-sol` at effort xhigh in the reference run). Sees the parent's
   code, the parent's review, the condensed evolution history, and downstream
   diagnostics.
2. **Teacher LLM** -- called *from inside* the generated pipeline to rewrite or
   derive solutions (`gpt-4o` in the reference run).
3. **ReviewAgent** -- scores the resulting dataset (`gpt-5.6-sol`). Never sees the
   pipeline code, so it cannot be argued with.

## Acceptance rule

```text
llm_composite     = 0.40*correctness + 0.25*relevance + 0.15*difficulty + 0.20*schema
embedding_quality = 0.40*das + 0.30*vendi + 0.30*nn_spread
review_score      = (0.75*llm_composite + 0.25*embedding_quality) * hard_metric_factor
```

`hard_metric_factor` is 0.0 if any threshold is violated -- correctness, schema,
relevance, null rate, n-gram contamination, or duplication -- and 1.0 otherwise. A
single violation makes a candidate unacceptable no matter how good its soft
scores are.

A challenger replaces the incumbent when it passes and scores strictly higher.
Ties keep the incumbent. Rejected candidates leave a condensed summary in the
evolution history so later iterations do not retry a failed strategy, but their
code is not inherited -- every challenger branches from the current best.

The acceptance rule is described precisely in the `feedback_signal` field of
`method.json` and in the `review_score` formula in the reference configuration.
The upstream source repository contains the full implementation.

## Downstream feedback, deliberately excluded from the score

Every `interval` iterations the current best dataset is handed to an external
command that runs a real SFT plus benchmark evaluation and returns per-benchmark
scores with bad cases. Those results are injected into the next proposal prompt
as diagnostic evidence -- and are **kept out of `review_score`**.

The reason is leakage: letting benchmark scores drive acceptance would optimise
the pipeline against the test sets. For the same reason the downstream command
should evaluate a **fixed diagnostic subset** rather than complete test sets, so
that even the diagnostics reaching the proposal prompt expose as little benchmark
content as possible. The reference configuration does exactly this.

The cost is that acceptance rests on a proxy for the signal it actually cares
about. In the reference run the proxy tracked downstream transfer in direction --
each accepted incumbent trained a better model than the last -- but not in exact
rank order across rejected challengers. See the
[result submission](../../../results/submissions/dataflow-evolver-qwen25-7b-math-3k/README.md)
for the measured correlation.

## Scope of this contribution

This directory contains the method manifest, documentation, and the portable
reference configuration:

```text
rsi/methods/dataflow-evolver/
|-- method.json
|-- README.md
+-- configs/math-periodic.yaml   Reference run configuration
```

The full implementation (pipeline-authoring agent driver, DataFlow operator
runtime, SFT/evaluation harness) lives in the upstream source repository at
revision `7b788746b24c2b9d6713709cb64d8c37602a2fbb`, branch
`refactor/open-dataflow-pipeline-runtime`. That repository is not public at the
time of writing, which is a real reproducibility limitation: the configuration
and recorded per-iteration metrics are auditable here, but a full end-to-end
rerun is not yet possible from this repository alone.

## Running the full loop

The full loop needs the upstream package and the environment variables below.
**No credential belongs in the config file.**

```bash
export DF_WORKSPACE_DIR=...        # run artifacts root
export DF_INPUT_PATH=...           # fixed raw corpus (sha256 f9bae97a...)
export DF_RUN_NAME=...             # unique; the framework refuses to reuse one
export DF_API_URL=... DF_API_KEY=... DF_REVIEW_MODEL=...
export DF_PIPELINE_API_URL=... DF_PIPELINE_API_KEY=... DF_PIPELINE_MODEL=...
export DF_AGENT_BASE_URL=... DF_AGENT_API_KEY=...
export CODEX_BIN=... CODEX_MODEL=... CODEX_BASE_URL=...
export DF_DAS_EMBEDDING_URL=... DF_DAS_PROXY_PATH=... DF_DAS_EMBEDDING_MODEL=...
export DF_BENCHMARK_DATA_DIR=...   # benchmark test files, for decontamination
export DF_AGENT_PYTHON=...

python -m dataflow_evolver.main \
  --config rsi/methods/dataflow-evolver/configs/math-periodic.yaml \
  --task task.txt --log-level INFO
```

## External services

A coding-agent endpoint (Anthropic- or OpenAI-compatible, depending on backend);
a chat endpoint for the ReviewAgent; a chat endpoint for the teacher model; and an
OpenAI-compatible embedding endpoint. Periodic downstream evaluation additionally
needs GPUs plus separate SFT and evaluation environments -- the framework shells
out to an external command and imports no training, CUDA, or evaluation
dependency itself. The reference run used 8x A100-80GB with GPU 7 held for the
embedding service, LLaMA-Factory 0.9.3 for SFT, and the
[vendored Qwen2.5-Math harness](../../../evaluation/qwen25_math/) for evaluation.

## Cost profile

From the reference run (5 iterations, 14,181-row corpus, 6,000-row LLM pool cap,
3,000-row output):

| Component | Cost |
| --- | --- |
| Pipeline evolution | 3.53 h wall clock (25-63 min per iteration) |
| Downstream checkpoint | ~17-19 min each (SFT + diagnostic-subset evaluation) |
| Embedding review | 91-134 s per iteration |
| Teacher tokens | 88.36M -- **83% of all token spend** |
| Agent tokens | 9.25M, of which 73% cached input |
| Embedding tokens | 8.82M |
| ReviewAgent tokens | 0.22M |

Two consequences for budgeting. Teacher cost scales with the **candidate pool
cap**, not the output size, because generation runs once per pooled row every
iteration -- halving the cap roughly halves the dominant cost. And the agent's
input is heavily cached because repair attempts resume a session instead of
restarting it, so an endpoint without prompt caching makes authoring markedly
more expensive.

## Safety, rollback, and budget

- **Rollback** is inherent: a rejected challenger is discarded and the incumbent
  is untouched. Every iteration writes its own directory, so any earlier pipeline
  and dataset can be recovered.
- **Refuses to reuse a run name** when the target directory is non-empty, so one
  experiment cannot overwrite another's artifacts.
- **Compile-only pre-flight** runs before any full execution; the agent may edit
  artifacts but never bypasses that check.
- **Contamination** is measured per candidate against every benchmark test file
  and enforced as a hard gate.
- **Budget** is a fixed iteration count with a per-iteration wall-clock timeout
  and a bounded number of repair attempts. There is no convergence criterion, so
  the loop always spends its whole budget.

## Known failure modes

Observed in the reference run, all worth anticipating:

- **Deterministic validators over-reject.** The single largest failure. In
  iteration 1 a `\boxed{}` parser treated LaTeX closing delimiters and sentence
  punctuation as text after the answer and cut 6,000 generated rows to 4. A later
  iteration's coverage heuristics cut 6,000 to 1,975. The loop eventually
  converged on a durable division of labour: deterministic checks handle only
  decidable format and integrity constraints, and an independent LLM audit judges
  mathematical correctness.
- **Correctness and difficulty trade off.** Tightening the audit raised
  correctness 0.78 -> 0.93 while difficulty fell 0.68 -> 0.58, and the reviewer
  flagged template lock-in. The iteration that pushed difficulty back up dropped
  correctness to 0.68 and was rejected. The loop did not resolve this.
- **The proxy score is not perfectly rank-correlated with downstream transfer.**
  It moved the incumbent in the right direction at every step of the reference
  run, but a rejected challenger turned out to score highest on the full test
  sets. Expect the acceptance signal to be directionally sound and imperfect at
  fine-grained ranking; a cheap static proxy standing in for a GPU-day
  SFT-plus-evaluation cycle cannot be more than that.
- **Agent-authored code fails at runtime.** Six repair invocations across five
  iterations. Budget for them.
