# DataFlow-Self-Improver

**Diagnose on frozen validation sets, then rewrite the pipeline.**

DataFlow-Self-Improver recursively improves a DataFlow DAG from bad-case
attribution. Two **fixed** validation sets score the current checkpoint; an
analyzer clusters the failures; a coding agent then retrieves matching data,
rewrites the pipeline operators to ingest it, and refines or filters the
incumbent corpus.

```text
  incumbent DAG + corpus
           |
           v
  SFT checkpoint --> score MATH-500 and AIME26 --> attribute bad cases
           |                                              |
           |                                              v
           |                         phase = format alignment | capability completion
           |                                              |
           |                                              v
           +-- agent retrieves data, rewrites DAG, refines/filters corpus
           |
           v
  accept unless val mean acc drops > 3 pp
         or format rate drops > 5 pp vs alignment incumbent
```

## What recurses

| Component | Recursively modified? |
| --- | --- |
| Pipeline operator code (DataFlow DAG) | **Yes** — rewritten so the pipeline can ingest the newly retrieved data |
| Output dataset | **Yes** — refined, filtered, and supplemented each iteration |
| Retrieved source data | **Yes** — the agent may fetch new sources to close attributed gaps |
| Validation sets (MATH-500, AIME26) | No — frozen for the whole run |
| Rollback thresholds | No — fixed by configuration |
| Model weights in-loop | No — weights change only through the external SFT command that produces the next diagnostic checkpoint |

Three roles, kept separate:

1. **Analyzer** — attributes errors on the two frozen validation sets and
   chooses the phase objective.
2. **Pipeline agent** — a coding agent that edits DataFlow operators, retrieves
   matching data, and refines or filters the incumbent corpus.
3. **Teacher LLM** — called from inside the generated pipeline to rewrite or
   derive solutions.

## Two-phase iteration

The analyzer labels the current failure mass as **format** or **capability**,
and that label selects the budget for the next dataset:

| Phase | Objective | Row cap | What the agent may do |
| --- | --- | ---: | --- |
| Format alignment | Schema, boxed answers, parseable conclusions | **< 2,000** | Retrieve data, rewrite the DAG, refine and filter the incumbent |
| Capability completion | Close attributed skill gaps | **< 6,000** | Same actions, but after the first capability incumbent every later iteration may only refine, filter, or supplement **that 6k subset** |

The 6k lock is the method's data-lite constraint. Capability iterations are
not allowed to grow a new corpus from scratch; they edit the subset that
phase 2 already accepted.

The reported run followed that schedule:

| Iteration | Phase (as run) | Dataset | Rows |
| ---: | --- | --- | ---: |
| 1 | Format alignment | MATH-500-oriented subset | 1,800 |
| 2 | Capability completion | AIME26-oriented subset | 5,200 |
| 3 | Capability completion | Combined subset | 7,000 |
| **4** | Capability completion | Combined subset, refined | **6,000** |

Iteration 3 is a combined merge that temporarily exceeded the 6k cap; iteration
4 filtered it back onto the locked 6k subset, which is the selected incumbent.

## Acceptance and rollback

A challenger replaces the incumbent unless either:

1. mean accuracy on the two validation sets falls by **more than 3 percentage
   points**, or
2. the format-success rate falls by **more than 5 percentage points** relative
   to the **alignment-phase** incumbent.

Ties and threshold violations keep the incumbent. Rejected pipelines remain in
the run directory so an earlier DAG and dataset can be restored. The two
validation sets are the only scores that enter this rule. Held-out transfer
numbers are measured after the fact.

Because validation accuracy drives acceptance, MATH-500 and AIME26 are
**in-loop** sets. Headline transfer is the unweighted mean over the seven
held-out benchmarks that never entered the rollback rule: GSM8K, Minerva,
Gaokao, OlympiadBench, AMC23, AIME24, and AIME25. The nine-benchmark mean that
still includes MATH and AIME26 is reported separately and is not the primary
claim. In the reference submission, every reported score is the mean over
three seeds: 42, 42, and 44.

## Scope of this contribution

This directory contains the method manifest, documentation, and the portable
reference configuration:

```text
rsi/methods/dataflow-self-improver/
|-- method.json
|-- README.md
`-- configs/math-badcase.yaml
```

The full implementation (analyzer, pipeline-authoring agent, DataFlow runtime,
SFT and evaluation harness) lives in the upstream source repository and is
not public at the time of writing. The configuration and recorded
per-iteration metrics are auditable here; a full end-to-end rerun is not yet
possible from this repository alone.

## Running the full loop

The full loop needs the upstream package and the environment variables below.
**No credential belongs in the config file.**

```bash
export DF_WORKSPACE_DIR=...
export DF_INPUT_PATH=...
export DF_RUN_NAME=...
export DF_MATH500_PATH=...
export DF_AIME26_PATH=...
export DF_API_URL=... DF_API_KEY=... DF_ANALYZER_MODEL=...
export DF_PIPELINE_API_URL=... DF_PIPELINE_API_KEY=... DF_PIPELINE_MODEL=...
export DF_AGENT_BASE_URL=... DF_AGENT_API_KEY=...
export CODEX_BIN=... CODEX_MODEL=... CODEX_BASE_URL=...
export DF_BENCHMARK_DATA_DIR=...
export DF_AGENT_PYTHON=...

python -m dataflow_self_improver.main \
  --config rsi/methods/dataflow-self-improver/configs/math-badcase.yaml \
  --log-level INFO
```

## External services

A coding-agent endpoint; a chat endpoint for the analyzer; a chat endpoint for
the teacher model inside the generated pipeline; and GPUs plus separate SFT
and evaluation environments for the per-iteration diagnostic command. The
reference evaluation uses the
[vendored Qwen2.5-Math harness](../../../evaluation/qwen25_math/) under the
[math SFT transfer protocol](../../../benchmarks/math-sft-transfer-suite/).

## Safety, rollback, and budget

- **Rollback** is a threshold on the frozen validation sets. A rejected
  challenger leaves the incumbent DAG and dataset untouched.
- **Validation sets stay frozen.** The agent may retrieve new training data; it
  may not edit MATH-500 or AIME26.
- **Phase caps** bound how much data the loop can spend: 2k for format
  alignment, 6k for capability completion, with later capability work locked
  to that 6k subset.
- **Contamination** should be re-checked each iteration against the held-out
  test files and against the two in-loop validation files.

## Known limitations

- **In-loop validation is a leakage channel.** MATH-500 and AIME26 scores can
  rise because the loop optimises against them. Read the held-out primary
  first; treat MATH and AIME26 as diagnostic.
- **Iteration 3 exceeded the 6k cap** before iteration 4 filtered back to 6k.
  The lock is a method rule, not something the reported run enforced on every
  intermediate artifact.
- **Reported scores are three-seed means** (42, 42, 44). Per-seed tables are
  not in this manifest.
- **Small competition sets.** AIME24/25/26 are 30 questions at avg@4. A 3 pp
  rollback threshold is coarse relative to that noise.
- **Upstream code is not yet public**, so pipeline execution cannot be audited
  from this repository alone.
