# DataLite-RSI evaluation contract

DataLite-RSI evaluators should expose a deterministic, scriptable entry point. A
track-specific implementation may add fields, but the evaluation record should
always identify:

- benchmark ID and version;
- dataset revision;
- model and method IDs;
- baseline and post-RSI metrics;
- iteration and compute budgets;
- seeds and generation settings;
- code commit and container digest;
- output artifact locations.

Evaluators should separate model inference from metric computation whenever
practical. This allows independent verification from saved predictions without
requiring access to the original model or API. The first adapter is
[`benchmarks/agents-last-exam/evaluator.py`](../benchmarks/agents-last-exam/evaluator.py):
it consumes the official ALE `RunWriter` artifacts and produces deterministic
JSON metrics without importing the agent runner or contacting a model/API.

Each benchmark may expose a track-specific adapter while following the same
pattern: accept an explicit predictions/artifacts path, preserve per-run
records, and emit aggregate metrics plus invalid-input diagnostics. Baseline
and post-RSI aggregates belong in the result manifest's `metrics.baseline` and
`metrics.final` fields.
