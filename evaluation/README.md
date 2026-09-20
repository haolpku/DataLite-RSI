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
requiring access to the original model or API.

The initial cross-track evaluator API will be added with the first benchmark.

## Vendored harnesses

Where a benchmark's scores depend on a third-party grader, that grader is
committed here so the decisions behind a published result can be read rather
than inferred.

| Directory | Harness | Used by |
| --- | --- | --- |
| [`qwen25_math/`](qwen25_math/) | [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math) @ `a45202bd` (MIT) | [`math-sft-transfer-suite`](../benchmarks/math-sft-transfer-suite/) |
| [`ttpo_math/`](ttpo_math/) | [TTPO](https://github.com/ZJU-REAL/TTPO) @ `36326ed5` | [`opsd-math-competition-suite`](../benchmarks/opsd-math-competition-suite/) |

[`image-edit-transfer-suite`](../benchmarks/image-edit-transfer-suite/) is graded
by a hosted VLM judge, so there is no vendored harness. The suite evaluator
re-aggregates saved per-sample scores without calling the judge.

Each directory adds an `UPSTREAM.md` with the upstream URL and exact commit, a
per-file statement of what was and was not modified, the upstream license
status, and a command to diff the copy against upstream. Preserve an upstream
`LICENSE` unchanged when one is published; otherwise use `NOASSERTION` until
the redistribution terms are confirmed. Vendor only what grading and generation
need — not upstream datasets, generated parsers, caches, or outputs.
