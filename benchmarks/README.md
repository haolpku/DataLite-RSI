# DataFlow-RSI benchmark contributions

Each benchmark lives in `benchmarks/<benchmark-id>/` and owns its task
definition, evaluator, metric documentation, and deterministic tests. Data is
registered separately under `datasets/` and stored on Hugging Face.

## Minimum layout

```text
benchmarks/my-benchmark/
├── benchmark.json
├── README.md
├── evaluator.py
└── tests/
```

Copy [`../templates/benchmark.json`](../templates/benchmark.json), then make the
manifest `id` match the directory name. The evaluator should accept an explicit
predictions path and output structured metrics rather than relying on notebook
state.

Document task construction, metric definitions, aggregation, tie handling,
invalid outputs, contamination risks, and known limitations. Include tiny
synthetic fixtures so CI can exercise the evaluator without the full dataset or
a model API.

See [CONTRIBUTING.md](../CONTRIBUTING.md#1-benchmark) for DataFlow-RSI review requirements.
