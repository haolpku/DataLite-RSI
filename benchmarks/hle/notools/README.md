# HLE No-Tools Benchmark

This benchmark runs the lightweight HLE evaluator against a chat model without
tool use. The default registered dataset is `hle-verified`; the full HLE
dataset is registered separately as `hle` and can be evaluated with the same
runner by passing its mounted Parquet directory.

The evaluator keeps the existing solver/grader OpenAI-compatible API contract.
Solver and grader may use different endpoints and credentials. API keys are
read from the environment variables named in the YAML configuration and are
never written to the repository or image.

The current runner sends text questions only. It can load the full HLE Parquet
file, but image-containing HLE items require a multimodal runner before they
can be evaluated faithfully.

## Run

```bash
python benchmarks/hle/notools/evaluator.py \
  --config benchmarks/hle/notools/configs/lite_claude_sonnet5.yaml \
  --data-dir /path/to/hle/parquet \
  --max-samples 1 \
  --output-dir outputs
```

The directory may contain `Revision_subset.part*.parquet` files or other
Parquet files with `id`, `question`, and `answer` fields. Each run writes a
stable `checkpoint.json`, per-sample `results.json`, and aggregate
`metrics.json`; pass `--checkpoint-file` to resume a run explicitly.

The official metric is accuracy over samples with a binary grader decision.
API failures and malformed responses are retained in the per-sample artifact
and excluded from the accuracy denominator. The grader is an external model,
so its model revision, endpoint, prompt, and runtime settings must be recorded
when submitting a result.

This contribution intentionally has no offline evaluator tests. Verification
is performed with a real API smoke run using one sample and configured keys.
