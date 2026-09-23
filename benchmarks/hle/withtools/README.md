# HLE With Tools Benchmark

This benchmark preserves the lightweight single-sample HLE agent runner. The
agent uses an OpenAI-compatible chat-completions endpoint and may call
`web_search`, `web_fetch`, `code_execution`, and `submit`. A separate
OpenAI-compatible judge produces the structured correctness decision.

The repository keeps the corresponding source under `src/hle_eval` for review.
Build the execution image locally from
[`docker/llm-hle-with-tools`](../../../docker/llm-hle-with-tools). This
contribution does not publish an image; the benchmark manifest leaves
`container_image` unset.

## Run one sample

Prepare an `eval.json` whose dataset path is mounted inside the container (or
use the Hugging Face provider), then run:

```bash
docker build -f docker/llm-hle-with-tools/Dockerfile \
  -t datalite-rsi-hle-with-tools:0.1.0 .
docker run --rm \
  -e MODEL_API_KEY \
  -e JUDGE_API_KEY \
  -e TAVILY_API_KEY \
  -v "$PWD/eval.json:/config/eval.json:ro" \
  -v "$PWD/data:/data:ro" \
  -v "$PWD/output:/output" \
  datalite-rsi-hle-with-tools:0.1.0 \
  --config /config/eval.json --sample-id <sample-id>
```

The image is only an execution sandbox. Secrets are supplied at runtime and
must not be committed. `docker/llm-hle-with-tools/config.example.json` shows
the expected configuration without embedding any credentials.

## Offline aggregation

The runner appends one JSON object per sample. Aggregate those records without
calling an API:

```bash
python benchmarks/hle/withtools/evaluator.py \
  --predictions /path/to/results.jsonl \
  --output /path/to/metrics.json
```

`accuracy` is computed over successful records with a binary `score`, while
`completion_rate` is successful records divided by all valid records. Failed
records remain visible in the output and are excluded from the accuracy
denominator.

## Real API smoke test

Set the required provider variables and run one known sample:

```bash
IMAGE=datalite-rsi-hle-with-tools:0.1.0 \
CONFIG=/path/to/eval.json \
SAMPLE_ID=<sample-id> \
DATA_DIR=/path/to/data \
OUTPUT_DIR=/path/to/output \
bash benchmarks/hle/withtools/smoke_api.sh
```

This is intentionally a real API smoke test, not a mock or unit-test suite.

## Batch execution

For one-container-per-sample execution, use
`benchmarks/hle/withtools/scripts/run_batches.sh`. It accepts an environment
file, stable sample-ID list, runtime config, mounted data directory, and output
directory; it records run metadata, container logs, exit codes, and resource
snapshots. Merge the resulting JSONL files with `scripts/merge_results.py`,
then run this benchmark's offline evaluator. See the Docker environment README
for the exact commands and constraints.
