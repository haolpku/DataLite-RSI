# HLE No-Tools Docker Environment

This image reuses the existing lightweight evaluator runtime. It contains
only the evaluator, its YAML configs, and the existing Python dependencies.
HLE data is mounted at runtime and API keys are supplied through environment
variables.

## Build

Run from the DataLite-RSI repository root:

```bash
docker build -f docker/llm-hle-notools/Dockerfile \
  -t ghcr.io/haolpku/datalite-rsi-llm:0.1.0 .
```

## Real API smoke test

```bash
docker run --rm \
  -v "$DATA_DIR:/data:ro" \
  -e ZCLOUD_API_KEY \
  -e JUDGE_API_KEY \
  ghcr.io/haolpku/datalite-rsi-llm:0.1.0 \
  --config /app/configs/lite_claude_sonnet5.yaml \
  --data-dir /data \
  --max-samples 1 \
  --max-workers 1 \
  --checkpoint-interval 0 \
  --output-dir /tmp/hle-output
```

Use the Qwen config and `LOCAL_API_KEY` when the solver is a local
OpenAI-compatible service. Do not put credentials in an image, config file,
or command committed to Git.

## Publish

After the real API smoke test succeeds, push the versioned image and record
the resulting digest in any result submission:

```bash
docker push ghcr.io/haolpku/datalite-rsi-llm:0.1.0
docker inspect --format='{{index .RepoDigests 0}}' \
  ghcr.io/haolpku/datalite-rsi-llm:0.1.0
```
