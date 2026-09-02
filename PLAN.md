# HLE No-Tools Evaluation Migration Plan

## Scope

Migrate the existing `hle_lite` evaluator into the DataLite-RSI benchmark
layout while preserving its OpenAI-compatible API, YAML configuration,
concurrent execution, retry, checkpoint, and Parquet loading behavior.

## Target layout

- `benchmarks/hle/notools/`: benchmark manifest, evaluator, documentation, and
  reusable configs.
- `datasets/hle/`: HLE full-dataset registry metadata.
- `datasets/hle-verified/`: HLE-Verified registry metadata.
- `docker/llm-hle-notools/`: reused lightweight evaluation image recipe and
  runtime documentation.

## Implementation

1. Copy the current evaluator and configs, then adapt paths and CLI defaults
   only where required by the target repository.
2. Keep solver and grader as independent OpenAI-compatible endpoints and keep
   credentials runtime-only through environment variables.
3. Support both registered datasets through explicit `--data-dir` input and
   retain the current Parquet shard format.
4. Emit the existing per-sample results plus a structured aggregate metrics
   document suitable for a benchmark run.
5. Register `hle` and `hle-verified` with immutable Hugging Face revisions and
   document provenance, licensing, privacy, and limitations. Completed with
   `cais/hle@5a81a4c7271a2a2a312b9a690f0c2fde837e4c29` and
   `skylenage-ai/HLE-Verified@0bc83643672d4f68a5f89998617a639d85e7318b`.
6. Reuse the existing `python:3.11-slim` image recipe and dependencies under
   `docker/llm-hle-notools/`, then publish a versioned image to GHCR.
7. Extend the repository validator to discover nested benchmark manifests and
   derive `benchmarks/hle/notools`'s manifest ID as `hle-notools`.

## Verification

- Do not add or migrate evaluator unit tests, mock API tests, or synthetic
  fixtures.
- Run only a real API smoke test with one sample and configured credentials.
- Run the repository manifest validator to catch malformed manifests and broken
  cross-references.
- Build the Docker image and run the same one-sample smoke test in the image.

## Publication status

The evaluator does not download or commit dataset files during normal runs.
The versioned image is built locally as
`ghcr.io/haolpku/datalite-rsi-llm:0.1.0`; publishing requires permission to
create packages in the `haolpku` GHCR namespace.
