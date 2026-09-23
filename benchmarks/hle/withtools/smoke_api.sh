#!/usr/bin/env bash
set -euo pipefail

: "${CONFIG:?Set CONFIG to an eval.json file}"
: "${SAMPLE_ID:?Set SAMPLE_ID to one HLE sample ID}"
: "${DATA_DIR:?Set DATA_DIR to the mounted dataset directory}"

IMAGE="${IMAGE:-datalite-rsi-hle-with-tools:0.1.0}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/output/hle-withtools-smoke}"
mkdir -p "$OUTPUT_DIR"

if [[ "$IMAGE" == ghcr.io/* ]]; then
  docker pull "$IMAGE"
fi
docker run --rm \
  -e MODEL_API_KEY \
  -e JUDGE_API_KEY \
  -e TAVILY_API_KEY \
  -e EXA_API_KEY \
  -e GOOGLE_CSE_API_KEY \
  -e GOOGLE_CSE_ID \
  -e HF_TOKEN \
  -v "$CONFIG:/config/eval.json:ro" \
  -v "$DATA_DIR:/data:ro" \
  -v "$OUTPUT_DIR:/output" \
  "$IMAGE" \
  --config /config/eval.json \
  --sample-id "$SAMPLE_ID"
