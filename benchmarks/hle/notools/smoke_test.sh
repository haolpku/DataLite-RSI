#!/usr/bin/env bash
set -euo pipefail

: "${DATA_DIR:?Set DATA_DIR to a mounted HLE Parquet directory}"

CONFIG="${CONFIG:-benchmarks/hle/notools/configs/lite_claude_sonnet5.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hle-notools-smoke}"

python benchmarks/hle/notools/evaluator.py \
  --config "$CONFIG" \
  --data-dir "$DATA_DIR" \
  --max-samples 1 \
  --max-workers 1 \
  --checkpoint-interval 0 \
  --output-dir "$OUTPUT_DIR"
