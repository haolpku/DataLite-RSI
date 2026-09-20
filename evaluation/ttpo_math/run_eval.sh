#!/usr/bin/env bash

set -euo pipefail

: "${OPSD_MODEL_PATH:?Set OPSD_MODEL_PATH to the base model path or ID}"
: "${OPSD_EVAL_SEEDS:?Set OPSD_EVAL_SEEDS to space-separated numeric seeds}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${OPSD_EVAL_OUTPUT_DIR:-eval_results/opsd-math-competition-suite}"
tensor_parallel_size="${OPSD_TENSOR_PARALLEL_SIZE:-1}"
datasets="${OPSD_EVAL_DATASETS:-aime25 hmmt25 aime26 hmmt26 brumo25}"

checkpoint_args=()
if [[ -n "${OPSD_CHECKPOINT_DIR:-}" ]]; then
  checkpoint_args=(--checkpoint_dir "$OPSD_CHECKPOINT_DIR")
fi

mkdir -p "$output_dir"

for seed in $OPSD_EVAL_SEEDS; do
  for dataset in $datasets; do
    python "$script_dir/evaluate_math.py" \
      --base_model "$OPSD_MODEL_PATH" \
      "${checkpoint_args[@]}" \
      --dataset "$dataset" \
      --enable_thinking \
      --temperature 1.0 \
      --top_p 0.95 \
      --top_k -1 \
      --val_n 12 \
      --seed "$seed" \
      --max_new_tokens 38912 \
      --max_model_len 40960 \
      --tensor_parallel_size "$tensor_parallel_size" \
      --output_file "$output_dir/${dataset}-seed${seed}.json"
  done
done
