#!/usr/bin/env bash

set -euo pipefail

: "${OPSD_MODEL_PATH:?Set OPSD_MODEL_PATH to the base model path or ID}"
: "${OPSD_EVAL_SEEDS:?Set OPSD_EVAL_SEEDS to space-separated numeric seeds}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${OPSD_EVAL_OUTPUT_DIR:-eval_results/opsd-math-competition-suite-nonthinking}"
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
      --no_thinking \
      --temperature 0.7 \
      --top_p 0.8 \
      --top_k 20 \
      --val_n 12 \
      --seed "$seed" \
      --max_new_tokens 32768 \
      --max_model_len 32768 \
      --tensor_parallel_size "$tensor_parallel_size" \
      --output_file "$output_dir/${dataset}-seed${seed}.json"
  done
done
