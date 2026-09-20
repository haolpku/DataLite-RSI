# TTPO Math Evaluation

This vendored evaluator is used by `opsd-math-competition-suite`. The
OPSD-data-lite demo reports Qwen3 thinking-mode results with temperature 1.0,
top-p 0.95, top-k disabled, and 12 samples per problem.

## One dataset

```bash
python evaluation/ttpo_math/evaluate_math.py \
  --base_model "$OPSD_MODEL_PATH" \
  --dataset aime26 \
  --enable_thinking \
  --temperature 1.0 \
  --top_p 0.95 \
  --val_n 12 \
  --output_file /tmp/aime26.json
```

Optional: `--dataset_path` for a local JSONL/Parquet file; `--checkpoint_dir`
for a LoRA adapter. See [`UPSTREAM.md`](UPSTREAM.md).

## Full suite

The batch launcher evaluates all five datasets once for every explicitly
provided seed. Seed IDs are required rather than silently invented by the
script.

```bash
export OPSD_MODEL_PATH=/path/to/Qwen3-8B
export OPSD_CHECKPOINT_DIR=/path/to/checkpoint-50  # omit for base
export OPSD_EVAL_SEEDS="0 1 2"
export OPSD_TENSOR_PARALLEL_SIZE=4

bash evaluation/ttpo_math/run_eval.sh
```

`OPSD_EVAL_DATASETS` can override the default five-dataset list, and
`OPSD_EVAL_OUTPUT_DIR` controls the output directory. The non-thinking launcher
is separate because it uses `--no_thinking` and its corresponding sampling
defaults; it is not the protocol behind the OPSD-data-lite demo table.

Dependencies: `vllm`, `transformers`, `datasets`, `math_verify`, `tqdm`, `pyarrow`.
