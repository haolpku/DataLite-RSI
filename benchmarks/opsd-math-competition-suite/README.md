# OPSD math competition suite

Five held-out contests under the TTPO Qwen3-thinking protocol. The score
attributable to a method is the delta over the untrained base under identical
settings.

## Benchmarks

| Benchmark | Questions | Metric |
| --- | ---: | --- |
| `aime25` | 30 | avg@n |
| `hmmt25` | 30 | avg@n |
| `aime26` | 30 | avg@n |
| `hmmt26` | 33 | avg@n |
| `brumo25` | 30 | avg@n |

`primary_score` is the unweighted mean across all five.

## Reference protocol v0.1.0

| Setting | Value |
| --- | --- |
| Harness | [`evaluation/ttpo_math/`](../../evaluation/ttpo_math/) |
| Thinking | enabled |
| Temperature | 1.0 |
| Top-p | 0.95 |
| Top-k | -1 (disabled) |
| Samples per problem | `val_n=12` |
| Maximum generated / model tokens | 38,912 / 40,960 |
| Stop token IDs | 151643, 151645 |
| Seeds | 3; headline numbers are the mean, and each submission must record the numeric IDs |

```bash
python evaluation/ttpo_math/evaluate_math.py \
  --base_model /path/to/Qwen3-8B \
  --dataset aime26 \
  --enable_thinking --temperature 1.0 --top_p 0.95 --top_k -1 \
  --val_n 12 --max_new_tokens 38912 --max_model_len 40960 \
  --seed 0 --output_file /tmp/aime26-seed0.json
```

## Offline re-score

```bash
python benchmarks/opsd-math-competition-suite/evaluator.py \
  --predictions predictions.jsonl --output metrics.json
```

Input JSONL fields: `benchmark`, `question_id`, `predicted_answer`,
`correct_answer`, optional `finish_reason`.

```bash
python -m unittest discover -s benchmarks/opsd-math-competition-suite/tests -v
```

## Data

| Dataset | Hub |
| --- | --- |
| [`aime25`](../../datasets/aime25/) | MathArena / Qwen2.5-Math harness |
| [`hmmt25`](../../datasets/hmmt25/) | `MathArena/hmmt_feb_2025` |
| [`aime26`](../../datasets/aime26/) | `MathArena/aime_2026` |
| [`hmmt26`](../../datasets/hmmt26/) | `MathArena/hmmt_feb_2026` |
| [`brumo25`](../../datasets/brumo25/) | `MathArena/brumo_2025` |
