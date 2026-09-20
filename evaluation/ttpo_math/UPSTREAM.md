# Vendored TTPO math evaluator

Copied from <https://github.com/ZJU-REAL/TTPO> (`TTPO/eval/evaluate_math.py`) at
commit `36326ed5b04517e272a90244188b83a8bc17150c`, plus local patches used by
OPSD-data-lite. This is the harness behind
[`opsd-math-competition-suite`](../../benchmarks/opsd-math-competition-suite/).

## License status

The pinned upstream commit does not contain a standalone `LICENSE` or `COPYING`
file. This vendored integration therefore records its license as `NOASSERTION`;
redistribution terms must be confirmed before a non-demo release.

## Local patches

None change `grade_answer` (`math_verify`):

- scoped to the five OPSD-data-lite eval sets: `aime25`, `hmmt25`, `aime26`,
  `hmmt26`, `brumo25`
- `--dataset_path` / `load_local_eval_dataset` for local JSONL or Parquet
- `max_num_seqs`, `seed`, `stop_token_ids` forwarded to vLLM
- LoRA checkpoint tokenizer pickup when `tokenizer.json` is present
- richer JSON summary fields; `FORCE_EMPTY_THINK=1` escape hatch

## Verifying this copy

```bash
git clone https://github.com/ZJU-REAL/TTPO /tmp/ttpo
git -C /tmp/ttpo checkout 36326ed5b04517e272a90244188b83a8bc17150c
diff -u /tmp/ttpo/TTPO/eval/evaluate_math.py evaluation/ttpo_math/evaluate_math.py
```

## Requirements

`vllm`, `transformers`, `datasets`, `math_verify`, `tqdm`, and `pyarrow` for
local MathArena parquet.
