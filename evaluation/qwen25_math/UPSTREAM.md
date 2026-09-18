# Vendored Qwen2.5-Math evaluator

Copied from <https://github.com/QwenLM/Qwen2.5-Math> at commit
`a45202bd16f1ec06f433442dc1152d0074773465`, MIT licensed (`LICENSE`). This is the
exact code that produced the scores in
[`results/submissions/dataflow-evolver-qwen25-7b-math-3k`](../../results/submissions/dataflow-evolver-qwen25-7b-math-3k/).

## Why it is here

Answer grading decides every score in the
[math SFT transfer suite](../../benchmarks/math-sft-transfer-suite/), so the code
that made those decisions is committed rather than described. Reading `grader.py`
answers "would this answer have been marked correct?" definitively.

## Grading is unmodified

| File | Status |
| --- | --- |
| `grader.py` | byte-identical to upstream |
| `utils.py`, `python_executor.py`, `data_loader.py`, `model_utils.py`, `trajectory.py`, `examples.py` | byte-identical to upstream |
| `evaluate.py` | trailing whitespace only |
| `parser.py` | one line added: `"aime25"` |
| `math_eval.py` | inference orchestration only, see below |

`parser.py` adds `aime25` to the list of benchmarks whose ground-truth answer
comes from the `answer` field, alongside the existing `aime24`. No parsing or
comparison logic changed.

## Changes to math_eval.py

None affect grading; all concern how generation is driven.

- `--data_parallel_size`, `--data_parallel_rank`, `--tensor_parallel_size`:
  contiguous sharding so ranks can run concurrently and be concatenated in order.
- `--max_model_len`, and `--seed` forwarded to `vllm.LLM` (upstream accepts
  `--seed` but does not pass it to the engine, so runs were not reproducible).
- **Stop tokens resolved from the tokenizer.** Upstream hard-codes
  `[151645, 151643]` whenever the model path contains `qwen2`. A base model
  terminates on `<|endoftext|>` alone, so the vendored copy reads the IDs from the
  tokenizer instead. This is a correctness fix: with the wrong stop set, base-model
  generations do not terminate.
- Records per-sample `finish_reason` and output token counts, which is what lets a
  length-truncated generation be distinguished from a wrong answer.
- Returns an empty metrics record for an empty shard instead of raising.
- Rejects `--prompt_type qwen25-math-cot` combined with `--apply_chat_template`,
  because that prompt already contains complete ChatML markers and applying a
  template double-wraps it.

## Verifying this copy

```bash
git clone https://github.com/QwenLM/Qwen2.5-Math /tmp/q25m
git -C /tmp/q25m checkout a45202bd16f1ec06f433442dc1152d0074773465
diff -r /tmp/q25m/evaluation evaluation/qwen25_math
```

## Not included

- `latex2sympy/`: the upstream tree bundles a generated ANTLR parser. Install the
  published `latex2sympy2` package instead; `grader.py` imports it by name.
- Upstream datasets, shell launchers, caches, and generated outputs.

## Requirements

`sympy`, `latex2sympy2`, `regex`, `word2number`, `numpy`, `tqdm`, `pebble`, and
for generation `vllm` plus `transformers`. The reference run used vLLM 0.9.2 and
transformers 4.52.4 under Python 3.11.15.
