# Math SFT transfer suite

Measures how much a small SFT dataset improves a base model on mathematical
reasoning. The unit under test is **the dataset, not the model**: base model,
training recipe, and generation settings are all fixed, so the only variable is
which rows go into the SFT file. The score attributable to the method is the
delta over the untrained base model under identical settings.

## Benchmarks

Eight held-out benchmark test sets under the Qwen2.5-Math chain-of-thought protocol:

| Benchmark | Questions | Metric | Temperature | Samples |
| --- | ---: | --- | ---: | ---: |
| `gsm8k` | 1,319 | accuracy | 0.0 | 1 |
| `math` | 5,000 | accuracy | 0.0 | 1 |
| `minerva_math` | 272 | accuracy | 0.0 | 1 |
| `gaokao2024_mix` | 91 | accuracy | 0.0 | 1 |
| `olympiadbench` | 675 | accuracy | 0.0 | 1 |
| `amc23` | 40 | avg@4 | 0.6 | 4 |
| `aime24` | 30 | avg@4 | 0.6 | 4 |
| `aime25` | 30 | avg@4 | 0.6 | 4 |

`primary_score` is the unweighted mean across all eight. Every benchmark counts
once regardless of question count, so the 30-question AIME sets weigh as much as
the 5,000-question MATH set. This measures breadth of transfer, not aggregate
per-question accuracy, but it also makes `primary_score` noisy on the small
competition sets. Read the per-benchmark columns.

## Reference protocol v0.1.0

### SFT

| Setting | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-7B` (base, no `-Base` suffix on HF) |
| Trainer | LLaMA-Factory 0.9.3, `llamafactory-cli train`, `FORCE_TORCHRUN=1` |
| Finetuning | full-parameter, stage `sft` |
| Distribution | DeepSpeed ZeRO-2, 7 GPUs |
| Epochs | 1.0 |
| Cutoff length | 16,384 tokens |
| Batch | `per_device_train_batch_size` 1 × `gradient_accumulation_steps` 4 |
| Optimiser | `adamw_torch_fused`, lr 5.0e-6, cosine, `warmup_ratio` 0.1 |
| Precision | BF16 + TF32, attention `sdpa` |
| Seed | 42 (`seed` and `data_seed`) |
| Template | `qwen2_5_base`; assistant EOS is `<|endoftext|>` (151643) |
| Schema | `instruction` → prompt, `output` → response |

Qwen2.5-7B base terminates on `<|endoftext|>`, not `<|im_end|>`. Using the wrong
EOS trains the model never to stop.

### Evaluation

| Setting | Value |
| --- | --- |
| Engine | vLLM, 4 data-parallel replicas (one 7B model per GPU) |
| Context | `max_model_len` 32,768, `max_tokens_per_call` 16,384 |
| Prompt type | `qwen25-math-cot`, `apply_chat_template` false |
| Greedy group | temperature 0.0, top_p 1.0, n_sampling 1 |
| Sampled group | temperature 0.6, top_p 1.0, n_sampling 4 |

Each data-parallel rank needs its own vLLM, Triton, and TorchInductor cache
on node-local storage — shared compile caches corrupt each other.

### Grader

Grading is the Qwen2.5-Math harness committed verbatim under
[`evaluation/qwen25_math/`](../../evaluation/qwen25_math/). Read `grader.py` to
see exactly how an answer is compared. See
[`UPSTREAM.md`](../../evaluation/qwen25_math/UPSTREAM.md) for the upstream commit
and the full list of local changes (grading logic is unmodified).

## Evaluator

`evaluator.py` re-scores an already-graded predictions file — useful for checking
a published result without a GPU, sympy, or vLLM. The harness produces the file.

```bash
python benchmarks/math-sft-transfer-suite/evaluator.py \
  --predictions predictions.jsonl --output metrics.json
```

Input: one JSONL record per generated sample with `benchmark`, `question_id`,
`predicted_answer`, `correct_answer`, and optionally `finish_reason`. Answer
comparison is normalised string equality (strips `\boxed{}`, `$`, thousands
separators, trailing period). `predicted_answer` null or absent = parse failure.
`finish_reason` other than `"stop"` = runaway. Both count as incorrect and are
reported separately.

```bash
python -m unittest discover -s benchmarks/math-sft-transfer-suite/tests -v
```

## Data

All eight test sets are registered under `datasets/`:

| Dataset directory | Benchmark | Source |
| --- | --- | --- |
| [`gsm8k`](../../datasets/gsm8k/) | `gsm8k` | HF `openai/gsm8k` |
| [`hendrycks-math`](../../datasets/hendrycks-math/) | `math` | HF `EleutherAI/hendrycks_math` |
| [`aime24`](../../datasets/aime24/) | `aime24` | Qwen2.5-Math harness; identical to HF `HuggingFaceH4/aime_2024` |
| [`aime25`](../../datasets/aime25/) | `aime25` | Qwen2.5-Math harness; identical answers to HF `MathArena/aime_2025` |
| [`amc23`](../../datasets/amc23/) | `amc23` | Qwen2.5-Math harness; identical to HF `zwhe99/amc23` |
| [`minerva-math`](../../datasets/minerva-math/) | `minerva_math` | Qwen2.5-Math harness; identical to HF `math-ai/minervamath` |
| [`olympiadbench`](../../datasets/olympiadbench/) | `olympiadbench` | Qwen2.5-Math harness; filtered subset of HF `Hothan/OlympiadBench` |
| [`gaokao2024-mix`](../../datasets/gaokao2024-mix/) | `gaokao2024_mix` | Qwen2.5-Math harness; no standalone HF mirror identified |

All six sourced from the harness use the commit vendored at
[`evaluation/qwen25_math/`](../../evaluation/qwen25_math/)
(`a45202bd16f1ec06f433442dc1152d0074773465`).

### Diagnostic subset for in-loop feedback

A method that uses an evaluation signal inside its optimisation loop should feed
it a fixed diagnostic subset, not these full test sets — full benchmark content
reaching the loop is a leakage channel. The reference subset is a 10% sample per
benchmark drawn at seed 42 and materialised before evolution begins:

| Benchmark | Full | Diagnostic |
| --- | ---: | ---: |
| `gsm8k` | 1,319 | 132 |
| `math` | 5,000 | 500 |
| `minerva_math` | 272 | 27 |
| `gaokao2024_mix` | 91 | 9 |
| `olympiadbench` | 675 | 68 |
| `amc23` | 40 | 4 |
| `aime24` | 30 | 3 |
| `aime25` | 30 | 3 |

Sampled indices are recorded in a manifest so the subset is reconstructible.
Submissions must state which scope the loop consumed and which scope the reported
scores come from, and a baseline must exist in the same scope.

## Contamination

- Decontaminate the SFT dataset against all eight test files before submission.
  Report the procedure and measured overlap rate.
- The reference procedure uses exact normalised 13-gram overlap at `max_rate` 0.01.
- N-gram filtering removes surface duplication only; paraphrases survive.
- MATH train and test overlap in contest coverage, so `math` accuracy is
  upper-biased for any model trained on MATH train.
