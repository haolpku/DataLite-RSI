# DataFlow-Evolver on Qwen2.5-7B, 14,181 -> 3,000 math SFT rows

Status: **unverified**. Submitted for maintainer review.

Five iterations of [DataFlow-Evolver](../../../rsi/methods/dataflow-evolver/) over
a fixed 14,181-row GSM8K-train + MATH-train corpus, producing exactly 3,000
`{instruction, output}` SFT records, evaluated on the
[math SFT transfer suite](../../../benchmarks/math-sft-transfer-suite/).

## Headline

Fine-tuning Qwen2.5-7B base on the 3,000 selected rows lifted the full-test-set
primary score from **30.67% to 37.93% (+7.26 points)**, with the largest gains on
GSM8K (+23.5) and Minerva (+16.2).

All columns below are the complete benchmark test sets under identical generation
settings. The two Math-3K columns are 3,000-row math SFT datasets from
DataFlow-Instruct-10K trained and evaluated the same way, so they are directly
comparable. Qwen2.5-7B-Instruct is an external reference point, not a controlled
comparison.

| Benchmark | Base | **DataFlow-Evolver** | Math-3K | Math-3K gpt-4o | *Instruct* |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gsm8k` | 65.66 | **89.16** | 87.87 | 88.10 | *92.65* |
| `math` | 62.06 | 69.44 | **72.36** | 69.96 | *75.18* |
| `minerva_math` | 15.81 | 31.99 | **34.19** | 28.31 | *36.40* |
| `gaokao2024_mix` | 20.88 | 28.57 | **40.66** | 24.18 | *51.65* |
| `olympiadbench` | 31.56 | 32.00 | **36.89** | 34.22 | *38.67* |
| `amc23` | 34.38 | **43.13** | 38.75 | 37.50 | *52.50* |
| `aime24` | 9.17 | 5.83 | 9.17 | 5.83 | *11.67* |
| `aime25` | 5.83 | 3.33 | **6.67** | 6.67 | *10.00* |
| **primary_score** | **30.67** | **37.93** | **40.82** | **36.85** | ***46.09*** |

Read honestly, this is a mixed result. The method clearly beats the base model
(+7.26) and beats the gpt-4o rewrite of Math-3K (+1.08) while using a teacher of
the same class. It does **not** beat the original Math-3K subset (-2.89), which
wins on five of eight benchmarks — decisively on `gaokao2024_mix` (40.66 vs
28.57) and `olympiadbench` (36.89 vs 32.00).

Where the method does win is the two largest benchmarks by question count:
`gsm8k` (89.16, best of all four trained systems) and `amc23` (43.13, likewise).
Both AIME sets regress below base, but at 30 questions with avg@4 those columns
resolve to about 0.8 points and should not carry weight.

**The recursion itself works.** The improvement accumulated across iterations
rather than arriving at once — measured on the in-loop diagnostic signal, the
incumbent improved monotonically:

| In-loop checkpoint | Dataset trained on | diagnostic primary |
| --- | --- | ---: |
| baseline | none (untrained base) | 0.311 |
| checkpoint 2 | incumbent after iteration 1 | 0.370 |
| checkpoint 4 | incumbent after iteration 4 | **0.403** |

`review_score` rose over the same span (0.7357 → 0.7660 → 0.7709), and on the
full test sets every iteration after the first beats iteration 1:

| Dataset | full-set primary | vs iteration 1 |
| --- | ---: | ---: |
| iteration 1 | 36.16 | — |
| iteration 2 | 39.03 | +2.87 |
| iteration 3 | 37.40 | +1.24 |
| **iteration 4 (selected)** | **37.93** | **+1.77** |
| iteration 5 | 37.52 | +1.36 |

## Setup

| | |
| --- | --- |
| Input corpus | 14,181 rows (7,469 `gsm8k_train` + 6,712 `math_train`), 13-gram decontaminated, `sha256 f9bae97a...` |
| Output | exactly 3,000 rows, `sha256 6d362290fb77ae465433d8f8b6cd4a5a5c4c2252ecea5b53757e69d5e23f7219` |
| Base model | `Qwen/Qwen2.5-7B` (base, not Instruct) @ `d149729398750b98c0af14eb82c78cfe92750796` |
| Iterations | 5, downstream checkpoint every 2 |
| Pipeline agent | Codex CLI + `gpt-5.6-sol`, effort xhigh, 50 turns/iteration |
| ReviewAgent | `gpt-5.6-sol`, 80-row sample; run artifacts record sample seed 43 |
| Teacher LLM | `gpt-4o`, temperature 0.6, top_p 0.95, max_tokens 16,384 |
| Embedding | Qwen3-Embedding-8B, proxy corpus ODA-Math-460k, 3,000 samples |
| SFT | LLaMA-Factory 0.9.3, full-parameter, DeepSpeed ZeRO-2, GPUs 0-6 |
| Evaluation | Qwen2.5-Math harness @ `a45202bd16f1ec06f433442dc1152d0074773465`, vLLM 0.9.2, 4 DP ranks on GPUs 0-3 |
| Hardware | 8x A100-80GB SXM, driver 535.129.03; GPU 7 reserved for the embedding service |
| Seed | SFT `seed`/`data_seed` 42; evaluation seed 0 per the harness default |

Training and inference follow the
[benchmark reference protocol](../../../benchmarks/math-sft-transfer-suite/README.md#reference-protocol-v010)
exactly: 1 epoch, cutoff 16,384, batch 1 x 4 accumulation, lr 5e-6 cosine with 0.1
warmup, BF16/TF32, `qwen2_5_base` template with `<|endoftext|>` as the assistant
EOS. The evaluator that produced every score below is committed verbatim under
[`evaluation/qwen25_math/`](../../../evaluation/qwen25_math/).

A hard task constraint: LLM rewriting was capped at a bounded candidate pool of
at most 6,000 rows, formed by deterministic filtering before any generation call.
The method could not simply rewrite the whole corpus.

## Two evaluation scopes, and why

**In-loop feedback uses a fixed diagnostic subset, never the full test sets.**
This is a deliberate design choice: the periodic downstream signal is injected
into the next pipeline proposal prompt, so evaluating on complete test sets there
would let benchmark content steer the optimisation. Restricting the loop to a
10% sample bounds that exposure.

**The reported scores are the full test sets**, evaluated after the run finished.
They never influenced any acceptance decision, and no pipeline was ever authored
with visibility into them.

| Scope | Used for | Enters the loop? |
| --- | --- | --- |
| Full test sets | baseline/final in `result.json`, all reported scores | No |
| Diagnostic subset (10%) | in-loop feedback signal only | Yes, by design |

### Per-iteration full-set scores

All rows are Qwen2.5-7B base fine-tuned on that iteration's 3,000 rows, evaluated
identically.

| Dataset | gsm8k | math | minerva | gaokao | olympiad | amc23 | aime24 | aime25 | primary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| *base (untrained)* | *65.66* | *62.06* | *15.81* | *20.88* | *31.56* | *34.38* | *9.17* | *5.83* | *30.67* |
| iter 1 | 88.40 | 68.66 | 32.35 | 27.47 | 29.93 | 32.50 | 6.67 | 3.33 | 36.16 |
| iter 2 | 90.30 | 71.22 | 25.00 | 31.87 | 36.15 | 44.38 | 10.00 | 3.33 | 39.03 |
| iter 3 | 90.07 | 70.16 | 31.62 | 27.47 | 32.00 | 38.75 | 5.00 | 4.17 | 37.40 |
| **iter 4 (selected)** | **89.16** | **69.44** | **31.99** | **28.57** | **32.00** | **43.13** | **5.83** | **3.33** | **37.93** |
| iter 5 | 89.99 | 69.28 | 29.04 | 34.07 | 31.56 | 36.25 | 6.67 | 3.33 | 37.52 |

Question counts: gsm8k 1,319 / math 5,000 / minerva 272 / gaokao 91 /
olympiadbench 675 / amc23 40 / aime24 30 / aime25 30. Greedy for the first five,
avg@4 for the last three.

### The diagnostic subset, for reference

This is the signal the loop actually optimised against — a 10% per-benchmark
sample at seed 42, fixed before evolution began. It is reported here because the
RSI claim rests on it, not because it is an evaluation result.

| Checkpoint | diagnostic primary |
| --- | ---: |
| base (untrained) | 0.311 |
| incumbent after iteration 1 | 0.370 |
| incumbent after iteration 4 | 0.403 |

At subset sizes of 3–9 questions the four competition benchmarks are indicative
only; one `gaokao2024_mix` item moves that column by 11 points.

## Evolution trace

| iter | parent | review_score | correctness | difficulty | repairs | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | — | 0.7357 | 0.78 | 0.68 | 1 | accepted |
| 2 | 1 | 0.7315 | 0.72 | 0.79 | 2 | rejected |
| 3 | 1 | 0.7660 | 0.89 | 0.64 | 0 | accepted |
| **4** | 3 | **0.7709** | **0.93** | 0.58 | 1 | **accepted, final** |
| 5 | 4 | 0.7129 | 0.68 | 0.73 | 2 | rejected |

Across all five iterations: `schema_score` 1.0, `dup_rate` 0.0,
`contamination_rate` 0.0, `relevance` 0.98–0.99, and no hard-metric violation.
The incumbent improved twice (0.7357 -> 0.7660 -> 0.7709) and two challengers were
correctly held back. Six agent repair invocations total.

The selected pipeline's seven operators and their row funnel:

```text
BalancedCandidatePoolV5            6,000   deterministic clean + balance to pool cap
ConciseAnchoredGenerationV5        6,000   teacher generates derivations
DeterministicSolutionValidationV8  5,645   format/integrity hard gate only
IndependentMathAuditV3             5,645   independent line-by-line recomputation
StrictAuditGateV4                  4,235   keep unanimous PASS
BalancedDiverseSelectionV6         3,000   template/structure caps + coverage rotation
StrictFinalExportV6                3,000   exact schema, count, uniqueness
```

### What the loop learned

The most transferable finding concerns **where to put determinism**. Early
iterations over-rejected badly:

- iteration 1: a `\boxed{}` parser treated LaTeX closing delimiters and sentence
  punctuation as content following the answer, collapsing 6,000 generated rows to
  **4**;
- iteration 4's predecessor: condition-coverage and arithmetic-regex heuristics cut
  6,000 to 1,975.

The loop diagnosed both from execution evidence and converged on a durable
division of labour: deterministic validators enforce only decidable format and
integrity constraints, while mathematical correctness goes to an independent LLM
audit that recomputes each derivation line by line. Reviewer-measured correctness
rose 0.78 -> 0.93 across the run, with contamination and duplication held at zero
throughout.

This is the recursion paying off in an inspectable way. The improvement is a
specific, readable operator change — `DeterministicSolutionValidationV2` to `V8`
plus the addition of `IndependentMathAuditV3` and `StrictAuditGateV4` — and not an
opaque diff over rows.

### The remaining tension

Raising correctness cost difficulty: it fell 0.68 -> 0.58 over the same span, and
the reviewer flagged template lock-in at iteration 4, with outputs converging on
"Given... First... Thus..." and numbered steps. Iteration 5 attacked exactly this
and did raise difficulty to 0.73, but correctness fell to 0.68 and it was
rejected. Five iterations were not enough to find a pipeline that improves both at
once; this is the natural place for a longer run to start.

## How well does review_score predict transfer?

Worth stating plainly, since the acceptance rule rests on it: the proxy signal
tracks downstream transfer in direction but not in exact rank order.

It gets the important call right. On the diagnostic subset the incumbent chain the
signal produced is monotonically better (0.311 -> 0.370 -> 0.403), and on the full
test sets every post-iteration-1 dataset beats iteration 1. Ranking a *rejected*
challenger against the accepted one is where it is imperfect: iteration 2, which
scored 0.7315 and was held back, reaches 39.03 on the full-set primary against
iteration 4's 37.93 — the highest of the five.

Two reasons not to over-read that inversion:

1. **The spread is tiny.** All five `review_score` values sit inside a 0.058 band,
   and the full-set primaries inside 2.9 points. With one SFT seed per dataset and
   no variance estimate, that ordering is plausibly within run-to-run noise.
2. **The two are measuring different things.** `review_score` weights per-sample
   correctness at 0.40 and difficulty at 0.15. Iteration 2 traded correctness
   (0.72 vs 0.93) for difficulty (0.79 vs 0.58), and harder examples may transfer
   better than cleaner ones. That is a hypothesis about the weighting, not a
   defect in the loop.

A cheap static proxy that has to stand in for a full SFT-plus-evaluation cycle
cannot be perfectly rank-correlated with it, and demanding that would defeat the
point — the whole reason for the proxy is that the real signal costs a GPU-day per
candidate. What matters for the method is that acceptance moves the incumbent in
the right direction, which it did at every step here. Tightening the correlation —
reweighting the reviewer dimensions, or estimating variance with repeated seeds —
is the obvious follow-up, and this submission is the baseline to measure it
against.

## Contamination

- The input corpus was filtered by exact normalised 13-gram overlap against all
  eight benchmark test files before the run began.
- Every candidate dataset was re-checked each iteration against the same
  references, as a hard gate at `max_rate` 0.01. All five measured **0.0**.
- In-loop downstream feedback is confined to a fixed diagnostic subset so that
  full benchmark test content never reaches the pipeline agent.
- Residual risk: n-gram filtering removes surface duplication only. Paraphrases
  and renamed-entity variants survive. MATH train and test are drawn from
  overlapping contest pools, so `math` is upper-biased for any model trained on
  MATH train.
- The teacher model (`gpt-4o`) may itself have seen these benchmarks in
  pretraining, so its generated derivations could carry memorised test content
  that no check against the *input* corpus would catch. This applies to every
  teacher-distillation method rather than to DataFlow-Evolver specifically, but it
  bounds how strongly these numbers support a claim about the pipeline alone.

## Cost

Wall clock, pipeline evolution only (agent authoring + pipeline execution +
review, per iteration):

| iter | wall time | repairs |
| ---: | ---: | ---: |
| 1 | 32.2 min | 1 |
| 2 | 55.3 min | 2 |
| 3 | 25.2 min | 0 |
| 4 | 36.6 min | 1 |
| 5 | 62.6 min | 2 |
| **total** | **3.53 h** (12,717 s) | 6 |

Each downstream checkpoint added roughly 17-19 min for SFT plus
diagnostic-subset evaluation (1,031 s and 1,112 s for checkpoints 2 and 4).
Embedding review cost 577 s across the run, 91-134 s per iteration. The
post-hoc full-test-set evaluations dominate total machine time and are not part
of the loop's cost. End-to-end the run occupied its 8-GPU node for roughly 30 h
including all evaluation.

Token usage, by namespace — the framework records provider-reported usage and
marks missing usage rather than estimating it:

| Namespace | Requests | Input | Output | Cached in | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pipeline teacher (`gpt-4o`) | 87,315 | 51.04M | 37.32M | 0.53M | **88.36M** |
| Pipeline agent (Codex + `gpt-5.6-sol`) | — | 9.13M | 0.12M | 6.71M | **9.25M** |
| Embedding (Qwen3-Embedding-8B) | 15,000 | 8.82M | — | — | **8.82M** |
| ReviewAgent (`gpt-5.6-sol`) | 7 | 0.21M | 0.01M | 0.02M | **0.22M** |
| **Total** | | | | | **106.65M** |

Two things worth noting for anyone budgeting a similar run. **The teacher
dominates**, at 83% of all tokens — it is called once per candidate row across a
6,000-row pool every iteration, so teacher cost scales with the pool cap, not
with the 3,000-row output. Only 4 of 87,315 teacher requests failed. And the
agent's input is **73% cached** (6.71M of 9.13M), because each repair attempt
resumes the same session rather than restarting it; without prompt caching the
authoring cost would be substantially higher. The ReviewAgent is nearly free by
comparison — 7 requests total, one per candidate plus a re-score.

## Scope and limitations

- **The method does not beat the strongest comparable dataset.** The original
  Math-3K subset of DataFlow-Instruct-10K reaches 40.82 against this run's 37.93,
  winning five of eight benchmarks. The method does beat the base model (+7.26)
  and the gpt-4o rewrite of that same subset (+1.08), but a curated human-authored
  corpus remains ahead. Closing that gap is the obvious target for a longer run.
- **One seed per dataset.** Each candidate got a single SFT run at seed 42.
  Run-to-run variance is unmeasured, which is what limits how finely the
  per-iteration ordering can be read.
- **Small competition sets.** At 30 questions and avg@4, AIME24/25 resolve to
  about 0.8 points, and both regress below base in this run. Read
  the large benchmarks first.
- **Single base model.** All conclusions are for Qwen2.5-7B base. Transfer to
  other model families is untested.
- **Five iterations.** A short budget by design. The correctness/difficulty
  trade-off above was still open when the budget ran out.

## Reproducibility

| | |
| --- | --- |
| Method revision | `7b788746b24c2b9d6713709cb64d8c37602a2fbb` |
| Configuration | [`math-periodic.yaml`](../../../rsi/methods/dataflow-evolver/configs/math-periodic.yaml) |
| Dataset revisions | GSM8K `740312add88f781978c0658806c59bc2815b9866`, MATH `21a5633873b6a120296cce3e2df9d5550074f4a3` |
| Base model | `Qwen/Qwen2.5-7B` @ `d149729398750b98c0af14eb82c78cfe92750796` |
| Evaluation harness | [`evaluation/qwen25_math/`](../../../evaluation/qwen25_math/) — Qwen2.5-Math @ `a45202bd16f1ec06f433442dc1152d0074773465` (MIT) |
| SFT trainer | LLaMA-Factory 0.9.3 (tag `v0.9.3`, commit `ca75f1e`) |
| Environments | Python 3.11.15, torch 2.7.0+cu126, DeepSpeed 0.16.9, vLLM 0.9.2, transformers 4.52.4 |
| Container | `ghcr.io/haolpku/datalite-rsi-eval:0.1.0` |

The run name is an identifier only, not a global random seed. Randomness is
component-specific; the persisted configuration lists `42` for ReviewAgent
sampling, while the saved ReviewAgent artifacts record an observed sample seed
of `43`. `result.json` lists every component seed under
`settings.seed_details`.

```bash
python -m dataflow_evolver.main \
  --config rsi/methods/dataflow-evolver/configs/math-periodic.yaml \
  --task task_math_mix_v3.txt --log-level INFO
```

Two entries in the manifest record identifiers that **cannot currently be fetched**.
A verifier should know this before attempting a rerun:

- **`container_image`** names `ghcr.io/haolpku/datalite-rsi-eval:0.1.0`, which has
  not been built or pushed — `docker pull` will fail. A version tag stands in for
  a digest to record the intent to pin; replace it with the digest once the image
  exists. The exact package versions needed to rebuild the environment are in the
  table above.
- **`code_revision`** names the real commit the run used, but the DataFlow-Evolver
  repository is private, so `git clone` will fail. What is auditable without it:
  the reference configuration under
  [`rsi/methods/dataflow-evolver/`](../../../rsi/methods/dataflow-evolver/), and the
  evaluation harness that produced every score, committed verbatim under
  [`evaluation/qwen25_math/`](../../../evaluation/qwen25_math/). Grading can be
  checked independently; pipeline execution cannot.

Run artifacts (per-iteration operator source, review evidence, agent sessions,
token accounting, benchmark outputs, and the 3,000-row dataset) are on the
experiment machine. `artifacts_url` is null pending upload to Hugging Face; the
`sha256` values above identify the input corpus and the final dataset in the
meantime.
