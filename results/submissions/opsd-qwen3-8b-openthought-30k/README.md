# OPSD on Qwen3-8B, OpenThought-30k with LSC-based data selection

Status: **unverified**. Submitted for maintainer review.

Six iterations of [OPSD](../../../rsi/methods/opsd/) over the OpenThought-30k corpus, 
using LSC (Late Support Collapse) as an unlabeled rollout-quality proxy to
guide data selection. Evaluated on the [math SFT transfer suite](../../../benchmarks/math-sft-transfer-suite/)
with an additional AIME 2026 benchmark.

## Headline

On the supplied 30-problem AIME26 evaluation, the base model scored **67.78%**
(244/360) and iteration 6 scored **71.67%** (258/360), a **+3.89 percentage-point**
change. These are not full-suite primary scores.

All columns below are the complete benchmark test sets under identical generation
settings.

| Benchmark | Base (iter 0) | iter 1 | iter 2 | iter 3 | iter 4 | iter 5 | **iter 6** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gsm8k` | TODO | TODO | TODO | TODO | TODO | TODO | **TODO** |
| `math` | TODO | TODO | TODO | TODO | TODO | TODO | **TODO** |
| `minerva_math` | TODO | TODO | TODO | TODO | TODO | TODO | **TODO** |
| `gaokao2024_mix` | TODO | TODO | TODO | TODO | TODO | TODO | **TODO** |
| `olympiadbench` | TODO | TODO | TODO | TODO | TODO | TODO | **TODO** |
| `amc23` | TODO | TODO | TODO | TODO | TODO | TODO | **TODO** |
| `aime24` | TODO | TODO | TODO | TODO | TODO | TODO | **TODO** |
| `aime25` | TODO | TODO | TODO | TODO | TODO | TODO | **TODO** |
| `aime26` | TODO | TODO | TODO | TODO | TODO | TODO | **TODO** |
| **primary_score** | **TODO** | **TODO** | **TODO** | **TODO** | **TODO** | **TODO** | **TODO** |

The reported aggregate is 30 AIME26 problems, 4 samples per seed, averaged over 3 seeds (360 judgments). The candidate pool is the first 6400 OpenThought-30k rows; every iteration trains for 100 steps and selects exactly 3200 rows using an LSC threshold computed from the previous iteration evaluation.

## LSC Metric and Threshold Calibration

LSC is `J_80 - J_100`, where each `J` is the mean per-token Jaccard overlap between the student/base top-20 token IDs and the privileged top-20 IDs for the same forced-scored continuation. Privileged prompts contain the first 80% or all of one fixed text-distinct reference reasoning prefix plus its final answer. The protocol uses an 8192-token model limit, reserves 1024 response tokens, crops reasoning only on overflow, and records truncation. After each evaluation, its LSC scores determine the threshold used on the next fixed 6400-row candidate pool; exactly 3200 rows are retained. The supplied correctness totals document evaluation outcomes and are not used to retrofit thresholds.

The recursion is defined as six iterations. The supplied scores are not monotonic: iteration 2 dips below iteration 1, and rollback runs are recorded separately.

| Iteration | LSC Cut (ascending) | Validation Primary | Train Data Selected |
| --- | ---: | ---: | ---: |
| 0 (base) | — | 67.78% (244/360) | 3200 for first update |
| 1 | previous-eval LSC cut | 68.33% (246/360) | 3200 |
| 2 | previous-eval LSC cut | 68.06% (245/360) | 3200 |
| 3 | previous-eval LSC cut | 70.28% (253/360) | 3200 |
| 4 (rollback) | previous-eval LSC cut | 69.72% (251/360) | 3200 |
| 4 | previous-eval LSC cut | 70.00% (252/360) | 3200 |
| 5 (rollback) | previous-eval LSC cut | 67.50% (243/360) | 3200 |
| 5 | previous-eval LSC cut | 71.94% (259/360) | 3200 |
| 6 | previous-eval LSC cut | 71.67% (258/360) | 3200 |

## Bad Case Analysis

Document patterns observed in bad cases across iterations:
- Which problem types consistently have low LSC scores?
- Which problem types show overconfidence (high LSC, low accuracy)?
- How does the LSC distribution evolve?
- What insights from bad case analysis guided data selection?

## Setup

| | |
| --- | --- |
| Input corpus | OpenThought-30k, TODO: add hash and decontamination details |
| Output | TODO: rows selected per iteration, add hash |
| Base model | `Qwen/Qwen3-8B` (base) @ TODO: add commit hash |
| Iterations | 6 |
| Selection method | Previous-evaluation LSC threshold, exactly 3200 of first 6400 rows |
| Validation split | Previous iteration AIME26 evaluation (30 problems, 3 seeds) |
| SFT | TODO: training framework and hyperparameters |
| Evaluation | Qwen2.5-Math harness @ `a45202bd16f1ec06f433442dc1152d0074773465` |
| Hardware | TODO: GPU configuration |
| Seed | TODO: training and evaluation seeds |

Training runs use 100 optimizer steps per iteration. Training and inference follow the
[benchmark reference protocol](../../../benchmarks/math-sft-transfer-suite/README.md#reference-protocol-v010).
TODO: Add specific training hyperparameters (epochs, batch size, learning rate, etc.)

## Two evaluation scopes, and why

**In-loop feedback uses the previous iteration's AIME26 evaluation.**
The LSC selection cut for each iteration is computed from that previous evaluation;
no separate 10% diagnostic subset was supplied for this run.

**The supplied reported scores are the 30-problem AIME26 evaluation**, evaluated
after each iteration. Each previous evaluation fed the next threshold calculation.

| Scope | Used for | Enters the loop? |
| --- | --- | --- |
| Full benchmark suite | Not supplied in this result | No |
| AIME26 30-problem evaluation | LSC calibration and iteration feedback | Yes, by design |

### Per-iteration full-set scores

All rows are Qwen3-8B base fine-tuned on that iteration's selected data, evaluated
identically on complete test sets.

| Dataset | gsm8k | math | minerva | gaokao | olympiad | amc23 | aime24 | aime25 | aime26 | primary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| *base (untrained)* | — | — | — | — | — | — | — | — | 67.78% | — |
| iter 1 | — | — | — | — | — | — | — | — | 68.33% | — |
| iter 2 | — | — | — | — | — | — | — | — | 68.06% | — |
| iter 3 | — | — | — | — | — | — | — | — | 70.28% | — |
| iter 4 | — | — | — | — | — | — | — | — | 70.00% | — |
| iter 5 | — | — | — | — | — | — | — | — | 71.94% | — |
| **iter 6 (final)** | — | — | — | — | — | — | — | — | 71.67% | — |

### AIME26 per-problem counts

The detailed counts below are correct judgments out of 12 per problem (four
samples for each of three seeds). They are retained so aggregate percentages
can be audited.

```text
run             P01 P02 P03 P04 P05 P06 P07 P08 P09 P10 P11 P12 P13 P14 P15 P16 P17 P18 P19 P20 P21 P22 P23 P24 P25 P26 P27 P28 P29 P30 | total / 360
base             12  12  11  12  12  12  12  12   1   5   8  12  11   0   0  12   5   3  12  12  12  12  11  10   7   7   7   2   0   0 | 244 / 360 (67.78%)
opsd raw         12  12  12  12  12  12  11  12   4   3  11  11  10   0   0  12   3   7  11  12  12  12  10  11   8   8   2   6   0   0 | 248 / 360 (68.89%)
iteration 1      12  12  11  12  12  12  12  12   1   5   8  12   9   0   0  11   3   8  12  11  12  12  11  12   6   8   3   6   0   1 | 246 / 360 (68.33%)
iteration 2      12  11  12  11  12  12  11  12   4   4  10  12   7   1   0  12   3   7  12  12  12  12   8  12   5   9   2   8   0   0 | 245 / 360 (68.06%)
iteration 3      12  12  11  12  12  12  12  12   3   5   9  12   9   0   0  12   5   6  12  11  12  11  11  11   8   9   5   6   1   0 | 253 / 360 (70.28%)
iteration 4 rb   12  12  12  12  12  12  12  11   2   3   8  12   6   1   0  12   5   7  12  11  12  12  10  12   9  10   7   4   1   0 | 251 / 360 (69.72%)
iteration 4      12  12  11  12  12  12  12  12   1   6   8  12   8   0   0  12   6   5  12  11  12  12  11  12   9   9   3   7   1   0 | 252 / 360 (70.00%)
iteration 5 rb   12  12  11  12  12  12  12  12   1   6   9  12   7   0   0  12   4   3  12  12  12  12  11  12   8   5   6   4   0   0 | 243 / 360 (67.50%)
iteration 5      12  12  11  12  12  12  12  12   3   7  11  12  11   0   0  12   1   9  12  12  12  12  12  11   8  10   3   5   1   0 | 259 / 360 (71.94%)
iteration 6      12  12  11  11  12  12  12  12   4   5   9  12  10   1   0  12   6   9  12  12  12  12  11  11   9  11   2   4   0   0 | 258 / 360 (71.67%)
```

Available question count for this report: AIME26 30. Each problem has four
samples per seed and results are averaged over three evaluation seeds.

### The diagnostic subset, for reference

This is the signal used for LSC calibration: the previous iteration's AIME26
evaluation (30 problems, 3 seeds). No separate 10% diagnostic subset was supplied.

| Iteration | Diagnostic Primary | LSC Cut |
| --- | ---: | ---: |
| base (untrained) | 67.78% (244/360) | used for iter 1 cut |
| iter 1 | 68.33% (246/360) | used for iter 2 cut |
| iter 2 | 68.06% (245/360) | used for iter 3 cut |
| iter 3 | 70.28% (253/360) | used for iter 4 cut |
| iter 4 | 70.00% (252/360) | used for iter 5 cut |
| iter 5 | 71.94% (259/360) | used for iter 6 cut |
| iter 6 (final) | 71.67% (258/360) | end of supplied run |

## Evolution trace

Record the evolution of the ascending LSC cut and selected row count across iterations.

| iter | parent | LSC cut | validation primary | data selected | decision notes |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | base | previous-eval cut | 68.33% | 3200 | 100 steps |
| 2 | 1 | previous-eval cut | 68.06% | 3200 | 100 steps |
| 3 | 2 | previous-eval cut | 70.28% | 3200 | 100 steps |
| 4 | 3 | previous-eval cut | 70.00% | 3200 | 100 steps |
| 5 | 4 | previous-eval cut | 71.94% | 3200 | 100 steps |
| 6 | 5 | previous-eval cut | 71.67% | 3200 | 100 steps |

## Key Findings

Key findings to report after verification:
- Does LSC-based selection lead to monotonic improvement?
- Which datasets benefit most from the method?
- How does AIME 2026 (new) compare to AIME 2024/2025?
- What patterns emerged in bad case analysis?
- How does performance compare to random selection or other baselines?

## Reproducibility

TODO: Add complete reproducibility details:
- Environment setup (Python version, library versions, CUDA version)
- LSC computation implementation details
- Threshold calibration algorithm
- Data selection procedure
- Random seeds for all stochastic steps
- Complete training command lines
- Evaluation scripts

## Artifacts

- Training data: TODO: path/hash
- Model checkpoints: TODO: paths
- Evaluation predictions: TODO: paths
- LSC scores: `artifacts/first3200_multilevel_l20_100_1024/question_features.jsonl` when generated
- Bad case analysis: TODO: paths

## References

TODO: Add references to:
- OpenThought-30k dataset
- Qwen3-8B technical report
- Related work on confidence-based data selection
- Self-teacher methodology
