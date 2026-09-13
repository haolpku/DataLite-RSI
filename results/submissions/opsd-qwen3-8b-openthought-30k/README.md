# OPSD on Qwen3-8B, OpenThought-30k with LCS-based data selection

Status: **unverified**. Submitted for maintainer review.

Six iterations of [OPSD](../../../rsi/methods/opsd/) over the OpenThought-30k corpus, 
using LCS (TODO: expand acronym) as a proxy metric for self-teacher confidence to 
guide data selection. Evaluated on the [math SFT transfer suite](../../../benchmarks/math-sft-transfer-suite/)
with an additional AIME 2026 benchmark.

## Headline

Fine-tuning Qwen3-8B base on LCS-selected data from OpenThought-30k lifted the 
full-test-set primary score from **TODO% to TODO% (+TODO points)**.

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

TODO: Add comparison analysis with baseline and other methods.

## LCS Metric and Threshold Calibration

TODO: Describe the LCS metric and how thresholds were calibrated on validation data.

**The recursion itself works.** TODO: Document whether improvement accumulated 
monotonically across iterations.

| Iteration | LCS Threshold | Validation Primary | Train Data Selected |
| --- | ---: | ---: | ---: |
| 0 (base) | — | TODO | — |
| 1 | TODO | TODO | TODO |
| 2 | TODO | TODO | TODO |
| 3 | TODO | TODO | TODO |
| 4 | TODO | TODO | TODO |
| 5 | TODO | TODO | TODO |
| 6 | TODO | TODO | TODO |

## Bad Case Analysis

TODO: Document patterns observed in bad cases across iterations:
- Which problem types consistently have low LCS scores?
- Which problem types show overconfidence (high LCS, low accuracy)?
- How does the LCS distribution evolve?
- What insights from bad case analysis guided data selection?

## Setup

| | |
| --- | --- |
| Input corpus | OpenThought-30k, TODO: add hash and decontamination details |
| Output | TODO: rows selected per iteration, add hash |
| Base model | `Qwen/Qwen3-8B` (base) @ TODO: add commit hash |
| Iterations | 6 |
| Selection method | LCS threshold-based filtering |
| Validation split | 10% diagnostic subset of each benchmark |
| SFT | TODO: training framework and hyperparameters |
| Evaluation | Qwen2.5-Math harness @ `a45202bd16f1ec06f433442dc1152d0074773465` |
| Hardware | TODO: GPU configuration |
| Seed | TODO: training and evaluation seeds |

Training and inference follow the
[benchmark reference protocol](../../../benchmarks/math-sft-transfer-suite/README.md#reference-protocol-v010).
TODO: Add specific training hyperparameters (epochs, batch size, learning rate, etc.)

## Two evaluation scopes, and why

**In-loop feedback uses a fixed diagnostic subset, never the full test sets.**
The LCS threshold is calibrated on a 10% sample of each validation benchmark. This
bounds the exposure of benchmark content to the optimization loop.

**The reported scores are the full test sets**, evaluated after each iteration finished.
They never influenced threshold calibration or data selection decisions during the run.

| Scope | Used for | Enters the loop? |
| --- | --- | --- |
| Full test sets | baseline/final scores, all reported results | No |
| Diagnostic subset (10%) | LCS threshold calibration and in-loop feedback | Yes, by design |

### Per-iteration full-set scores

All rows are Qwen3-8B base fine-tuned on that iteration's selected data, evaluated
identically on complete test sets.

| Dataset | gsm8k | math | minerva | gaokao | olympiad | amc23 | aime24 | aime25 | aime26 | primary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| *base (untrained)* | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| iter 1 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| iter 2 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| iter 3 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| iter 4 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| iter 5 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| **iter 6 (final)** | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

Question counts: gsm8k 1,319 / math 5,000 / minerva 272 / gaokao 91 /
olympiadbench 675 / amc23 40 / aime24 30 / aime25 30 / aime26 30.
Greedy for the first five, avg@4 for the last four.

### The diagnostic subset, for reference

This is the signal used for LCS threshold calibration — a 10% per-benchmark
sample at seed TODO, fixed before iterations began.

| Iteration | Diagnostic Primary | LCS Threshold |
| --- | ---: | ---: |
| base (untrained) | TODO | — |
| iter 1 | TODO | TODO |
| iter 2 | TODO | TODO |
| iter 3 | TODO | TODO |
| iter 4 | TODO | TODO |
| iter 5 | TODO | TODO |
| iter 6 (final) | TODO | TODO |

## Evolution trace

TODO: Document the evolution of data selection across iterations

| iter | parent | LCS threshold | validation primary | data selected | decision notes |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | — | TODO | TODO | TODO | TODO |
| 2 | 1 | TODO | TODO | TODO | TODO |
| 3 | TODO | TODO | TODO | TODO | TODO |
| 4 | TODO | TODO | TODO | TODO | TODO |
| 5 | TODO | TODO | TODO | TODO | TODO |
| 6 | TODO | TODO | TODO | TODO | TODO |

## Key Findings

TODO: Summarize key findings:
- Does LCS-based selection lead to monotonic improvement?
- Which datasets benefit most from the method?
- How does AIME 2026 (new) compare to AIME 2024/2025?
- What patterns emerged in bad case analysis?
- How does performance compare to random selection or other baselines?

## Reproducibility

TODO: Add complete reproducibility details:
- Environment setup (Python version, library versions, CUDA version)
- LCS computation implementation details
- Threshold calibration algorithm
- Data selection procedure
- Random seeds for all stochastic steps
- Complete training command lines
- Evaluation scripts

## Artifacts

- Training data: TODO: path/hash
- Model checkpoints: TODO: paths
- Evaluation predictions: TODO: paths
- LCS scores: TODO: paths
- Bad case analysis: TODO: paths

## References

TODO: Add references to:
- OpenThought-30k dataset
- Qwen3-8B technical report
- Related work on confidence-based data selection
- Self-teacher methodology
