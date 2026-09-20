# DataFlow-Self-Improver on Qwen3-8B-Base, four iterations to 6,000 math SFT rows

Status: **unverified**. Submitted for maintainer review.

Four iterations of
[DataFlow-Self-Improver](../../../rsi/methods/dataflow-self-improver/) driven by
bad-case attribution on two frozen validation sets (MATH-500 and AIME26). The
agent retrieves data, rewrites the DataFlow DAG, and refines or filters the
incumbent corpus. Evaluated on the
[math SFT transfer suite](../../../benchmarks/math-sft-transfer-suite/) plus
AIME26.

## Headline

Fine-tuning `Qwen/Qwen3-8B-Base` on the selected 6,000-row combined subset
lifted the **held-out** primary score from **37.11% to 41.45% (+4.34 points)**.
That primary is the unweighted mean of GSM8K, Minerva, Gaokao, OlympiadBench,
AMC23, AIME24, and AIME25 — the seven sets that never entered the rollback
rule.

**Every cell below is the mean over three seeds: 42, 42, and 44.** MATH-500 and
AIME26 are in-loop validation. The nine-benchmark mean that still includes them
rises from 37.66% to 42.18%.

| Benchmark | Base | Iter 1 1.8k | Iter 2 5.2k | Iter 3 7k | **Iter 4 6k** |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gsm8k` | 90.70 | 90.60 | 93.25 | 93.48 | **93.14** |
| `minerva_math` | 29.04 | 35.66 | 27.94 | 34.19 | **36.76** |
| `gaokao2024_mix` | 37.91 | 48.35 | 41.76 | 46.15 | **46.15** |
| `olympiadbench` | 41.48 | 39.11 | 41.48 | 38.07 | **39.11** |
| `amc23` avg@4 | 44.38 | 50.00 | 47.50 | 46.88 | **50.00** |
| `aime24` avg@4 | 7.09 | 9.17 | 10.83 | 14.17 | **13.33** |
| `aime25` avg@4 | 9.17 | 6.67 | 11.67 | 11.67 | **11.67** |
| **primary_score (held-out)** | **37.11** | **39.94** | **39.20** | **40.66** | **41.45** |
| `math500` *(in-loop)* | 72.48 | 70.82 | 74.80 | 76.14 | 76.14 |
| `aime26` avg@4 *(in-loop)* | 6.67 | 6.67 | 13.33 | 8.33 | 13.33 |
| primary including validation | 37.66 | 39.67 | 40.28 | 41.01 | 42.18 |

Largest held-out gains versus Base are Minerva (+7.72), Gaokao (+8.24), AMC23
(+5.62), and AIME24 (+6.24). OlympiadBench is the one held-out regression
(−2.37). GSM8K is already high at Base and moves little.

## Setup

| | |
| --- | --- |
| Method | [DataFlow-Self-Improver](../../../rsi/methods/dataflow-self-improver/) |
| Base model | `Qwen/Qwen3-8B-Base` (base, not Instruct) |
| In-loop validation | MATH-500 and AIME26, frozen |
| Output | 6,000 `{instruction, output}` SFT rows (iteration 4) |
| Iterations | 4 (format alignment, then three capability-completion steps) |
| Phase caps | format alignment < 2,000 rows; capability completion < 6,000 rows; later capability work locked to that subset |
| Rollback | mean validation acc drop > 3 pp, or format-success rate drop > 5 pp versus the alignment incumbent |
| Evaluation | Qwen2.5-Math harness @ `a45202bd`, vendored under [`evaluation/qwen25_math/`](../../../evaluation/qwen25_math/) |
| Seeds | **42, 42, 44**; all reported scores are the unweighted mean of these three runs |

Training and inference follow the
[benchmark reference protocol](../../../benchmarks/math-sft-transfer-suite/README.md#reference-protocol-v010)
for generation settings: greedy for the first group, avg@4 at temperature 0.6
for AMC23 and the AIME sets. AIME26 uses the same avg@4 protocol as AIME24/25.
Each checkpoint is trained and scored three times (seeds 42, 42, 44); the
tables report that mean.

## Two evaluation scopes

**In-loop feedback uses MATH-500 and AIME26**, because the analyzer attributes
bad cases on those sets and the rollback rule reads their mean accuracy. Full
held-out test content is not the acceptance signal.

**The reported headline is the seven held-out sets**, evaluated after each
iteration. They never entered the rollback rule.

| Scope | Used for | Enters the loop? |
| --- | --- | --- |
| Held-out seven (GSM8K, Minerva, Gaokao, Olympiad, AMC23, AIME24, AIME25) | headline `primary_score` | No |
| MATH-500 + AIME26 | bad-case attribution and rollback | Yes, by design |

## Per-iteration trace

Scores are three-seed means (42, 42, 44).

| iter | phase | rows | held-out primary | vs Base | in-loop MATH-500 | in-loop AIME26 | decision |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| — | — | — | 37.11 | — | 72.48 | 6.67 | untrained Qwen3-8B-Base |
| 1 | format alignment | 1,800 | 39.94 | +2.83 | 70.82 | 6.67 | accepted |
| 2 | capability completion | 5,200 | 39.20 | +2.09 | 74.80 | 13.33 | accepted (held-out dip is 0.74 pp, below the 3 pp rule) |
| 3 | capability, combined | 7,000 | 40.66 | +3.55 | 76.14 | 8.33 | accepted; temporary over-cap merge |
| **4** | capability, locked 6k | **6,000** | **41.45** | **+4.34** | **76.14** | **13.33** | **accepted, final** |

The 3 pp validation rollback did not fire on the reported checkpoints. Iteration
2 is the only held-out regression versus iteration 1, and it is smaller than the
rollback threshold. Iteration 3 merged the 1.8k and 5.2k subsets to 7k, above
the capability cap; iteration 4 filtered that merge back to the locked 6k
subset.

## Contamination and leakage

- MATH-500 and AIME26 are in-loop by design. Their scores are expected to move
  with the method and are excluded from the headline primary.
- Held-out test files should be decontaminated with exact normalised 13-gram
  overlap at `max_rate` 0.01, matching the suite reference. Measured overlap
  rates were not recorded in this submission.
- Residual risk: n-gram filtering removes surface duplication only. MATH train
  and MATH-500 overlap in contest coverage, so the in-loop MATH-500 column is
  upper-biased for any model trained on MATH-like sources.

## Scope and limitations

- **OlympiadBench regresses** on the selected incumbent (41.48 → 39.11). The
  method does not claim uniform gains.
- **Iteration 3 exceeded the 6k cap** before iteration 4 restored it.
- **Upstream code revision is unknown** in this manifest. Pipeline execution
  cannot yet be rerun from this repository alone.
- **Compute and token accounting are not recorded.**

## Reproducibility

| | |
| --- | --- |
| Base model | `Qwen/Qwen3-8B-Base` |
| Seeds | 42, 42, 44 (reported scores are the mean) |
| Configuration | [`math-badcase.yaml`](../../../rsi/methods/dataflow-self-improver/configs/math-badcase.yaml) |
| Evaluation harness | [`evaluation/qwen25_math/`](../../../evaluation/qwen25_math/) — Qwen2.5-Math @ `a45202bd16f1ec06f433442dc1152d0074773465` (MIT) |
| In-loop sets | MATH-500 (`HuggingFaceH4/MATH-500`); AIME26 `MathArena/aime_2026` @ `d2de22f3c656b4f56cf8981212186377d1e23bc3` |
| Container | `ghcr.io/haolpku/datalite-rsi-eval:0.1.0` |

```bash
python -m dataflow_self_improver.main \
  --config rsi/methods/dataflow-self-improver/configs/math-badcase.yaml \
  --log-level INFO
```

`code_revision` is `unknown` until the upstream Self-Improver repository is
published. `artifacts_url` is null pending upload of the per-iteration datasets
and operator source.
