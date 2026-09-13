# OPSD (Open Problem Self-Discovery)

A data-centric RSI method that uses a proxy metric to guide data selection for mathematical problem-solving.

## Overview

OPSD addresses the challenge of selecting high-quality training data by leveraging a self-teacher confidence metric. Rather than recursively rewriting the dataset or the data pipeline, OPSD analyzes model behavior on validation data to calibrate a selection threshold, then applies this threshold to filter training examples.

## Method

### Core Innovation

The method introduces **LSC (Late Support Collapse)** as an unlabeled rollout-quality proxy. It measures how much the model's top-20 token support changes when the final 20% of a fixed reference reasoning trace is revealed. Lower LSC indicates a higher-priority candidate for review or additional sampling. This metric enables:

1. **Bad case analysis**: Identifying validation examples where the model struggles
2. **Threshold calibration**: Freezing the protocol and selecting a low-LSC tail on validation rollouts
3. **Data selection**: Applying the previous-evaluation LSC threshold and selecting exactly 3,200 rows

### Workflow

```
1. Initial Training
   └─> Train base model (Qwen3-8B) on selected subset of OpenThought-30k

2. Validation Analysis (each iteration)
   ├─> Generate rollouts on validation benchmarks
   ├─> Compute LSC metric for each validation example
   ├─> Analyze bad cases (low accuracy + LSC patterns)
   └─> Calibrate threshold based on validation performance

3. Data Selection
   ├─> Apply LSC metric to training data pool (OpenThought-30k)
   ├─> Filter examples using calibrated threshold
   └─> Select the fixed low-LSC candidate tail for review or additional sampling

4. Iteration
   └─> Train next iteration model on newly selected data
   └─> Repeat steps 2-4 for 6 iterations total
```

## Base Model

- **Model**: Qwen/Qwen3-8B
- **Rationale**: Qwen3-8B is the fixed base model used by the transfer protocol; the same tokenizer and logprob interface are required for comparable support sets.

## Training Data

- **Source**: OpenThought-30k subset
- **Selection**: LSC ascending low-tail filtering
- **Size**: Fixed 3,200 examples selected from the first 6,400 OpenThought-30k candidates using the current threshold.

## LSC Metric

LSC (Late Support Collapse) uses privileged-context prompt logprobs for one fixed student rollout. Let `S_t` be the base/student top-20 token IDs at position `t`, and `T_80,t` and `T_100,t` the supports after adding the first 80% or all of a text-distinct reference reasoning trace (both retain the final answer). For each common scored token:

```text
J_l = mean_t |S_t ∩ T_l,t| / max(1, |S_t ∪ T_l,t|)
LSC = J_80 - J_100
```

Scoring is forced on the same continuation, with an 8192-token model limit and a 1024-token response window. Only top-20 IDs are used; no full-vocabulary distance is inferred. If a privileged prompt exceeds budget, only the reasoning prefix is cropped, the final answer is retained, and `truncated=true` is recorded. A fixed first text-distinct reference is selected before labels are read. For this run, the first 6,400 candidates are scored and exactly 3,200 rows are selected using the current threshold.

For multiple fixed reference contexts, compute `LSC(q,c)` independently and average them as `CrossLSC(q)`. A single context remains named LSC. The optional frozen shape estimate is `sigmoid(0.842847 + 113.071472*LSC + 8.183589*G80)` and is auxiliary; LSC remains the selection key.

## Evaluation Protocol

### Validation Datasets (In-Loop)

The supplied run uses the previous iteration's AIME26 evaluation for threshold calibration. The generic GSM8K/MATH diagnostic workflow remains available for future runs.

### Test Datasets (Held-Out)

Used only for final evaluation, never enter the optimization loop:
- GSM8K (full test set)
- Hendrycks MATH (full test set)
- AIME 2024
- AIME 2025
- **AIME 2026** (new)
- AMC 2023
- Minerva Math
- OlympiadBench
- Gaokao 2024 Mix

The reported AIME26 diagnostic uses 30 problems, 4 samples per seed, and 3 evaluation seeds (360 judgments total). Training runs use 100 optimizer steps. At iteration `i`, the LSC threshold is computed from the previous iteration's evaluation and then applied to the fixed 6,400-row candidate pool to select exactly 3,200 rows for the next update. Full-suite scores are only populated when those benchmark artifacts are available.

## Iteration Results

The available evaluation is the AIME26 30-problem aggregate below. Each cell is correct judgments out of 360 across three seeds; percentages are `correct / 360`.

| Iteration | Primary Score | GSM8K | MATH | AIME24 | AIME25 | AIME26 | AMC23 | Minerva | Olympiad | Gaokao |
|-----------|---------------|-------|------|--------|--------|--------|-------|---------|----------|--------|
| 0 (Base)  | —             | —     | —    | —      | —      | 244/360 (67.78%) | —     | —      | —        | —      |
| OPSD raw  | —             | —     | —    | —      | —      | 248/360 (68.89%) | —     | —      | —        | —      |
| 1         | —             | —     | —    | —      | —      | 246/360 (68.33%) | —     | —      | —        | —      |
| 2         | —             | —     | —    | —      | —      | 245/360 (68.06%) | —     | —      | —        | —      |
| 3         | —             | —     | —    | —      | —      | 253/360 (70.28%) | —     | —      | —        | —      |
| 4 (rollback) | —           | —     | —    | —      | —      | 251/360 (69.72%) | —     | —      | —        | —      |
| 4         | —             | —     | —    | —      | —      | 252/360 (70.00%) | —     | —      | —        | —      |
| 5 (rollback) | —           | —     | —    | —      | —      | 243/360 (67.50%) | —     | —      | —        | —      |
| 5         | —             | —     | —    | —      | —      | 259/360 (71.94%) | —     | —      | —        | —      |
| 6         | —             | —     | —    | —      | —      | 258/360 (71.67%) | —     | —      | —        | —      |

## Bad Case Analysis

Bad-case analysis should report patterns observed in:
- Which problem types have low LSC but high difficulty?
- Which problem types have high LSC but low accuracy (overconfidence)?
- How does LSC distribution evolve across iterations?
- What insights guided data selection in each iteration?

## Key Findings

Key findings should summarize:
- Does LSC-based selection improve over random selection?
- How does performance compare to other data selection strategies?
- Which datasets benefit most from OPSD selection?
- Does the method show monotonic improvement across iterations?

## Configuration

Reference configuration: `configs/opsd-qwen3-openthought.yaml`

## Dependencies

- Qwen3-8B base model
- OpenThought-30k training data
- Math-SFT-Transfer-Suite benchmark
- Qwen2.5-Math evaluation harness (for grading)

## Reproducibility

Record training hyperparameters separately for each run. The LSC implementation is in [`lsc.py`](lsc.py); threshold calibration freezes top-k=20, levels 80/100, the 8192/1024 budget, and ascending ranking before any external labels are inspected. Report the diagnostic subset seed, selection seed, model/software revisions, and compute requirements. External labels are for post-hoc Spearman, AUC, precision, recall, and lift only.

## References

References to add:
- OpenThought dataset paper/release
- Qwen3 technical report
- Related work on confidence-based data selection
- Self-teacher methodology
