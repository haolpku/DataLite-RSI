# OPSD (Open Problem Self-Discovery)

A data-centric RSI method that uses a proxy metric to guide data selection for mathematical problem-solving.

## Overview

OPSD addresses the challenge of selecting high-quality training data by leveraging a self-teacher confidence metric. Rather than recursively rewriting the dataset or the data pipeline, OPSD analyzes model behavior on validation data to calibrate a selection threshold, then applies this threshold to filter training examples.

## Method

### Core Innovation

The method introduces **LCS (TODO: expand acronym)** as a proxy metric for self-teacher confidence—a quantitative measure of how well the model believes it can solve a given problem. This metric enables:

1. **Bad case analysis**: Identifying validation examples where the model struggles
2. **Threshold calibration**: Computing LCS thresholds on validation rollouts
3. **Data selection**: Filtering training data based on calibrated thresholds

### Workflow

```
1. Initial Training
   └─> Train base model (Qwen3-8B) on selected subset of OpenThought-30k

2. Validation Analysis (each iteration)
   ├─> Generate rollouts on validation benchmarks
   ├─> Compute LCS metric for each validation example
   ├─> Analyze bad cases (low accuracy + LCS patterns)
   └─> Calibrate threshold based on validation performance

3. Data Selection
   ├─> Apply LCS metric to training data pool (OpenThought-30k)
   ├─> Filter examples using calibrated threshold
   └─> Select diverse, high-confidence subset

4. Iteration
   └─> Train next iteration model on newly selected data
   └─> Repeat steps 2-4 for 6 iterations total
```

## Base Model

- **Model**: Qwen/Qwen3-8B
- **Rationale**: TODO: explain choice of Qwen3-8B

## Training Data

- **Source**: OpenThought-30k subset
- **Selection**: LCS threshold-based filtering
- **Size**: TODO: specify number of examples selected per iteration

## LCS Metric

TODO: Describe the LCS (TODO: expand acronym) metric in detail:
- Definition and computation
- Why it serves as a proxy for self-teacher confidence
- How it correlates with problem difficulty and model capability
- Threshold calibration strategy

## Evaluation Protocol

### Validation Datasets (In-Loop)

Used for threshold calibration and iteration feedback:
- GSM8K (10% diagnostic subset)
- Hendrycks MATH (10% diagnostic subset)

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

All evaluation uses the math-sft-transfer-suite benchmark protocol (avg@4, temperature 0.6).

## Iteration Results

TODO: Fill in results for each iteration

| Iteration | Primary Score | GSM8K | MATH | AIME24 | AIME25 | AIME26 | AMC23 | Minerva | Olympiad | Gaokao |
|-----------|---------------|-------|------|--------|--------|--------|-------|---------|----------|--------|
| 0 (Base)  | TODO          | TODO  | TODO | TODO   | TODO   | TODO   | TODO  | TODO    | TODO     | TODO   |
| 1         | TODO          | TODO  | TODO | TODO   | TODO   | TODO   | TODO  | TODO    | TODO     | TODO   |
| 2         | TODO          | TODO  | TODO | TODO   | TODO   | TODO   | TODO  | TODO    | TODO     | TODO   |
| 3         | TODO          | TODO  | TODO | TODO   | TODO   | TODO   | TODO  | TODO    | TODO     | TODO   |
| 4         | TODO          | TODO  | TODO | TODO   | TODO   | TODO   | TODO  | TODO    | TODO     | TODO   |
| 5         | TODO          | TODO  | TODO | TODO   | TODO   | TODO   | TODO  | TODO    | TODO     | TODO   |
| 6         | TODO          | TODO  | TODO | TODO   | TODO   | TODO   | TODO  | TODO    | TODO     | TODO   |

## Bad Case Analysis

TODO: Document patterns observed in bad cases:
- Which problem types have low LCS but high difficulty?
- Which problem types have high LCS but low accuracy (overconfidence)?
- How does LCS distribution evolve across iterations?
- What insights guided data selection in each iteration?

## Key Findings

TODO: Summarize key findings:
- Does LCS threshold-based selection improve over random selection?
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

TODO: Add details on:
- Training hyperparameters (learning rate, batch size, epochs, etc.)
- LCS computation implementation
- Threshold calibration algorithm
- Data selection randomness seeds
- Compute requirements

## References

TODO: Add references to:
- OpenThought dataset paper/release
- Qwen3 technical report
- Related work on confidence-based data selection
- Self-teacher methodology
