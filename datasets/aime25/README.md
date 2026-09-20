# AIME 2025

30 problems from the 2025 AIME I competition.

**Source:** Qwen2.5-Math evaluation harness, `https://github.com/QwenLM/Qwen2.5-Math/tree/a45202bd16f1ec06f433442dc1152d0074773465/evaluation/data`.
A HuggingFace mirror exists at
[MathArena/aime_2025](https://huggingface.co/datasets/MathArena/aime_2025)
(sha `c94da77eb22bbd6439e62a323bec18493a421302`).
All 30 answers are identical; problem text has minor LaTeX formatting
differences (e.g. `\operatorname{if}` vs `	ext{if}`).
These differences do not affect scoring because `math_equal` uses symbolic
comparison, not string matching on problem text.

**Metrics:** The dataset is shared by two benchmark protocols:

- `math-sft-transfer-suite`: avg@4, temperature 0.6, four samples per problem;
- `opsd-math-competition-suite`: avg@12, temperature 1.0, Qwen3 thinking.
