# Hendrycks MATH

Registration of the upstream MATH dataset for DataLite-RSI.

## Role in DataLite-RSI

| Role | Split | Rows used |
| --- | --- | --- |
| RSI input corpus | `train` | 6,712 rows contributed to the fixed 14,181-row math mix |
| Evaluation target | `test` | 5,000 rows, reported as `math` accuracy |

## Provenance

Competition mathematics problems collected from AMC, AIME, and similar contests,
spanning seven subjects (prealgebra, algebra, intermediate algebra, number
theory, counting and probability, geometry, precalculus) and five difficulty
levels. Every problem ships a full LaTeX reference solution whose final answer is
wrapped in `\boxed{}`.

- Paper: [Measuring Mathematical Problem Solving With the MATH Dataset](https://arxiv.org/abs/2103.03874)
- Upstream repository: <https://huggingface.co/datasets/EleutherAI/hendrycks_math>
- Pinned revision: `21a5633873b6a120296cce3e2df9d5550074f4a3`

This EleutherAI mirror is used because it exposes the subject configurations as
parquet with a stable revision. It carries the same problems as the original
`hendrycks/competition_math` release.

## Derived RSI input corpus

The train split is normalised into the shared schema with `source =
"math_train"`, `source_id` set to the original relative path (for example
`test/prealgebra/674.json`), `subject` taken from the problem's subject
directory, and `difficulty` from the problem's level. `golden_answer` is the
content of the reference solution's `\boxed{}`.

Exact normalised 13-gram decontamination against the eight benchmark test files
reduces 7,500 train rows to 6,712. Combined with GSM8K the corpus is:

```text
math_gsm8k_train_full_seed42_decontam.jsonl
sha256 f9bae97ae63e83c917c9b224bec40dedd7354fcd65e8db30a98b40e540ef3eec
rows   14181  (7,469 gsm8k_train + 6,712 math_train)
```

See [`gsm8k`](../gsm8k/README.md) for the regeneration procedure.

## License and terms

MIT, per the upstream dataset card.

## Privacy, safety, and known limitations

- No PII.
- Answers are LaTeX expressions, so answer matching needs symbolic comparison
  rather than string equality: `\frac{1}{2}`, `0.5`, and `\dfrac12` are the same
  answer. Grading uses the unmodified Qwen2.5-Math grader, committed under
  [`evaluation/qwen25_math/`](../../evaluation/qwen25_math/).
- MATH train and test are drawn from overlapping contest pools. Problems can be
  near-duplicates across the split boundary, which the 13-gram filter catches
  only when the surface form matches. Treat `math` test accuracy as an
  upper-biased estimate for any model trained on MATH train.
- Heavily represented in public pretraining corpora.
