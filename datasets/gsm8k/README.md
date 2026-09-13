# GSM8K

Registration of the upstream GSM8K dataset for DataLite-RSI.

## Role in DataLite-RSI

GSM8K plays two distinct roles, and they must not be confused:

| Role | Split | Rows used |
| --- | --- | --- |
| RSI input corpus | `train` | 7,469 rows contributed to the fixed 14,181-row math mix |
| Evaluation target | `test` | 1,319 rows, reported as `gsm8k` accuracy |

## Provenance

Grade-school math word problems written and solved by human annotators, released
by OpenAI. Each record contains a natural-language `question` and an `answer`
whose final line is a `#### <value>` numeric answer.

- Paper: [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
- Upstream repository: <https://huggingface.co/datasets/openai/gsm8k>
- Pinned revision: `740312add88f781978c0658806c59bc2815b9866`

## Derived RSI input corpus

The DataLite-RSI math input corpus normalises the train split into the shared
schema `source, source_id, problem, golden_solution, golden_answer, difficulty,
subject`, with `source = "gsm8k_train"` and `difficulty = "grade_school"`. Rows
whose `problem` shares an exact normalised 13-gram with any configured benchmark
test file are dropped, which is why 7,469 of 7,473 train rows survive.

The derived corpus is a local artifact, not a redistributed dataset. It is
identified by content hash rather than by a Hugging Face revision:

```text
math_gsm8k_train_full_seed42_decontam.jsonl
sha256 f9bae97ae63e83c917c9b224bec40dedd7354fcd65e8db30a98b40e540ef3eec
rows   14181  (7,469 gsm8k_train + 6,712 math_train)
```

Regenerate it from this pinned revision plus
[`hendrycks-math`](../hendrycks-math/README.md) using the decontamination
settings recorded in
[`rsi/methods/dataflow-evolver/configs/math-periodic.yaml`](../../rsi/methods/dataflow-evolver/configs/math-periodic.yaml)
(`review.decontamination`: 13-gram, fields `question`/`problem`).

## License and terms

MIT, per the upstream dataset card. Redistribution and derivative use are
permitted with attribution.

## Privacy, safety, and known limitations

- No PII. Problems are synthetic word problems about fictional people.
- Answers are short numerics, so string-normalised answer matching is reliable
  but sensitive to unit and formatting conventions.
- Widely used in pretraining corpora; contamination against the test split is a
  standing risk for any model not trained under a controlled data pipeline. The
  13-gram filter documented above removes exact overlap only, not paraphrase.
