# How to read the reported results

All ten initial submissions are author-reported and **unverified**. A successful
software check does not promote a scientific result to verified status.

| Result group | Headline | Comparator and boundary |
| --- | --- | --- |
| OPSD | Five-set math mean, Avg@12; shown as % | Base and full-pool OPSD; the featured comparator uses 100 optimizer steps. Selection compute is additional. Three seeds are evaluation seeds. |
| Evolver | Eight-set full-test macro mean; shown as % | Base, original Math-3K and rewritten Math-3K. A 10% diagnostic subset from these test files informs proposals; the final full-test result includes that subset. |
| Self-Improver | Seven-set transfer macro mean; shown as % | Base and four iterations. MATH-500/AIME26 are in-loop probes excluded from the headline. Seed identities and rollback evidence remain unresolved. |
| Image synthesis | 0.5 × (GEdit/10 + ImgEdit/5) | Base backbone and stratified synthesis. This normalized composite is not accuracy. Both trained arms use 1,000 pairs; total synthesis compute is not matched. |
| VideoRSI | Video-MME overall accuracy; shown as % | Each of five models versus its own base. No matched-data or larger-data controls. Overall gains do not imply every category improves. |

A difference between percentage scores is reported in **percentage points (pp)**.
Image differences stay on the normalized 0–1 scale. The homepage does not rank
results from different suites, models, or protocol variants together.

## Shared identifiers do not guarantee an identical protocol

Evolver and Self-Improver currently share `math-sft-transfer-suite@0.1.0`, but use
eight and seven headline sets respectively. The website records the explicit
protocol label for each result. A future schema revision should make evaluated
sets, feedback sets, and aggregation rules machine-validated fields; existing
experimental records are preserved pending author clarification.

## Evaluator outputs

- Math and OPSD lightweight evaluators normalize answer strings for fixture
  checks. They are not substitutes for the original mathematical-equivalence
  graders. Parse failures and non-`stop` completions are incorrect; counts can
  overlap when a truncated sample also fails parsing.
- Image evaluation aggregates saved native judge scores. Reaggregation does not
  reproduce hosted judge behavior or original synthesis.
- Video's lightweight evaluator reports `overall` and `categories`; export to a
  result manifest maps `overall` to `metrics.<phase>.primary_score` and merges
  category scores into that phase. This mapping is exercised by a CLI test.

Original predictions and judge records are required for a scientific rerun.
See [reproduction](reproduction.md) for currently executable checks.
