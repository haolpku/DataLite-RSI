# Review configuration

`review` controls the static candidate review performed after a generated pipeline runs. It has two independent layers:

1. a sample-level judge (`four_dimensions` or `criteria`);
2. deterministic full-file checks and optional embedding distribution evidence.

The active runtime reads only the fields documented here. Older flat names are not accepted.

## Sample-level review

| Field | Meaning |
|---|---|
| `review.mode` | `four_dimensions` uses the general correctness/relevance/difficulty/schema rubric. `criteria` asks the LLM to assess every `task.quality_criteria` item and uses the satisfied-criteria rate. |
| `review.sampling.size` | Number of JSONL records placed in the judge prompt, sampled with a deterministic reservoir sampler. |
| `review.sampling.seed` | Reservoir sampling seed. Use a fixed value for comparable iterations. |
| `review.criteria.pass_rate` | In criteria mode, minimum fraction of criteria marked `met` for the sample-level pass gate. This is separate from the score blend. |
| `review.four_dimensions.pass_thresholds.*` | Per-dimension pass gates for `correctness`, `relevance`, and `schema`. Used only in four-dimension mode. |
| `review.four_dimensions.weights.*` | Weights for the four-dimension composite (`correctness`, `relevance`, `difficulty`, `schema`). Used only in four-dimension mode. |

Criteria mode does not produce or use the four dimension scores. It returns per-criterion `met`, evidence, and suggestions in `ReviewResult.domain_feedback`. `issues` remains reserved for deterministic failures or explicitly reported critical failures.

## Deterministic validation

| Field | Meaning |
|---|---|
| `review.validation.enabled` | Run full-file validation. |
| `review.validation.max_null_rate` | Maximum fraction of required schema slots that may be empty. |

The full-file metrics are `dup_rate`, `null_rate`, `parse_valid_rate`, and `contamination_rate`. They remain observable validation evidence; four-dimension mode also uses them in its hard-metric score factor. Criteria mode uses the criteria satisfaction rate as its sample-level score and does not add duplicate or parse-rate thresholds.

## Decontamination

| Field | Meaning |
|---|---|
| `review.decontamination.reference_files` | JSONL/JSON/text files used to build the n-gram contamination index. |
| `review.decontamination.ngram_size` | Number of normalized tokens (or CJK characters) in an exact overlap window. |
| `review.decontamination.max_rate` | Maximum fraction of candidate rows that may contain an overlap. |
| `review.decontamination.reference_fields` | Optional fields read from reference rows. Empty means all nested text fields. |
| `review.decontamination.candidate_fields` | Optional candidate fields scanned for overlap. Empty means all nested text fields. |

## Embedding quality

Embedding quality is optional. It is enabled when at least one metric under
`review.embedding_quality.metrics` has `enabled: true`; there is no coarse global
switch. The three metrics are independently selectable:

| Field | Meaning |
|---|---|
| `review.embedding_quality.metrics.das.enabled` | Enable DAS, implemented as `-MMD(candidate, proxy)`. DAS is the only metric that needs a proxy dataset and MMD settings. |
| `review.embedding_quality.metrics.vendi.enabled` | Enable normalized Vendi diversity from candidate embeddings. |
| `review.embedding_quality.metrics.nearest_neighbor.enabled` | Enable nearest-neighbor cosine statistics from candidate embeddings. |
| `review.embedding_quality.metrics.*.weight` | Relative weight of that enabled metric in the embedding score. At least one enabled metric must have a positive weight. |
| `review.embedding_quality.review_score_weight` | Fraction of the final `review_score` assigned to the combined embedding score. The sample-level component gets `1 - weight`. Must be in `[0,1]`. |

The configuration layers reflect the computation:

- `sampling` is shared by all enabled metrics and fixes the candidate/proxy sample size and seed.
- `candidate` is shared by all metrics because every metric starts from candidate embeddings.
- `metrics.das.mmd` and `metrics.das.proxy` are DAS-only.
- `embedding` contains the MMD normalization switch plus embedding request failure-rate and cache settings.
- `service` contains the shared embedding API client settings.

| Field | Meaning |
|---|---|
| `review.embedding_quality.sampling.size` | Candidate sample size, and proxy sample size when DAS is enabled. |
| `review.embedding_quality.sampling.seed` | Deterministic sampling seed. |
| `review.embedding_quality.sampling.require_requested_size` | If true, fewer valid records than requested makes evaluation fail. |
| `review.embedding_quality.metrics.das.mmd.sigma` | RBF kernel bandwidth in `exp(-||x-y||²/(2*sigma²))`. Larger values make distant embeddings look more similar. Must be positive. |
| `review.embedding_quality.metrics.das.mmd.estimator` | `biased` includes kernel diagonal/self-similarity and is stable for small samples. `unbiased` removes diagonal terms, needs at least two records per side, and can have higher variance. |
| `review.embedding_quality.embedding.normalize_embeddings` | L2-normalize vectors before MMD. Vendi and nearest-neighbor metrics always use cosine similarity, so they normalize internally regardless of this switch. |
| `review.embedding_quality.embedding.max_failure_rate` | Maximum fraction of failed embedding requests before the metric evaluation is marked failed. |
| `review.embedding_quality.embedding.cache_dir` | Optional cache for encoded candidate/proxy samples. Empty uses the run workspace cache. |

`candidate.user_field`/`assistant_field` identify the two fields in generated
records; setting both to `messages` enables conversation parsing for the
embedding request only. `metrics.das.proxy.source`, `path`, `name`, `split`, and
proxy field names identify the fixed DAS comparison corpus. `service.model_name`,
`base_url`, `api_key`, `max_concurrent_requests`, `max_retries`,
`truncate_prompt_tokens`, and `truncation_side` configure the OpenAI-compatible
embedding service.

The final blend is:

```text
embedding_score = weighted mean of enabled metric scores
sample_score = criteria_met_rate                  # mode=criteria
sample_score = weighted_four_dimension_score      # mode=four_dimensions
review_score = (1 - review_score_weight) * sample_score
             + review_score_weight * embedding_score
```

If all three metric switches are false, no embedding evaluator is constructed
and `review_score` contains only the sample-level score.
