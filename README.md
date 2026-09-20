# DataLite-RSI

### Less is More for Recursive Self-Improvement

**Better data decisions through recursive feedback.** DataLite-RSI explores how
selecting, synthesizing, and refining compact training sets can improve models
across mathematical reasoning, image editing, and video understanding.

[**Project homepage**](https://haolpku.github.io/DataLite-RSI/) ·
[Concept](docs/concept.md) · [Results](#reported-results) ·
[Reproduction guide](docs/reproduction.md) · [Contribute](CONTRIBUTING.md)

```text
Candidate data → Select / synthesize / refine → Compact training set
                         ↑                              ↓
                  Updated strategy ← Evaluation & feedback
```

Training takes place within the loop or after search, depending on the method.
“Less” refers to the final training-data budget; search, synthesis, and evaluation
costs are separate. This collection explores the hypothesis, including mixed
results and limitations; it does not establish universal data or compute savings.

## Three perspectives on “Less”

- **Selection — OPSD-Data-Lite:** progressively smaller selected subsets from a
  fixed candidate pool; comparison with a full-pool control at the same reported
  optimizer-step budget. Search adds compute.
- **Synthesis — Policy-Evolving Edit Synthesis:** better transfer with evolving
  synthesis policies at the same final pair budget as stratified synthesis.
- **Transfer — VideoRSI:** one compact SFT dataset evaluated across five model
  families, compared with their respective base models.

DataFlow-Evolver and DataFlow-Self-Improver extend the collection with program
and corpus refinement. Evolver improves over base but falls below original
Math-3K; the two methods use different evaluation sets and are not ranked together.

## Reported results

All current results are **author-reported and unverified**. Scores, comparators,
and costs are linked to the original submissions. Percentages and normalized
image scores use different units; see the [protocol guide](docs/protocol.md).
The [verified leaderboard](results/leaderboard.json) remains separate.

<!-- results:start -->

| Method / model | Protocol / metric | Base → final | Status |
| --- | --- | --- | --- |
| [OPSD · Qwen3-8B](results/submissions/opsd-data-lite-qwen3-8b/result.json) | 5-set math (%) | 58.70 → 62.10 | unverified |
| [Evolver · Qwen2.5-7B](results/submissions/dataflow-evolver-qwen25-7b-math-3k/result.json) | 8-set math (%) | 30.67 → 37.93 | unverified |
| [Self-Improver · Qwen3-8B-Base](results/submissions/dataflow-self-improver-qwen3-8b-math-6k/result.json) | 7-set math (%) | 37.11 → 41.45 | unverified |
| [Policy synthesis · FLUX](results/submissions/policy-evolving-edit-synthesis-flux2-klein-9b/result.json) | Composite (0–1) | 0.79375 → 0.82400 | unverified |
| [Policy synthesis · Qwen Image](results/submissions/policy-evolving-edit-synthesis-qwen-image-edit-2511/result.json) | Composite (0–1) | 0.85010 → 0.86835 | unverified |
| [VideoRSI · Qwen3-VL-8B](results/submissions/video-rsi-qwen3-vl-8b-videomme-4p5k/result.json) | Video-MME (%) | 56.56 → 58.89 | unverified |
| [VideoRSI · LLaVA-OneVision-7B](results/submissions/video-rsi-llava-onevision-7b-videomme-4p5k/result.json) | Video-MME (%) | 58.52 → 59.48 | unverified |
| [VideoRSI · Qwen2.5-VL-3B](results/submissions/video-rsi-qwen25-vl-3b-videomme-4p5k/result.json) | Video-MME (%) | 42.44 → 59.04 | unverified |
| [VideoRSI · Gemma-4-E4B](results/submissions/video-rsi-gemma-4-e4b-videomme-4p5k/result.json) | Video-MME (%) | 47.74 → 48.37 | unverified |
| [VideoRSI · InternVL3-2B](results/submissions/video-rsi-internvl3-2b-videomme-4p5k/result.json) | Video-MME (%) | 34.85 → 55.22 | unverified |

<!-- results:end -->

## Methods and available materials

| Method | What evolves | Available in this repository | Still needed for full reruns |
| --- | --- | --- | --- |
| [OPSD-Data-Lite](rsi/methods/opsd-data-lite/) | Selected training subset | Description, config, math evaluation harness | Selector, selected subsets, checkpoints, predictions |
| [Policy-Evolving Edit Synthesis](rsi/methods/policy-evolving-edit-synthesis/) | Instruction-generation policies | Core loop, rubric tests, environment definition | Serving adapters, scene snapshot, judge records |
| [VideoRSI](rsi/methods/video-rsi/) | Video data pipeline | Description, config, Video-MME evaluation source | Full pipeline, complete run settings, artifacts |
| [DataFlow-Evolver](rsi/methods/dataflow-evolver/) | Data-processing program | Reference config, experiment report, grading harness | Full upstream implementation and run artifacts |
| [DataFlow-Self-Improver](rsi/methods/dataflow-self-improver/) | Data pipeline and corpus | Description, config, iteration summaries | Source revision, run identities, rollback logs |

## Quick start: inspect and validate

No model download or GPU is needed for these checks. Use Python 3.11+.

```bash
git clone https://github.com/haolpku/DataLite-RSI.git
cd DataLite-RSI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/validate_contributions.py .
python scripts/run_checks.py
python scripts/build_site.py --check
python -m http.server 8000 --directory _site
```

Open <http://localhost:8000> for the local homepage. Fixture tests check software
behavior, not the scientific validity of reported experiments. Follow the
[reproduction guide](docs/reproduction.md) for evaluator commands and method-specific
requirements. Some original training commands require unavailable upstream code.

## Repository map

| Directory | Purpose |
| --- | --- |
| [`site/`](site/) | Homepage template, styles, interaction, and editorial metadata |
| [`docs/`](docs/) | Concept, protocols, reproduction, release gaps, contribution guides |
| [`rsi/methods/`](rsi/methods/) | Five method packages, descriptions and configurations |
| [`results/`](results/) | Original result manifests and separate verified leaderboard |
| [`benchmarks/`](benchmarks/) | Four suite definitions, evaluator implementations and fixtures |
| [`datasets/`](datasets/) | Dataset metadata and pinned Hugging Face references |
| [`evaluation/`](evaluation/) | Shared and vendored evaluation harnesses |
| [`docker/`](docker/) | Evaluation environment definitions |
| [`schemas/`](schemas/) / [`templates/`](templates/) | Contribution contracts and starting templates |
| [`scripts/`](scripts/) | Validation, complete test runner, and static-site generation |

## Contribute and maintain

Contribute a [benchmark](benchmarks/README.md), [dataset](datasets/README.md),
[result](results/README.md), or [RSI method](rsi/README.md).
See [CONTRIBUTING.md](CONTRIBUTING.md) or the [中文贡献指南](docs/CONTRIBUTING_zh.md).
The [release gaps](docs/release-gaps.md) identify the materials most useful for
moving current results toward reproducibility.

Update scores in `results/submissions/*/result.json`, then run
`python scripts/build_site.py` to refresh this table. The homepage reads those same
manifests at build time; CI rejects a stale README. See [site maintenance](site/README.md).

Keep code, manifests, docs and website source in GitHub. Store large datasets and
model artifacts externally; the current dataset hub is
[Lite-RSI on Hugging Face](https://huggingface.co/datasets/lhpku20010120/Lite-RSI).
Do not commit model weights, generated media, secrets, or dataset archives.

## Citation and licensing

Paper and citation metadata are pending. Refer to this repository and a commit
when discussing the current collection. Method manifests currently declare
`NOASSERTION`; no project-wide reuse license has been assigned. Vendored components
retain their own license notices.
