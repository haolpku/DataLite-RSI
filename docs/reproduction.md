# Reproduction and executable checks

The repository currently supports manifest validation, small evaluator fixtures,
and selected core-method tests. All published experiment results remain
**unverified**; no full training rerun is claimed by these checks.

## Run the repository checks

From the repository root, with Python 3.11+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/validate_contributions.py .
python scripts/run_checks.py
python scripts/build_site.py --check
```

The runner discovers test directories under `tests/`, `benchmarks/`,
`evaluation/`, and `rsi/methods/`. Each directory runs in a separate process so
legacy `import evaluator` modules cannot collide. It runs both unittest classes
and plain pytest functions. No API credentials, GPU, models or dataset downloads
are required.

## Run an evaluator fixture

```bash
python benchmarks/opsd-math-competition-suite/evaluator.py --predictions benchmarks/opsd-math-competition-suite/tests/fixture_predictions.jsonl
python benchmarks/math-sft-transfer-suite/evaluator.py --predictions benchmarks/math-sft-transfer-suite/tests/fixture_predictions.jsonl
python benchmarks/image-edit-transfer-suite/evaluator.py --predictions benchmarks/image-edit-transfer-suite/tests/fixture_predictions.jsonl
python benchmarks/videomme-sft-transfer/evaluator.py benchmarks/videomme-sft-transfer/tests/fixture_predictions.jsonl
```

These synthetic inputs exercise parsing and aggregation. Fixture scores are not
research results and must never be copied into submissions or the homepage.
The lightweight math evaluators do not reproduce the original equivalence grader.

## Follow an original experiment

| Method | Available entry points | Missing for an end-to-end reproduction |
| --- | --- | --- |
| [OPSD](../rsi/methods/opsd-data-lite/README.md) | Config and TTPO evaluation source | Selector, selected-row identities, checkpoints and predictions; reconcile 100 steps / 1 epoch |
| [Image synthesis](../rsi/methods/policy-evolving-edit-synthesis/README.md) | Core loop, tests and Docker definition | Serving adapters, scene snapshot, seeds and judge records |
| [VideoRSI](../rsi/methods/video-rsi/README.md) | Reference config and Video-MME evaluator | Full data pipeline, complete run settings and artifacts |
| [Evolver](../rsi/methods/dataflow-evolver/README.md) | Experiment report, config and original grading harness | Upstream implementation, original run artifacts, diagnostics IDs and predictions; public release in preparation |
| [Self-Improver](../rsi/methods/dataflow-self-improver/README.md) | Config and iteration summaries | Method source, run identities, actual rollback logs and outputs |

Commands recorded in original result manifests describe the author environment;
some reference unavailable modules and are not local quick-start commands.
`unknown` revisions and missing artifact URLs are retained rather than inferred.
Container names are recorded environment references, not a guarantee that images
have been published or are retrievable.

## Website maintenance

The homepage is built entirely from this repository, without API keys or a
JavaScript package build. See [site/README.md](../site/README.md) for the local
preview and GitHub Pages workflow.
