# Contributing to DataLite-RSI

Thank you for helping build reproducible research on recursive
self-improvement. Contributions can be small and focused: a new benchmark,
dataset registration, result submission, evaluator fix, or RSI method is useful
on its own.

This is an open research initiative around **Less is More for RSI**. The current
experiments are initial explorations, and we invite the community to shape what
comes next. Research questions, new tasks, reproductions, matched-budget controls,
and negative findings are welcome alongside code. You can start with an
[issue](https://github.com/haolpku/DataLite-RSI/issues) before having a complete
implementation or experiment.

For a Chinese version, see [docs/CONTRIBUTING_zh.md](docs/CONTRIBUTING_zh.md).

## Before opening a pull request

1. Search existing issues and pull requests for related work.
2. For a large or potentially incompatible design, open the relevant proposal
   issue first. Small fixes and self-contained additions can go directly to a
   pull request.
3. Create a focused branch and avoid mixing unrelated changes.
4. Never include credentials, private data, model weights, dataset archives, or
   Docker image tarballs.

## Contribution types

### 1. Benchmark

Create `benchmarks/<benchmark-id>/` containing:

```text
benchmarks/<benchmark-id>/
├── benchmark.json       Required benchmark manifest
├── README.md            Task, data, metrics, and limitations
├── evaluator.py         Or an equivalent documented entry point
└── tests/               Small deterministic evaluator tests
```

Start from [`templates/benchmark.json`](templates/benchmark.json). A benchmark
pull request must:

- select exactly one primary track: `llm`, `multimodal`, or `generative`;
- define each metric and whether higher values are better;
- refer to a registered dataset by ID and immutable revision;
- document contamination, licensing, privacy, and safety considerations;
- include a small test fixture that does not require downloading the full data;
- define a deterministic evaluator entry point and its expected inputs/outputs.

Large benchmark files belong on Hugging Face, not under `benchmarks/`.

### 2. Dataset

Upload the data to Hugging Face, then create
`datasets/<dataset-id>/dataset.json` and `datasets/<dataset-id>/README.md`.
Start from [`templates/dataset.json`](templates/dataset.json).

The metadata must include:

- the Hugging Face repository ID and an immutable commit revision;
- split names, modalities, formats, and approximate sizes;
- provenance and collection or generation procedure;
- license, terms of use, and access restrictions;
- PII, consent, safety, and known-bias disclosures;
- checksums or generation scripts when practical.

Do not add large data files to GitHub. Tiny synthetic fixtures for evaluator
tests are allowed when redistribution is clearly permitted.

### 3. Experimental result

Create `results/submissions/<submission-id>/result.json` from
[`templates/result.json`](templates/result.json). You may include a short
`README.md` plus compact aggregate tables or plots. Raw generations, videos,
traces, and other large artifacts should be stored in a separate Hugging Face
repository and linked from the manifest.

A result submission must:

- use `"status": "unverified"`;
- refer to an existing benchmark ID and version;
- disclose model identity, method, sampling settings, seeds, and budget;
- report baseline and post-RSI results under comparable settings;
- pin code, dataset, and container revisions;
- include the exact reproduction command;
- disclose failed, excluded, or manually selected runs.

Do not edit `results/leaderboard.json` in a result pull request. It is generated
after maintainers verify a submission.

### 4. RSI method or implementation

Create `rsi/methods/<method-id>/` containing:

```text
rsi/methods/<method-id>/
├── method.json          Required method manifest
├── README.md            Algorithm, assumptions, and limitations
├── src/                 Implementation
├── configs/             Reproducible default configurations
└── tests/               Unit and smoke tests
```

Start from [`templates/method.json`](templates/method.json). Document:

- what is recursively modified: prompts, memory, tools, data, code, weights, or
  another component;
- the feedback signal and acceptance rule;
- stopping conditions and resource budget;
- whether human intervention is used at any iteration;
- supported DataLite-RSI tracks and required external services;
- safety constraints, rollback behavior, and known failure modes.

Keep model-specific adapters separate from the core algorithm when possible.
Never commit tokens or bake credentials into configs or containers.

## Evaluation environment and Docker

Docker source files belong in `docker/<track-or-environment>/`. Built images
belong in GHCR. An environment contribution should include:

- a readable `Dockerfile` and, where useful, `docker-compose.yml`;
- pinned system and Python package versions;
- OCI source, description, and license labels;
- a non-root runtime user when supported;
- a health check or smoke-test command;
- documentation for CPU/GPU requirements and mounted paths.

Do not copy model weights or benchmark data into an image. Download or mount
them at runtime, and pin their revisions in the result manifest. Do not put
secrets in `ARG`, `ENV`, image layers, examples, or build logs.

## Naming and manifests

- IDs and directory names use lowercase `kebab-case`.
- The manifest `id` must match its parent directory name.
- JSON files use two-space indentation and end with a newline.
- Versions use semantic versions such as `0.1.0` where applicable.
- Use immutable revisions for verified work; avoid `main` and `latest`.
- New manifests use the current `schema_version` shown in the templates.

Machine-readable schemas are under [`schemas/`](schemas/). The lightweight
repository validator checks the required cross-file conventions without adding
a runtime dependency on a JSON Schema library.

## Local checks

Run before opening a pull request:

```bash
python scripts/validate_contributions.py .
python -m unittest discover -s tests -v
```

Also run the tests belonging to the evaluator or method you changed. Pull
requests run the repository checks automatically.

## Pull-request checklist

- Choose one contribution type as the primary purpose of the pull request.
- Explain the scientific motivation and user-visible change.
- Link the proposal or tracking issue when one exists.
- Include tests or explain why tests do not apply.
- Update documentation and manifests together with the implementation.
- Confirm that all data and dependencies may legally be redistributed or used
  as documented.
- Confirm that no credentials, private data, or large generated artifacts are
  included.

Maintainers may request evaluation reruns, licensing clarification, schema
changes, or independent verification before accepting or ranking a result.
