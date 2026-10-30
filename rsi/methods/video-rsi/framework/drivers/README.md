# VideoRSI experiment drivers

`drivers/` contains the reproducible outer loop around the framework in
`src/video_rsi/`.  It is deliberately separate from generated experiment
directories: code can be versioned and reviewed, while video-derived data,
predictions, checkpoints and logs remain outside Git.

## Components

| File | Responsibility |
| --- | --- |
| `run_experiment.py` | Execute one named pipeline version, snapshot its source, preserve per-video stages and build its pool. |
| `evaluate_run.py` / `evaluation.py` | Apply fixed, model-free hard gates and compare a candidate with its parent. |
| `evolve.py` | Run the bounded, stateful Codex v0 -> vN loop in isolated candidate workspaces. |
| `api_backends.py` | Runtime-only OpenAI-compatible adapters. It reads credentials from an explicit private path or environment; no credentials are stored here. |

## Minimal commands

Run from the repository root after installing the package or setting
`PYTHONPATH=src`:

```bash
# No local target-model GPU/service: use a general API only as reference feedback.
export RSI_EVALUATION_MODEL=provider/general-vlm
export RSI_MODEL_ROLE=reference

# Create one immutable v0 run. Use a separate, writable experiment root.
PYTHONPATH=src python drivers/run_experiment.py \
  --caption-root /path/to/caption_corpus \
  --experiment-root /path/to/video_rsi_experiments \
  --version v0 --limit 10 --config-path /private/rsi_api_config.json

# Recompute only the fixed-policy decision for a completed version.
PYTHONPATH=src python drivers/evaluate_run.py \
  /path/to/video_rsi_experiments/versions/v0/outputs/<run-id>

# Optional Codex evolution. Configure the Codex binary and credentials in your
# own environment before invoking this command.
PYTHONPATH=src python drivers/evolve.py \
  --caption-root /path/to/caption_corpus \
  --experiment-root /path/to/video_rsi_experiments \
  --budget 5 --limit 10 --config-path /private/rsi_api_config.json
```

`evolve.py` records every source snapshot, candidate workspace, Codex result,
test result and decision under the supplied experiment root. It never deletes
a rejected candidate. The machine running this framework does not need a GPU:
sampled video frames are sent to the selected evaluation model over an
OpenAI-compatible endpoint. Select any compatible model using
`--evaluation-model MODEL_ID` (legacy alias: `--frozen-target-model`) or
`RSI_EVALUATION_MODEL`. On a CPU/no-GPU setup, a general API is `reference`;
do not label it `target` just because it returns vision judgments. Set
`--model-role target` only when the endpoint is backed by the exact model being
optimized for downstream training (same model/checkpoint), such as your own
GPU-hosted target service. Merely having a GPU is not sufficient either: if
the served model is not the training target, it remains a `reference`.
Configure the endpoint with `RSI_EVALUATION_MODEL_BASE_URL` and credential with
`RSI_EVALUATION_MODEL_API_KEY`. When omitted, the regular API endpoint and key
are reused. Keep model, role and endpoint fixed throughout a campaign.
`--local-only-api` is optional for users who already have a local compatible
service; it is not required on CPU-only hosts.

The framework-owned `VideoRSICandidateEvaluator` does not load weights. Its
optional `signal_provider` is an injected per-record callable, so deployments
may use a remote VLM, a CPU-served model, a GPU service on another host, or no
model judge. Its role defaults to `reference` when a provider is supplied; on a
no-GPU setup, keep it `reference`. Use `target` only for a service that really
serves the actual downstream model. Without an independent provider, model-dependent labels are not
independent visual verification; deterministic schema, provenance and pool
checks still run on CPU. A visual judge must be able to access the sampled
frames or a resolvable media reference.

## Repository boundary

Do commit code, tests, task skill packs, prompts, public configuration examples
and small documentation. Do not commit API keys, proxy credentials, user paths,
captions, frame caches, JSONL data pools, benchmark predictions, checkpoints or
process logs. The `.gitignore` rules enforce this boundary; inspect `git status`
before every commit.
