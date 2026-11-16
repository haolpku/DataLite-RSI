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
a rejected candidate. `--local-only-api` is available for a fixed local
vLLM-only probe and must be used when remote API cost should be avoided.

## Repository boundary

Do commit code, tests, task skill packs, prompts, public configuration examples
and small documentation. Do not commit API keys, proxy credentials, user paths,
captions, frame caches, JSONL data pools, benchmark predictions, checkpoints or
process logs. The `.gitignore` rules enforce this boundary; inspect `git status`
before every commit.
