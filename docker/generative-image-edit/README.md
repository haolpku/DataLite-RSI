# Generative track: instruction-guided image editing

Environment for LoRA-training a diffusion editing backbone on synthesised edit
pairs and scoring it on image-editing benchmarks. This is the environment the
`policy-evolving-edit-synthesis` results were produced in.

## What is pinned

| Component | Version |
| --- | --- |
| Base image | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`, by digest |
| Python | 3.11 |
| torch / torchvision | 2.6.0+cu124 / 0.21.0+cu124 (from the base image) |
| diffusers | 0.39.0 |
| transformers | 5.12.1 |
| accelerate | 1.14.0 |
| peft | 0.19.1 |
| safetensors | 0.8.0 |
| huggingface-hub | 1.21.0 |

## Build and smoke test

```bash
docker build -f docker/generative-image-edit/Dockerfile \
  -t ghcr.io/<owner>/datalite-rsi-generative:0.1.0 \
  docker/generative-image-edit

# Reports the stack it resolved; needs no GPU and no weights.
docker run --rm ghcr.io/<owner>/datalite-rsi-generative:0.1.0
```

On a GPU host, check that the runtime actually sees the devices:

```bash
docker run --rm --gpus all ghcr.io/<owner>/datalite-rsi-generative:0.1.0
```

A verified result should pin the image by `sha256` digest rather than by tag.

## Requirements

- **GPU.** NVIDIA with driver supporting CUDA 12.4. The reference runs used
  8x H20; a 9B backbone needs roughly 29 GB of VRAM for inference and more for
  LoRA training, so a single 24 GB card is not enough at default settings.
- **Host RAM.** 64 GB or more when sharding a 20B backbone across ranks.
- **Architecture.** `linux/amd64` only.
- **Disk.** Budget for the weights in the mounted cache; a 20B BF16 backbone is
  about 40 GB before any benchmark data.

## Mounted paths

| Path | Purpose |
| --- | --- |
| `/data/hf` | Hugging Face cache: backbone weights and benchmark datasets. `HF_HOME` already points here. |
| `/workspace` | The repository or method checkout. |
| `/outputs` | LoRA checkpoints, generated images, score files. |

Model weights and benchmark data are **not** baked into the image. Download or
mount them at runtime and pin their revisions in the result manifest, per
`CONTRIBUTING.md`.

```bash
docker run --rm --gpus all \
  -v "$PWD:/workspace" \
  -v /path/to/hf-cache:/data/hf \
  -v /path/to/outputs:/outputs \
  ghcr.io/<owner>/datalite-rsi-generative:0.1.0 \
  python your_eval_entrypoint.py
```

## Runtime secrets

Two kinds of credential may be needed, and neither belongs in the image or in a
config file — pass them as environment variables at `docker run` time:

- `HF_TOKEN`, to download gated weights. FLUX.2-klein-base-9B is gated and
  carries a **non-commercial** licence, so access must be accepted on Hugging
  Face first.
- An API key for the judge endpoint, if scoring uses a hosted VLM. The
  image-editing benchmarks here are graded by a VLM judge rather than by an
  offline metric.

The image runs as UID/GID `65532:65532`. Mounted directories must be writable by
that user, or pass `--user "$(id -u):$(id -g)"` to match the host.
