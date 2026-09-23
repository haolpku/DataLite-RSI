# Docker environments

Dockerfiles are version-controlled here; built images are published to GitHub
Container Registry (GHCR). Data and model weights are mounted or downloaded at
runtime and are never copied into images.

The `validator` image is a small working example:

```bash
docker build -f docker/validator/Dockerfile -t datalite-rsi-validator .
docker run --rm -v "$PWD:/workspace:ro" datalite-rsi-validator /workspace
```

Evaluation images should eventually be published using versioned names such as:

```text
ghcr.io/haolpku/datalite-rsi-hle-no-tools:0.1.0
ghcr.io/haolpku/datalite-rsi-multimodal:0.1.0
ghcr.io/haolpku/datalite-rsi-generative:0.1.0
```

Available environments:

| Directory | Track | Purpose |
| --- | --- | --- |
| [`validator/`](validator/) | — | Manifest validation; no dependencies |
| [`generative-image-edit/`](generative-image-edit/) | generative | Instruction-guided image editing: LoRA training on a diffusion backbone and benchmark scoring |

Verified result manifests should pin an image digest rather than `latest`.
Document required host drivers, minimum memory, supported architectures, mounted
paths, runtime secrets, and a smoke-test command beside every environment.
