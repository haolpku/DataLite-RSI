# HLE With Tools Agent Sandbox

This directory documents the prebuilt agent sandbox used by the HLE with-tools
benchmark. The image is not rebuilt from this repository. Pull the published
version directly:

```text
ghcr.io/haolpku/lite-rsi-eval/hle-with-tools:0.1.0
```

The image originates from the local tag `hle-eval-lite:tool-budget-120` and
keeps its existing `hle-eval` entrypoint and default command:

```text
hle-eval --config /config/eval.json
```

Runtime configuration is mounted at `/config/eval.json`, optional data at
`/data`, and JSONL results at `/output`. Provider credentials are injected as
environment variables, never baked into the image.

Publish a new version by retagging the existing image without rebuilding it:

```bash
docker tag hle-eval-lite:tool-budget-120 \
  ghcr.io/haolpku/lite-rsi-eval/hle-with-tools:<version>
docker push ghcr.io/haolpku/lite-rsi-eval/hle-with-tools:<version>
```

Record the resulting immutable registry digest with every reproducible result
and replace `registry_digest` in `image.json`. The local image ID is recorded
separately because it is not a registry manifest digest.
