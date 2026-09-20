# Image-edit transfer suite

Measures how much a small synthesised image-editing dataset improves a frozen
diffusion editing backbone. The unit under test is **the dataset, not the
backbone**: base model, LoRA recipe, and judge settings stay fixed, so the
only variable is which accepted edit pairs go into training.

## Benchmarks

| Benchmark | Registry | Scale | Judge |
| --- | --- | ---: | --- |
| `gedit_bench` | [`gedit-bench`](../../datasets/gedit-bench/) | 0–10 | VLM / VIEScore overall |
| `imgedit_bench` | [`imgedit-bench`](../../datasets/imgedit-bench/) | 0–5 | VLM judge |

`primary_score` is the unweighted mean after mapping each bench onto `[0, 1]`:

```text
primary_score = 0.5 * (gedit_bench / 10 + imgedit_bench / 5)
```

Read the two raw columns. The scales differ, and a 0.3-point GEdit move is not
the same as a 0.3-point ImgEdit move.

## Reference protocol v0.1.0

Downstream training and scoring run in
[`docker/generative-image-edit/`](../../docker/generative-image-edit/).

| Setting | Value |
| --- | --- |
| Training | LoRA on a frozen image-edit backbone, identical recipe across arms |
| Backbones in the reference method | `Qwen/Qwen-Image-Edit-2511`, `black-forest-labs/FLUX.2-klein-base-9B` |
| Accepted-pair budget | 1,000 pairs, 23 edit types |
| Judge | Hosted VLM; scores are **not** an offline pixel metric |
| Hardware (reference) | 8x H20 |

The VLM judge is not vendored. A published result must ship the per-sample
score file so aggregation can be checked without the judge API.

## Offline re-score

```bash
python benchmarks/image-edit-transfer-suite/evaluator.py \
  --predictions predictions.jsonl --output metrics.json
```

Input: one JSONL record per judged sample:

```text
{"benchmark": "gedit_bench", "sample_id": "g0", "score": 8.2}
{"benchmark": "imgedit_bench", "sample_id": "i0", "score": 4.5}
```

`score` is the already-graded judge value on that benchmark's native scale.
Null or out-of-range scores are counted as invalid and excluded from the mean,
but they are reported.

```bash
python -m unittest discover -s benchmarks/image-edit-transfer-suite/tests -v
```

## Contamination and leakage

- The synthesised training pairs must not copy GEdit-Bench or ImgEdit-Bench
  source images or instructions.
- In-loop acceptance uses a **different** three-axis rubric (`if_vc_vq`) on
  generated pairs, not these two test sets. Still, a method that peeks at
  GEdit/ImgEdit items while writing policies would leak; submissions should
  say the test sets were held out of the policy loop.
- Residual risk: a VLM judge may itself have seen these benches.

## Data

Large files stay on Hugging Face. GitHub only keeps the registry entries.
