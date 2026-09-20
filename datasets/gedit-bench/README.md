# GEdit-Bench

Real-user image-editing cases collected for
[Step1X-Edit](https://arxiv.org/abs/2504.17761).

**Source:** [`stepfun-ai/GEdit-Bench`](https://huggingface.co/datasets/stepfun-ai/GEdit-Bench)
at commit `50766778e2a737474c7e9bdf84cdce82c3ea3f4f`.

| | |
| --- | --- |
| Size | 606 unique cases; 1,212 rows with English and Chinese instructions |
| Categories | 11 genuine-edit types from internet user requests |
| Images | Real photographs, not synthetic canvases |
| License | MIT |
| Access | public |
| PII | Source images are de-identified user uploads; residual identity risk remains in photographs of people |

**Metric on this suite:** VLM-judge overall score on a 0–10 scale (VIEScore /
GPT-family overall, G_O). The DataLite-RSI evaluator does not call the judge; it
re-aggregates already-graded per-sample scores.

Do not commit the Arrow shards here. Pin the Hugging Face revision in any
result manifest.
