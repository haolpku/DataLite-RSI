# ImgEdit-Bench

Evaluation split of
[ImgEdit](https://arxiv.org/abs/2505.20275) (PKU-Yuan Group).

**Source:** [`sysuyy/ImgEdit`](https://huggingface.co/datasets/sysuyy/ImgEdit)
at commit `f8de753484a2b6bd37f135fd20d308b66b09a523`. The Hub repo is the full
1.2M-pair corpus plus `Benchmark.tar`; only the benchmark split is registered
here.

| | |
| --- | --- |
| Size | 811 cases (734 basic plus challenging and multi-turn suites) |
| What it measures | Instruction adherence, edit quality, detail preservation |
| License | Apache-2.0 |
| Access | public |
| PII | Constructed from filtered web images; residual identity risk remains |

**Metric on this suite:** VLM-judge overall on a 0–5 scale. The DataLite-RSI
evaluator does not call the judge; it re-aggregates already-graded per-sample
scores.

Do not commit `Benchmark.tar` or the 1.2M training shards. Pin the Hugging Face
revision in any result manifest.

Code: [PKU-YuanGroup/ImgEdit](https://github.com/PKU-YuanGroup/ImgEdit).
