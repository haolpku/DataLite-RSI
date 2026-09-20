# Policy-Evolving Edit Synthesis on Qwen-Image-Edit-2511

Status: **unverified**. Numbers transcribed from the method README in
[PR #4](https://github.com/haolpku/DataLite-RSI/pull/4).

LoRA-trains [`Qwen/Qwen-Image-Edit-2511`](https://huggingface.co/Qwen) on
1,000 accepted pairs from
[Policy-Evolving Edit Synthesis](../../../rsi/methods/policy-evolving-edit-synthesis/),
then scores
[GEdit-Bench](../../../datasets/gedit-bench/) (0–10) and
[ImgEdit-Bench](../../../datasets/imgedit-bench/) (0–5).

`primary_score = 0.5 * (gedit_bench / 10 + imgedit_bench / 5)`.

| Training data | GEdit-Bench | ImgEdit-Bench | primary |
| --- | ---: | ---: | ---: |
| none (untrained backbone) | 8.104 | 4.449 | 0.8501 |
| stratified pipeline, 1,000 pairs | 8.138 | 4.578 | 0.8647 |
| **policy-evolving, 1,000 pairs** | **8.171** | **4.598** | **0.86835** |

The policy-evolving arm is +0.01825 primary over the untrained backbone and
+0.00365 over the matched stratified control.

In-loop synthesis (shared across backbones, not the headline): acceptance rate
79.2% → 87.8%; destination entropy 0.853 → 0.894.

Per-sample judge files are not in this repository. `artifacts_url` is null
until they are uploaded.
