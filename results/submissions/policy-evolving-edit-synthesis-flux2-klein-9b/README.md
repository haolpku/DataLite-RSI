# Policy-Evolving Edit Synthesis on FLUX.2-klein-base-9B

Status: **unverified**. Numbers transcribed from the method README in
[PR #4](https://github.com/haolpku/DataLite-RSI/pull/4).

LoRA-trains `black-forest-labs/FLUX.2-klein-base-9B` (gated, non-commercial)
on 1,000 accepted pairs from
[Policy-Evolving Edit Synthesis](../../../rsi/methods/policy-evolving-edit-synthesis/),
then scores
[GEdit-Bench](../../../datasets/gedit-bench/) (0–10) and
[ImgEdit-Bench](../../../datasets/imgedit-bench/) (0–5).

`primary_score = 0.5 * (gedit_bench / 10 + imgedit_bench / 5)`.

| Training data | GEdit-Bench | ImgEdit-Bench | primary |
| --- | ---: | ---: | ---: |
| none (untrained backbone) | 7.653 | 4.111 | 0.79375 |
| stratified pipeline, 1,000 pairs | 7.844 | 4.188 | 0.811 |
| **policy-evolving, 1,000 pairs** | **7.920** | **4.280** | **0.824** |

The policy-evolving arm is +0.03025 primary over the untrained backbone and
+0.013 over the matched stratified control.

In-loop synthesis (shared across backbones, not the headline): acceptance rate
79.2% → 87.8%; destination entropy 0.853 → 0.894.

Per-sample judge files are not in this repository. `artifacts_url` is null
until they are uploaded.
