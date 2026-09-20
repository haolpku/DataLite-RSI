# VideoRSI on Qwen3-VL-8B-Instruct

Status: **unverified**. The 4,541-example VideoRSI SFT run
improved overall Video-MME accuracy from **56.56** to **58.89** (**+2.33**).

| Category | Base | VideoRSI |
| --- | ---: | ---: |
| Action Reasoning | 47.37 | 50.18 |
| Object Recognition | 61.02 | 62.99 |
| Counting | 37.31 | 40.30 |
| Information Synopsis | 74.30 | 76.16 |
| Object Reasoning | 53.96 | 56.61 |
| Temporal Perception | 61.82 | 63.64 |
| Attribute | 68.47 | 70.72 |
| Temporal Reasoning | 37.29 | 41.24 |
| Action Recognition | 56.87 | 58.47 |
| OCR | 61.87 | 63.31 |
| Spatial Perception | 68.52 | 70.37 |
| Spatial Reasoning | 67.86 | 69.64 |

## Setup

Base model: `Qwen/Qwen3-VL-8B-Instruct`; method: [VideoRSI](../../../rsi/methods/video-rsi/);
benchmark: [Video-MME SFT transfer](../../../benchmarks/videomme-sft-transfer/).
The paired scores use the same Video-MME taxonomy and are reported as percentages.
