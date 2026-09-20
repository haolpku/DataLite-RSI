# VideoRSI on Qwen2.5-VL-3B-Instruct

Status: **unverified**. The 4,541-example VideoRSI SFT run
improved overall Video-MME accuracy from **42.44** to **59.04** (**+16.60**).

| Category | Base | VideoRSI |
| --- | ---: | ---: |
| Action Reasoning | 41.40 | 50.18 |
| Object Recognition | 45.76 | 66.10 |
| Counting | 25.37 | 38.06 |
| Information Synopsis | 58.82 | 75.54 |
| Object Reasoning | 37.67 | 55.95 |
| Temporal Perception | 43.64 | 56.36 |
| Attribute | 50.45 | 74.32 |
| Temporal Reasoning | 32.20 | 37.85 |
| Action Recognition | 41.85 | 58.79 |
| OCR | 41.01 | 66.91 |
| Spatial Perception | 40.74 | 59.26 |
| Spatial Reasoning | 60.71 | 80.36 |

## Setup

Base model: `Qwen/Qwen2.5-VL-3B-Instruct`; method:
[VideoRSI](../../../rsi/methods/video-rsi/); benchmark:
[Video-MME SFT transfer](../../../benchmarks/videomme-sft-transfer/). The paired
scores use the same Video-MME taxonomy and are reported as percentages.
