# VideoRSI on LLaVA-OneVision-7B

Status: **unverified**. The 4,541-example VideoRSI SFT run
improved overall Video-MME accuracy from **58.52** to **59.48** (**+0.96**).

| Category | Base | VideoRSI |
| --- | ---: | ---: |
| Action Reasoning | 54.39 | 56.14 |
| Object Recognition | 64.41 | 65.25 |
| Counting | 37.69 | 36.94 |
| Information Synopsis | 72.45 | 73.68 |
| Object Reasoning | 55.95 | 56.83 |
| Temporal Perception | 58.18 | 60.00 |
| Attribute | 74.32 | 74.77 |
| Temporal Reasoning | 42.37 | 43.50 |
| Action Recognition | 55.91 | 56.87 |
| OCR | 61.15 | 63.31 |
| Spatial Perception | 59.26 | 61.11 |
| Spatial Reasoning | 78.57 | 80.36 |

## Setup

Base model: `llava-hf/llava-onevision-qwen2-7b-ov-hf`; method:
[VideoRSI](../../../rsi/methods/video-rsi/); benchmark:
[Video-MME SFT transfer](../../../benchmarks/videomme-sft-transfer/). The paired
scores use the same Video-MME taxonomy and are reported as percentages.
