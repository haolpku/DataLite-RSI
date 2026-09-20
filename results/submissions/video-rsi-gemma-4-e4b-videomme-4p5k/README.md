# VideoRSI on Gemma-4-E4B

Status: **unverified**. The 4,541-example VideoRSI SFT run
improved overall Video-MME accuracy from **47.74** to **48.37** (**+0.63**).

| Category | Base | VideoRSI |
| --- | ---: | ---: |
| Action Reasoning | 37.89 | 38.95 |
| Object Recognition | 48.59 | 49.44 |
| Counting | 34.70 | 33.96 |
| Information Synopsis | 65.02 | 65.94 |
| Object Reasoning | 46.92 | 46.92 |
| Temporal Perception | 47.27 | 47.27 |
| Attribute | 54.05 | 56.76 |
| Temporal Reasoning | 40.11 | 40.11 |
| Action Recognition | 46.33 | 45.05 |
| OCR | 49.64 | 53.96 |
| Spatial Perception | 51.85 | 50.00 |
| Spatial Reasoning | 60.71 | 66.07 |

## Setup

Base model: `local/gemma-4-E4B-it`; method: [VideoRSI](../../../rsi/methods/video-rsi/);
benchmark: [Video-MME SFT transfer](../../../benchmarks/videomme-sft-transfer/).
The paired scores use the same Video-MME taxonomy and are reported as percentages.
