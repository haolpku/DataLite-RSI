# VideoRSI on InternVL3-2B-hf

Status: **unverified**. The 4,541-example VideoRSI SFT run
improved overall Video-MME accuracy from **34.85** to **55.22** (**+20.37**).

| Category | Base | VideoRSI |
| --- | ---: | ---: |
| Action Reasoning | 36.49 | 49.47 |
| Object Recognition | 28.81 | 61.30 |
| Counting | 33.21 | 32.09 |
| Information Synopsis | 42.72 | 71.21 |
| Object Reasoning | 33.92 | 50.22 |
| Temporal Perception | 45.45 | 63.64 |
| Attribute | 33.78 | 68.92 |
| Temporal Reasoning | 32.20 | 32.77 |
| Action Recognition | 35.46 | 56.23 |
| OCR | 33.81 | 61.87 |
| Spatial Perception | 33.33 | 68.52 |
| Spatial Reasoning | 37.50 | 78.57 |

## Setup

Base model: `OpenGVLab/InternVL3-2B`; method: [VideoRSI](../../../rsi/methods/video-rsi/);
benchmark: [Video-MME SFT transfer](../../../benchmarks/videomme-sft-transfer/).
The paired scores use the same Video-MME taxonomy and are reported as percentages.
