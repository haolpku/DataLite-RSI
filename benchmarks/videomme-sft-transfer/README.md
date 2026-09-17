# Video-MME SFT Transfer

Video-MME multiple-choice evaluation for VideoRSI SFT results.

Prediction JSONL fields:

```json
{"category":"Action Recognition","prediction":"A","answer":"A"}
```

Run:

```bash
python benchmarks/videomme-sft-transfer/evaluator.py predictions.jsonl --output metrics.json
```
