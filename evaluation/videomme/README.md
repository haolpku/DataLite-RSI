# Video-MME Evaluation

```bash
python evaluation/videomme/evaluate.py \
  --parquet videomme/test-00000-of-00001.parquet \
  --video-dir data \
  --output predictions.jsonl \
  --endpoint "$VIDEO_RSI_ENDPOINT" \
  --model "$VIDEO_RSI_MODEL"
```

Dependencies: `pyarrow` and `requests`.
