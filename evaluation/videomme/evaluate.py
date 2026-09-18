"""Resume-safe Video-MME evaluation through an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import base64
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from .protocol import answer_letter, completed_ids, prompt_for, summarize
except ImportError:
    from protocol import answer_letter, completed_ids, prompt_for, summarize


def load_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def predict_one(row: dict[str, Any], video_dir: Path, endpoint: str, model: str, timeout: int) -> dict[str, Any]:
    import requests

    video_path = video_dir / f"{row['videoID']}.mp4"
    base = {"question_id": row["question_id"], "video_id": row["videoID"], "answer": row.get("answer")}
    if not video_path.is_file():
        return {**base, "error": f"missing video: {video_path}"}
    try:
        encoded = base64.b64encode(video_path.read_bytes()).decode("ascii")
        content = [
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{encoded}"}},
            {"type": "text", "text": prompt_for(row)},
        ]
        response = requests.post(
            endpoint.rstrip("/") + "/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": 8},
            timeout=timeout,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        return {
            **base,
            "prediction": answer_letter(raw),
            "raw": raw,
            "task_type": row.get("task_type"),
            "domain": row.get("domain"),
            "sub_category": row.get("sub_category"),
            "error": None,
        }
    except Exception as exc:
        return {**base, "error": repr(exc)}


def read_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = load_rows(args.parquet)
    if args.limit:
        rows = rows[:args.limit]
    prior = read_predictions(args.output)
    done = completed_ids(prior)
    pending = [row for row in rows if str(row["question_id"]) not in done]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as writer, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(predict_one, row, args.video_dir, args.endpoint, args.model, args.timeout) for row in pending]
        for future in as_completed(futures):
            writer.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
            writer.flush()

    summary = summarize(read_predictions(args.output))
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
