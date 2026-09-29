"""Command line entry point for the Video RSI bootstrap pipeline."""

from __future__ import annotations

import argparse
import json
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .corpus import iter_caption_corpus
from .core import RunContext
from .model_client import VLLMTextJSONClient
from .pipelines import build_bootstrap_pipeline, build_semantic_fusion_pipeline
from .storage import RunStore, utc_now


def _default_run_id() -> str:
    return "round0_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _run_one(record, pipeline, context):
    return pipeline.run_record(record.state, context)


def _run_pipeline(args: argparse.Namespace, pipeline) -> None:
    caption_root = args.caption_root.resolve()
    output_root = args.output_root.resolve()
    pipeline.compile(["video_key", "video", "source_result", "observations"])
    run_id = args.run_id or _default_run_id()
    store = RunStore(output_root, run_id)
    manifest = {
        "schema_version": "video-rsi-run-v1",
        "run_id": run_id,
        "created_at": utc_now(),
        "caption_root": str(caption_root),
        "output_root": str(output_root),
        "limit": args.limit,
        "pipeline": pipeline.description(),
    }
    store.write_manifest(manifest)
    records = list(iter_caption_corpus(caption_root, limit=args.limit))
    stats = Counter(total=len(records))
    event_count = 0
    opportunity_count = 0
    opportunity_by_task: Counter[str] = Counter()

    context = RunContext(run_id=run_id, store=store, resume=not args.no_resume)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run_one, record, pipeline, context): record
            for record in records
        }
        for future in as_completed(futures):
            record = futures[future]
            try:
                result = future.result()
                stats["completed"] += 1
                event_count += len(result["semantic_events"])
                opportunity_count += len(result["task_opportunities"])
                opportunity_by_task.update(
                    item["task_type"] for item in result["task_opportunities"]
                )
            except Exception as exc:
                stats["failed"] += 1
                store.append_error(
                    {
                        "timestamp": utc_now(),
                        "video_key": record.video_key,
                        "source_result": str(record.result_path),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

    summary: dict[str, Any] = {
        "schema_version": "video-rsi-run-summary-v1",
        "run_id": run_id,
        "pipeline": pipeline.name,
        "videos_total": stats["total"],
        "videos_completed": stats["completed"],
        "videos_failed": stats["failed"],
        "semantic_event_count": event_count,
        "task_opportunity_count": opportunity_count,
        "task_opportunity_by_type": dict(sorted(opportunity_by_task.items())),
        "finished_at": utc_now(),
    }
    store.write_summary(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_bootstrap(args: argparse.Namespace) -> None:
    _run_pipeline(args, build_bootstrap_pipeline())


def run_semantic_fusion(args: argparse.Namespace) -> None:
    client = VLLMTextJSONClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=0.0,
        seed=args.seed,
        timeout_sec=args.timeout_sec,
    )
    _run_pipeline(
        args,
        build_semantic_fusion_pipeline(client, max_tokens=args.max_tokens),
    )


def inspect_corpus(args: argparse.Namespace) -> None:
    counts = Counter()
    duration = 0.0
    for record in iter_caption_corpus(args.caption_root.resolve(), limit=args.limit):
        counts["videos"] += 1
        counts["segments"] += len(record.state["observations"])
        duration += float(record.state["video"]["duration_sec"])
    print(
        json.dumps(
            {
                "videos": counts["videos"],
                "segments": counts["segments"],
                "duration_hours": round(duration / 3600.0, 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video Data RSI pipelines")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-caption-corpus")
    inspect_parser.add_argument("--caption-root", type=Path, required=True)
    inspect_parser.add_argument("--limit", type=int)
    inspect_parser.set_defaults(func=inspect_corpus)

    run_parser = subparsers.add_parser("run-bootstrap")
    run_parser.add_argument("--caption-root", type=Path, required=True)
    run_parser.add_argument("--output-root", type=Path, required=True)
    run_parser.add_argument("--run-id", type=str)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--workers", type=int, default=4)
    run_parser.add_argument("--no-resume", action="store_true")
    run_parser.set_defaults(func=run_bootstrap)

    fusion_parser = subparsers.add_parser("run-semantic-fusion")
    fusion_parser.add_argument("--caption-root", type=Path, required=True)
    fusion_parser.add_argument("--output-root", type=Path, required=True)
    fusion_parser.add_argument("--run-id", type=str)
    fusion_parser.add_argument("--limit", type=int)
    fusion_parser.add_argument("--workers", type=int, default=4)
    fusion_parser.add_argument("--no-resume", action="store_true")
    fusion_parser.add_argument("--base-url", type=str, required=True)
    fusion_parser.add_argument("--model", type=str, required=True)
    fusion_parser.add_argument("--api-key", type=str, default="EMPTY")
    fusion_parser.add_argument("--max-tokens", type=int, default=5000)
    fusion_parser.add_argument("--seed", type=int, default=0)
    fusion_parser.add_argument("--timeout-sec", type=float, default=600.0)
    fusion_parser.set_defaults(func=run_semantic_fusion)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(args, "workers", 1) < 1:
        raise ValueError("workers must be at least 1")
    args.func(args)


if __name__ == "__main__":
    main()
