"""Run and preserve one version of the task-centric RSI experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
# This driver lives in the repository, so the default source is always the
# sibling package directory.  External experiment roots can still override it
# through RSI_SOURCE_ROOT or --source-root.
DEFAULT_SOURCE = HERE.parent / "src"
ACTIVE_SOURCE = Path(os.environ.get("RSI_SOURCE_ROOT", str(DEFAULT_SOURCE))).resolve()
if str(ACTIVE_SOURCE) not in sys.path:
    sys.path.insert(0, str(ACTIVE_SOURCE))

from video_rsi.api_client import OpenAICompatibleJSONClient, load_api_config  # noqa: E402
from video_rsi.core import RunContext  # noqa: E402
from video_rsi.corpus import iter_caption_corpus  # noqa: E402
from video_rsi.pipelines import build_task_centric_rsi_pipeline  # noqa: E402
from video_rsi.storage import RunStore, atomic_write_json, utc_now  # noqa: E402

from api_backends import APIQuestionGenerator, APIDistractorEnhancer, APITextOnly, APIFrozenTarget  # noqa: E402
from evaluation import EvaluationPolicy  # noqa: E402


def _norm(text: object) -> str:
    return " ".join(str(text).casefold().split())


def snapshot_source(source_root: Path, destination: Path) -> None:
    if destination.exists():
        return
    shutil.copytree(
        source_root / "video_rsi", destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def _build_backends(
    config_path: Path | None,
    frozen_target_model: str,
    allow_text_frozen_fallback: bool,
    local_only_api: bool = False,
):
    if local_only_api:
        base_url, api_key = "", ""
    else:
        base_url, api_key = load_api_config(config_path)
    # Keep a bounded per-request timeout so one unhealthy API connection does
    # not stall the whole evolution run indefinitely.  The value is overridable
    # for controlled experiments, while the default is short enough for the
    # watchdog to recover and long enough for normal provider latency.
    api_timeout_sec = float(os.environ.get("RSI_API_TIMEOUT_SEC", "300"))
    if local_only_api:
        # Evolution probes must not spend remote chat-API budget.  Reuse the
        # local OpenAI-compatible vLLM endpoint for every model-backed stage;
        # the endpoint/model are explicit so a missing local service fails
        # loudly instead of silently falling back to the remote provider.
        probe_base = os.environ.get("RSI_LOCAL_PROBE_BASE_URL") or os.environ.get(
            "RSI_FROZEN_TARGET_BASE_URL"
        )
        if not probe_base:
            raise RuntimeError(
                "--local-only-api requires RSI_LOCAL_PROBE_BASE_URL or "
                "RSI_FROZEN_TARGET_BASE_URL"
            )
        probe_key = os.environ.get("RSI_LOCAL_PROBE_API_KEY", "EMPTY")
        probe_model = os.environ.get("RSI_LOCAL_PROBE_MODEL", frozen_target_model)
    else:
        probe_base, probe_key, probe_model = base_url, api_key, ""

    def client(model: str):
        return OpenAICompatibleJSONClient(
            model=probe_model or model,
            base_url=probe_base,
            api_key=probe_key,
            timeout_sec=api_timeout_sec,
        )
    # The frozen vision target may be served locally (for example, Qwen3-VL on
    # the experiment GPU), while question/verifier/text operators use the
    # external API.  Keep these endpoints independent so the real RSI loop can
    # evaluate against the intended frozen model instead of silently calling the
    # external provider for every backend.
    frozen_base = probe_base if local_only_api else os.environ.get("RSI_FROZEN_TARGET_BASE_URL", base_url)
    frozen_key = probe_key if local_only_api else os.environ.get("RSI_FROZEN_TARGET_API_KEY", "EMPTY")
    frozen_client = OpenAICompatibleJSONClient(
        model=probe_model or frozen_target_model,
        base_url=frozen_base, api_key=frozen_key,
        timeout_sec=api_timeout_sec,
    )
    generator = APIQuestionGenerator(client("gpt-5.5"))
    # Distractor rewriting is a constrained text transformation rather than
    # open-ended question synthesis; use the cheaper GLM model while keeping
    # the question generator on GPT-5.5.
    distractor = APIDistractorEnhancer(client("glm-5.3-flash"))
    text_only = APITextOnly(client("glm-5.3-flash"))
    frozen = APIFrozenTarget(
        frozen_client, frozen_target_model, allow_text_frozen_fallback
    )
    if local_only_api:
        # Make lineage/manifests truthful: these are local endpoint calls even
        # though the adapter classes retain their historical role names.
        generator.name = f"local:{probe_model}:question-generation"
        distractor.name = f"local:{probe_model}:distractor-enhancement"
        text_only.name = f"local:{probe_model}:text-only"
        frozen.name = f"local:{probe_model}"
    return (generator, distractor, text_only, frozen)


def run(args: argparse.Namespace) -> dict:
    root = args.experiment_root.resolve()
    requested_source = args.source_root.resolve()
    if ACTIVE_SOURCE != requested_source:
        raise RuntimeError(
            "runtime source mismatch: "
            f"imported={ACTIVE_SOURCE} requested={requested_source}. "
            "Run through main() so it can re-exec with RSI_SOURCE_ROOT pinned."
        )
    version_root = root / "versions" / args.version
    source_snapshot = version_root / "source"
    output_root = version_root / "outputs"
    snapshot_source(requested_source, source_snapshot)
    version_root.mkdir(parents=True, exist_ok=True)
    evaluation_policy = EvaluationPolicy(
        frozen_target_model=args.frozen_target_model,
        require_no_text_fallback=not args.allow_text_frozen_fallback,
    )
    (version_root / "manifest.json").write_text(
        json.dumps({"version": args.version, "created_at": utc_now(),
                    "source_snapshot": str(source_snapshot),
                    "evaluation_policy": evaluation_policy.as_dict(),
                    "runtime": {"ffmpeg_bin": os.environ.get("RSI_FFMPEG_BIN", "ffmpeg"),
                                "source_root": str(ACTIVE_SOURCE)}},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    generator, distractor, text_only, frozen = _build_backends(
        args.config_path,
        args.frozen_target_model,
        args.allow_text_frozen_fallback,
        args.local_only_api,
    )
    pipeline = build_task_centric_rsi_pipeline(
        generator_backend=generator, text_only_backend=text_only, frozen_target_backend=frozen,
        distractor_backend=distractor,
        max_per_task=args.max_per_task, frontier_trials=args.frontier_trials,
    )
    records = list(iter_caption_corpus(args.caption_root.resolve(), limit=args.limit))
    pipeline.compile(records[0].state.keys()) if records else None
    run_id = args.run_id or f"{args.version}_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}"
    store = RunStore(output_root, run_id)
    store.write_manifest({"schema_version": "video-rsi-experiment-v1", "run_id": run_id,
                          "version": args.version, "created_at": utc_now(),
                          "pipeline": pipeline.description(), "videos_total": len(records),
                          "question_limit": args.question_limit,
                          "evaluation_policy": evaluation_policy.as_dict(),
                          "local_only_api": bool(args.local_only_api),
                          "runtime": {"ffmpeg_bin": os.environ.get("RSI_FFMPEG_BIN", "ffmpeg"),
                                      "source_root": str(ACTIVE_SOURCE)}})
    pool_path = output_root / run_id / "high_quality_pool.jsonl"
    progress_path = output_root / run_id / "progress.json"
    started_monotonic = time.monotonic()
    seen: set[tuple[str, str]] = set()
    stats = Counter(total=len(records), completed=0, failed=0)
    feedback = Counter()

    def write_progress(*, last_video: str | None = None, status: str = "running") -> None:
        """Write a live, machine-readable heartbeat for long batch runs."""
        atomic_write_json(progress_path, {
            "schema_version": "video-rsi-progress-v1",
            "run_id": run_id,
            "version": args.version,
            "status": status,
            "videos_total": stats["total"],
            "videos_completed": stats["completed"],
            "videos_failed": stats["failed"],
            "videos_remaining": max(0, stats["total"] - stats["completed"] - stats["failed"]),
            "pool_rows_so_far": len(seen),
            "last_completed_video": last_video,
            "elapsed_sec": round(time.monotonic() - started_monotonic, 1),
            "updated_at": utc_now(),
        })

    write_progress()
    def process(record):
        return pipeline.run_record(record.state, RunContext(run_id, store, resume=not args.no_resume))

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process, record): record for record in records}
        for future in as_completed(futures):
            record = futures[future]
            try:
                results.append(future.result())
                stats["completed"] += 1
                write_progress(last_video=record.video_key)
            except Exception as exc:  # preserve failures and continue the corpus
                stats["failed"] += 1
                store.append_error({"timestamp": utc_now(), "video_key": record.video_key,
                                    "error_type": type(exc).__name__, "error": str(exc)})
                write_progress(last_video=record.video_key)

    with pool_path.open("w", encoding="utf-8") as pool:
        # Stable ordering makes a bounded probe comparable across generations
        # even though videos are processed concurrently.
        results.sort(key=lambda item: str(item.get("video", {}).get("video_id", "")))
        limit_reached = False
        for result in results:
            feedback.update(result.get("feedback_signals", {}))
            for sample in result.get("pool_candidates", []):
                if args.question_limit is not None and len(seen) >= args.question_limit:
                    limit_reached = True
                    break
                key = (_norm(sample.get("question")), _norm(sample.get("answer")))
                if key in seen:
                    feedback["rejected_global_duplicate"] += 1
                    continue
                seen.add(key)
                pool.write(json.dumps(sample, ensure_ascii=False) + "\n")
            if limit_reached:
                break
    summary = {"schema_version": "video-rsi-experiment-summary-v1", "run_id": run_id,
               "version": args.version, **dict(stats),
               "accepted_frontier_novel_count": len(seen),
               "feedback_signals": dict(sorted(feedback.items())),
               "finished_at": utc_now()}
    store.write_summary(summary)
    write_progress(status="complete")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, default=HERE)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--version", default="v0")
    parser.add_argument("--run-id")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--question-limit", type=int,
                        help="Keep at most this many deterministically ordered questions in the probe pool.")
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("RSI_WORKERS", "4")))
    parser.add_argument("--max-per-task", type=int, default=1,
                        help="Maximum one candidate per task type and video by default.")
    parser.add_argument("--frontier-trials", type=int, default=4)
    parser.add_argument("--frozen-target-model", default="qwen3-vl-8b-instruct",
                        help="Production default is qwen3-vl-8b-instruct; use an explicitly marked API substitute only for smoke tests.")
    parser.add_argument("--allow-text-frozen-fallback", action="store_true",
                        help="Smoke-only fallback when server has no ffmpeg: sends evidence text, never use for quality evaluation.")
    parser.add_argument(
        "--local-only-api", action="store_true",
        help="Use the local OpenAI-compatible endpoint for every model-backed stage; never call the remote chat API.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--config-path", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    # ``video_rsi`` is imported at module load time, while ``--source-root``
    # is parsed only here.  Without this hand-off, a caller could snapshot a
    # candidate source tree into the experiment directory but actually execute
    # the default tree already imported by this process.  Re-exec once with the
    # requested source explicitly pinned, so manifest, snapshot and runtime
    # always refer to the same versioned pipeline.
    requested_source = args.source_root.resolve()
    if requested_source != ACTIVE_SOURCE:
        if not (requested_source / "video_rsi").is_dir():
            parser.error(f"--source-root must contain video_rsi/: {requested_source}")
        os.environ["RSI_SOURCE_ROOT"] = str(requested_source)
        os.execv(
            sys.executable,
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        )
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
