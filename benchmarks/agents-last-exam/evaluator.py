"""Aggregate official Agents' Last Exam run artifacts.

The official ALE runner writes one directory containing ``run.json`` and
``eval_result.json`` for each attempt. This adapter deliberately evaluates
those saved records only; it never imports the runner or contacts a model/API.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable


BENCHMARK_ID = "agents-last-exam"
BENCHMARK_VERSION = "0.1.0"
_FAILURE_STATUSES = {"failed", "timeout", "cancelled"}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON: {exc}") from exc


def _run_directories(predictions: Path) -> list[Path]:
    if predictions.is_file():
        if predictions.name not in {"eval_result.json", "run.json"}:
            raise ValueError(
                "predictions file must be an ALE eval_result.json or run.json"
            )
        return [predictions.parent]
    if not predictions.is_dir():
        raise FileNotFoundError(f"predictions path does not exist: {predictions}")

    candidates = [
        path.parent for path in predictions.rglob("eval_result.json") if path.is_file()
    ]
    if (predictions / "eval_result.json").is_file():
        candidates.append(predictions)
    return sorted(set(candidates), key=lambda path: path.as_posix())


def _number(value: Any, field: str, *, minimum: float | None = None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return result


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _evaluate_run(
    run_dir: Path, root: Path
) -> tuple[dict[str, Any] | None, str | None]:
    eval_path = run_dir / "eval_result.json"
    run_path = run_dir / "run.json"
    if not run_path.is_file():
        return None, "run.json is missing"

    try:
        evaluation = _read_json(eval_path)
        metadata = _read_json(run_path)
    except ValueError as exc:
        return None, str(exc)
    if not isinstance(evaluation, dict) or not isinstance(metadata, dict):
        return None, "run.json and eval_result.json must contain JSON objects"

    eval_status = evaluation.get("eval_status")
    run_status = metadata.get("status")
    try:
        score = _number(evaluation.get("score"), "score", minimum=0.0)
    except ValueError as exc:
        return None, str(exc)
    if score is not None and score > 1.0:
        return None, "score must be between 0 and 1"
    if score is None:
        if eval_status in _FAILURE_STATUSES or run_status in _FAILURE_STATUSES:
            score = 0.0
        else:
            return None, "score is missing for a non-failed run"

    task = metadata.get("task")
    agent = metadata.get("agent")
    timings = metadata.get("timings")
    usage = metadata.get("usage")
    if not isinstance(task, dict) or not isinstance(agent, dict):
        return None, "run.json must contain task and agent objects"
    task_id = task.get("path") or task.get("slug")
    if not isinstance(task_id, str) or not task_id.strip():
        return None, "run.json task must contain a non-empty path or slug"
    if timings is not None and not isinstance(timings, dict):
        return None, "run.json timings must be an object"
    if usage is not None and not isinstance(usage, dict):
        return None, "run.json usage must be an object"

    try:
        duration = _number(
            (timings or {}).get("duration_s"), "timings.duration_s", minimum=0.0
        )
        cost = _number(
            (usage or {}).get("total_cost_usd"), "usage.total_cost_usd", minimum=0.0
        )
    except ValueError as exc:
        return None, str(exc)
    record = {
        "run_id": metadata.get("run_id"),
        "task_id": task_id,
        "agent_id": agent.get("id") or agent.get("class"),
        "model_id": agent.get("model"),
        "status": run_status,
        "eval_status": eval_status,
        "score": score,
        "cost_usd": cost,
        "duration_s": duration,
        "artifact_dir": _relative(run_dir, root),
    }
    return record, None


def evaluate_predictions(predictions: str | Path) -> dict[str, Any]:
    """Return deterministic aggregate metrics for an ALE output directory."""
    root = Path(predictions).expanduser().resolve()
    run_dirs = _run_directories(root)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for run_dir in run_dirs:
        record, error = _evaluate_run(run_dir, root if root.is_dir() else run_dir)
        if error:
            errors.append({"artifact_dir": _relative(run_dir, root), "error": error})
        elif record is not None:
            records.append(record)

    if not run_dirs:
        raise ValueError(f"no eval_result.json files found under {root}")

    scores = [record["score"] for record in records]
    costs = [record["cost_usd"] for record in records if record["cost_usd"] is not None]
    durations = [
        record["duration_s"] for record in records if record["duration_s"] is not None
    ]
    completed = sum(record["status"] == "completed" for record in records)
    passed = sum(record["score"] >= 1.0 for record in records)
    metrics = {
        "primary_score": sum(scores) / len(scores) if scores else 0.0,
        "pass_rate": passed / len(scores) if scores else 0.0,
        "mean_cost_usd": sum(costs) / len(costs) if costs else None,
        "mean_duration_s": sum(durations) / len(durations) if durations else None,
    }
    return {
        "schema_version": "0.1",
        "benchmark": {"id": BENCHMARK_ID, "version": BENCHMARK_VERSION},
        "source": {"predictions": str(root), "format": "ALE RunWriter v2"},
        "metrics": metrics,
        "counts": {
            "runs": len(run_dirs),
            "scored_runs": len(records),
            "successful_runs": sum(
                record["eval_status"] == "success" for record in records
            ),
            "completed_runs": completed,
            "failed_runs": len(records) - completed,
            "invalid_runs": len(errors),
            "tasks": len(
                {record["task_id"] for record in records if record["task_id"]}
            ),
        },
        "runs": records,
        "errors": errors,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "predictions", type=Path, help="ALE output root or run directory"
    )
    parser.add_argument("--output", type=Path, help="write JSON metrics to this path")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = evaluate_predictions(args.predictions)
    except (OSError, ValueError) as exc:
        print(f"agentlastexam evaluator: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
