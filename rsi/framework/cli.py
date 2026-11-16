"""Command-line entry for the DataFlow-centered framework."""

from __future__ import annotations

import argparse
import json

from .runtime.framework import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a DataFlow evolution task")
    parser.add_argument("--task", required=True, help="JSON or YAML task config")
    parser.add_argument("--run-id")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    manifest = run(args.task, run_id=args.run_id, resume=not args.no_resume)
    print(json.dumps({
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "method_id": manifest["method_id"],
        "failure": manifest.get("failure"),
    }, ensure_ascii=False))
    return 0 if manifest["status"] == "completed" else 1
