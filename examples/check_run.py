"""Inspect one validation run and report whether the iteration strategy held.

Usage (on the server):
    python .validation/check_run.py <run_dir>/workspace <run_id>

Checks the loop's own artifacts rather than trusting the exit code: iteration
count, acceptance, prefix reuse, final dataset schema, diagnostic isolation.
Exits non-zero if any invariant is violated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    workspace, run_id = Path(argv[1]), argv[2]
    root = workspace / "runs" / run_id
    if not root.is_dir():
        print(f"FAIL: no run directory at {root}")
        return 1

    problems: list[str] = []
    manifest = _load(root / "run_manifest.json")
    print(f"status            : {manifest['status']}")
    if manifest["status"] != "completed":
        problems.append(f"status={manifest['status']} failure={manifest.get('failure')}")

    acceptance = manifest.get("acceptance", {})
    print(f"accepted iters    : {acceptance.get('accepted_iterations')}")
    print(f"best score        : {acceptance.get('best_score')}")
    print(f"diag used in score: {acceptance.get('diagnostic_used_for_score')}")
    if acceptance.get("diagnostic_used_for_score") is not False:
        problems.append("diagnostic must never enter the acceptance score")
    if not acceptance.get("accepted_iterations"):
        problems.append("no candidate was accepted")

    final = root / "records" / "dataflow" / "final_dataset.jsonl"
    if not final.is_file():
        problems.append("final dataset is missing")
    else:
        rows = [json.loads(l) for l in final.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"final dataset     : {len(rows)} rows, keys={sorted(rows[0]) if rows else []}")
        if not rows:
            problems.append("final dataset is empty")
        else:
            missing = [i for i, r in enumerate(rows)
                       if not str(r.get("instruction") or "").strip()
                       or not str(r.get("output") or "").strip()]
            if missing:
                problems.append(f"{len(missing)} rows miss instruction/output")

    evolution = root / "native" / "runs" / "evolution"
    iterations = sorted(p for p in evolution.glob("iteration_*") if p.is_dir())
    print(f"iterations on disk: {[p.name for p in iterations]}")
    for iteration in iterations:
        plan_path = iteration / "step_reuse_plan.json"
        manifest_path = iteration / "step_manifest.json"
        if not plan_path.is_file():
            problems.append(f"{iteration.name}: no step_reuse_plan.json")
            continue
        plan = _load(plan_path)
        reused = []
        if manifest_path.is_file():
            reused = [s.get("reused") for s in _load(manifest_path).get("steps", [])]
        print(f"  {iteration.name}: prefix={plan['prefix_count']}/{len(plan['operator_names'])} "
              f"reused={reused} reason={plan['reason'][:50]!r}")

    observation = root / "execution_observation.json"
    if observation.is_file():
        print(f"observation       : {_load(observation)['status']}")
    else:
        problems.append("no execution_observation.json")
    if not (root / "provenance" / "events.jsonl").is_file():
        problems.append("no provenance events")

    diagnostics = sorted((root / "diagnostics").glob("*.json")) if (root / "diagnostics").is_dir() else []
    print(f"diagnostics       : {len(diagnostics)} artifact(s)")

    print()
    if problems:
        print("RESULT: PROBLEMS FOUND")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("RESULT: iteration strategy held on every checked invariant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
