"""Bounded, reproducible Codex-driven RSI evolution loop.

The agent edits an isolated candidate workspace.  A candidate is promoted only
after tests and the fixed probe run succeed and the single primary proxy
strictly improves.  Every accepted version gets its own source and output
directory; rejected candidates remain in the round directory for audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_FOR_CONTROLLER = Path(os.environ.get(
    "RSI_SOURCE_ROOT", str(HERE.parent / "src")
)).resolve()
if str(SOURCE_FOR_CONTROLLER) not in sys.path:
    sys.path.insert(0, str(SOURCE_FOR_CONTROLLER))

from video_rsi.iteration import IterationConfig, RSIIterationController  # noqa: E402
from evaluation import EvaluationPolicy, evaluate_run  # noqa: E402


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    source = source.resolve()

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored = {name for name in names if name in {"__pycache__", ".git", ".pytest_cache"} or name.endswith((".pyc", ".log"))}
        # Historical experiments and benchmark predictions are audit data,
        # not editable pipeline context.  Excluding them prevents Codex from
        # recursively reading hundreds of thousands of irrelevant tokens.
        if Path(directory).resolve() == source:
            ignored.update(name for name in names if name in {"experiments", "results"})
        return ignored

    shutil.copytree(source, destination, ignore=ignore)


def _write_state(root: Path, **fields: object) -> None:
    """Persist a small live heartbeat without exposing credentials."""
    state = root / "evolution_state.json"
    current = {}
    if state.exists():
        try:
            current = json.loads(state.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    current.update(fields)
    state.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")


def _candidate_fingerprint(workspace: Path) -> str:
    """Hash editable pipeline surfaces, excluding generated artifacts."""
    digest = hashlib.sha256()
    for relative in ("src", "tests", "configs", "docs"):
        root = workspace / relative
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            digest.update(str(path.relative_to(workspace)).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _surface_fingerprints(workspace: Path) -> dict[str, str]:
    """Return per-file hashes used to identify Lego-level changes."""
    result: dict[str, str] = {}
    for relative in ("src", "tests", "configs", "docs"):
        root = workspace / relative
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            result[str(path.relative_to(workspace))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return result


def _changed_surfaces(before: dict[str, str], workspace: Path) -> list[str]:
    """Classify changed files as prompt, operator, pipeline or task surfaces."""
    after = _surface_fingerprints(workspace)
    surfaces: set[str] = set()
    for relative in set(before) | set(after):
        if before.get(relative) == after.get(relative):
            continue
        parts = Path(relative).parts
        name = Path(relative).name
        if "operators" in parts:
            surfaces.add("operator")
        if "pipelines" in parts or name in {"rsi_round.py", "bootstrap.py"}:
            surfaces.add("pipeline")
        if name == "task_prompts.py" or "prompts" in parts:
            surfaces.add("prompt")
        if name in {"task_catalog.py", "task_candidates.py"}:
            surfaces.add("task")
    return sorted(surfaces)


def _operator_feedback(summary: dict) -> str:
    """Compact, operator-oriented feedback for the next Codex proposal."""
    return json.dumps({
        "feedback_signals": summary.get("raw_run_summary", {}).get("feedback_signals", {}),
        "diagnostics": summary.get("diagnostics", {}),
    }, ensure_ascii=False, sort_keys=True)


def _probe_examples(root: Path, version: str, limit: int = 6) -> list[str]:
    """Extract small concrete samples so Codex can see failure modes, not counts only."""
    outputs = root / "versions" / version / "outputs"
    # A resumed evolution run may already contain a probe with the historical
    # fixed name.  Search all probe attempts and use the newest available one
    # instead of assuming the first attempt completed successfully.
    probes = sorted(outputs.glob(f"{version}_fixed_probe*/videos"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    probe = probes[0] if probes else outputs / f"{version}_fixed_probe" / "videos"
    by_bucket: dict[str, list[dict[str, Any]]] = {
        "too_easy": [], "hard_review_required": [], "frontier": [], "other": []
    }
    # Stage numbers are intentionally not hard-coded: optional operators (such
    # as the external Gemini verifier) may be omitted from a pipeline variant.
    # Select the frozen-target artifact by operator name instead.
    for path in sorted(probe.glob("*/stages/*_frozen_target_frontier_filter.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in (payload.get("outputs", {}).get("frontier_results", []) or []):
            target = row.get("target_frontier", {}) or {}
            verification = row.get("verification", {}) or {}
            example = {
                "task_type": row.get("task_type"),
                "question": row.get("question"),
                "choices": row.get("choices"),
                "answer": row.get("answer"),
                "bucket": target.get("bucket"),
                "verification": verification,
            }
            bucket = str(target.get("bucket") or "other")
            by_bucket.setdefault(bucket, by_bucket["other"]).append(example)
    # Deliberately show the agent both sides of the frontier.  A prefix of the
    # artifact list is often dominated by too-easy questions and hides the
    # hard/ambiguous failure mode that needs a different task or evidence plan.
    examples: list[str] = []
    order = ("too_easy", "hard_review_required", "frontier", "other")
    while len(examples) < limit and any(by_bucket.get(key) for key in order):
        for key in order:
            rows = by_bucket.get(key) or []
            if rows and len(examples) < limit:
                examples.append(json.dumps(rows.pop(0), ensure_ascii=False))
    return examples


def _last_decision_marker(text: str) -> str | None:
    """Return the final explicit decision, ignoring prompt instructions."""
    matches = re.findall(r"DECISION\s*[:=]\s*(STOP|CONTINUE)\b", text.upper())
    return matches[-1] if matches else None


def _environment_failure(text: str) -> bool:
    lowered = text.lower()
    return bool(re.search(
        r"bwrap|namespace|read-only|permission denied|operation not permitted|"
        r"sandbox.*(fail|error)|failed to start.*sandbox|failed to write file|"
        r"approval policy is never",
        lowered,
    ))


def _transient_codex_transport_failure(result: dict) -> bool:
    """Return true for provider/network failures that must not spend a round."""
    text = (str(result.get("stdout", "")) + "\n" + str(result.get("stderr", ""))).lower()
    return bool(re.search(
        r"no available channel|service unavailable|status (429|502|503|504)|"
        r"connection (reset|closed|refused)|failed to connect|reconnecting\.\.\.|"
        r"stream disconnected|stream closed before response\.completed",
        # Some compatible Responses gateways close an otherwise successful
        # SSE request before ``response.completed``.  It is transient and
        # must not consume an RSI version/round.
        text,
    ))


def _run_probe(source_root: Path, args: argparse.Namespace, version: str, root: Path) -> dict:
    # RunStore intentionally rejects reusing a run_id with a different
    # manifest.  This commonly happens after an interrupted probe (for
    # example, after fixing ffmpeg or changing frontier_trials).  Preserve the
    # old attempt and allocate a deterministic retry id rather than crashing
    # the whole evolution loop.
    outputs = root / "versions" / version / "outputs"
    base_id = f"{version}_fixed_probe"
    run_id = base_id
    attempt = 1
    while (outputs / run_id / "run_manifest.json").exists():
        try:
            existing = json.loads((outputs / run_id / "run_manifest.json").read_text(encoding="utf-8"))
        except Exception:
            existing = None
        # A completed run can be reused safely; an incompatible/stale one
        # gets a new id and remains available for audit.
        if existing and existing.get("pipeline", {}).get("operators"):
            attempt += 1
            run_id = f"{base_id}_retry{attempt}"
        else:
            break
    command = [
        sys.executable, str(HERE / "run_experiment.py"),
        "--caption-root", str(args.caption_root), "--experiment-root", str(root),
        "--source-root", str(source_root), "--version", version,
        "--run-id", run_id,
        "--limit", str(args.limit), "--max-per-task", str(args.max_per_task),
        "--frontier-trials", str(args.frontier_trials),
        "--frozen-target-model", args.frozen_target_model,
        # Candidate probes are deliberately local-only.  This keeps RSI
        # iteration from repeatedly consuming the expensive remote chat API;
        # the local vLLM endpoint is still exercised with real probe videos.
        "--local-only-api",
    ]
    if args.question_limit is not None:
        command.extend(["--question-limit", str(args.question_limit)])
    if args.allow_text_frozen_fallback:
        command.append("--allow-text-frozen-fallback")
    if args.config_path:
        command.extend(["--config-path", str(args.config_path)])
    env = dict(os.environ)
    env["RSI_SOURCE_ROOT"] = str(source_root)
    completed = subprocess.run(command, cwd=str(HERE), env=env, text=True,
                               capture_output=True, check=False)
    log_root = root / "iteration_rounds" / f"{version}_probe"
    log_root.mkdir(parents=True, exist_ok=True)
    (log_root / "probe.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (log_root / "probe.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"probe failed with returncode={completed.returncode}")
    run_dir = root / "versions" / version / "outputs" / run_id
    result = evaluate_run(run_dir, EvaluationPolicy(args.frozen_target_model))
    result["raw_run_summary"] = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    return result


def _load_completed_probe(root: Path, version: str, frozen_target_model: str) -> dict:
    """Load the newest completed fixed probe when resuming an evolution run."""
    outputs = root / "versions" / version / "outputs"
    candidates = [
        path.parent for path in outputs.glob(f"{version}_fixed_probe*/run_summary.json")
        if (path.parent / "run_manifest.json").exists()
    ]
    if not candidates:
        raise RuntimeError(f"no completed probe found for resume version {version}")
    run_dir = max(candidates, key=lambda path: path.stat().st_mtime)
    result = evaluate_run(run_dir, EvaluationPolicy(frozen_target_model))
    result["raw_run_summary"] = json.loads(
        (run_dir / "run_summary.json").read_text(encoding="utf-8")
    )
    return result


def _run_tests(workspace: Path, source_root: Path, round_root: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(source_root)
    preflight = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "src"],
        cwd=str(workspace), env=env, text=True, capture_output=True, check=False,
    )
    (round_root / "preflight_compile.stdout.log").write_text(preflight.stdout, encoding="utf-8")
    (round_root / "preflight_compile.stderr.log").write_text(preflight.stderr, encoding="utf-8")
    if preflight.returncode != 0:
        raise RuntimeError(f"candidate compile preflight failed with returncode={preflight.returncode}")
    import_check = subprocess.run(
        [sys.executable, "-c", "from video_rsi.pipelines import build_task_centric_rsi_pipeline; from video_rsi.registry import OPERATOR_REGISTRY; print('pipeline_import_ok', len(OPERATOR_REGISTRY.names()))"],
        cwd=str(workspace), env=env, text=True, capture_output=True, check=False,
    )
    (round_root / "preflight_import.stdout.log").write_text(import_check.stdout, encoding="utf-8")
    (round_root / "preflight_import.stderr.log").write_text(import_check.stderr, encoding="utf-8")
    if import_check.returncode != 0:
        raise RuntimeError(f"candidate import preflight failed with returncode={import_check.returncode}")
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=str(workspace), env=env, text=True, capture_output=True, check=False,
    )
    (round_root / "tests.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (round_root / "tests.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"candidate tests failed with returncode={completed.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded autonomous RSI evolution")
    parser.add_argument("--caption-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, default=HERE)
    parser.add_argument("--source-tree", type=Path, default=HERE.parent)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--question-limit", type=int, default=None,
                        help="Maximum number of retained questions in every fixed version probe.")
    parser.add_argument("--budget", type=int, default=3,
                        help="Maximum candidate proposals; Codex decides when to stop earlier.")
    parser.add_argument("--max-per-task", type=int, default=1,
                        help="Maximum one candidate per task type and video by default.")
    parser.add_argument("--frontier-trials", type=int, default=4)
    parser.add_argument("--frozen-target-model", default="qwen3-vl-8b-instruct")
    parser.add_argument("--allow-text-frozen-fallback", action="store_true")
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--coding-model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument(
        "--resume-existing", action="store_true",
        help="Resume after the newest contiguous CONTINUOUS_ACTIVE version.",
    )
    parser.add_argument(
        "--no-exploration-ties", action="store_true",
        help="Disable structural promotion when baseline and candidate both have zero primary samples.",
    )
    args = parser.parse_args()

    root = args.experiment_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_state(root, status="running", phase="initializing", budget=args.budget)
    if args.resume_existing:
        completed = []
        for version_index in range(1, args.budget + 1):
            marker = root / "versions" / f"v{version_index}" / "CONTINUOUS_ACTIVE"
            if not marker.exists():
                break
            completed.append(version_index)
        if not completed:
            raise RuntimeError("--resume-existing requires at least one CONTINUOUS_ACTIVE version")
        last_index = completed[-1]
        active_version = f"v{last_index}"
        active_tree = root / "iteration_rounds" / f"round_{last_index:03d}" / "candidate_workspace"
        if not (active_tree / "src").exists():
            raise RuntimeError(f"candidate workspace missing for {active_version}: {active_tree}")
        baseline = _load_completed_probe(root, active_version, args.frozen_target_model)
        start_index = last_index + 1
        _write_state(
            root, phase="resumed", version=active_version,
            baseline_proxy=baseline.get("accepted_frontier_novel_count", 0),
            resume_from=active_version,
        )
    else:
        active_tree = root / "active_source"
        _copy_tree(args.source_tree.resolve(), active_tree)
        _write_state(root, phase="v0_probe_started", version="v0")
        baseline = _run_probe(active_tree / "src", args, "v0", root)
        active_version = "v0"
        start_index = 1
        (root / "baseline_summary.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_state(root, phase="v0_complete", version="v0", baseline_proxy=baseline.get("accepted_frontier_novel_count", 0))

    # ``budget`` is only a safety ceiling; Codex decides whether to stop.
    for index in range(start_index, args.budget + 1):
        round_id = f"round_{index:03d}"
        round_root = root / "iteration_rounds" / round_id
        _write_state(root, phase="codex_proposal", round_id=round_id, budget_used=index)
        candidate_tree = round_root / "candidate_workspace"
        _copy_tree(active_tree, candidate_tree)
        before_fingerprint = _candidate_fingerprint(candidate_tree)
        before_surfaces = _surface_fingerprints(candidate_tree)
        controller = RSIIterationController(IterationConfig(
            workspace=candidate_tree, coding_model=args.coding_model,
            reasoning_effort=args.reasoning_effort, max_rounds=1,
            # Shared servers disallow unprivileged bwrap namespaces.  Use the
            # already-approved full-access mode for the isolated candidate
            # workspace; otherwise every proposal can be lost to infrastructure
            # before Codex gets a chance to edit.
            sandbox_mode="danger-full-access",
        ))
        transport_attempts = int(os.environ.get("RSI_CODEX_ROUND_RETRIES", "20"))
        transport_sleep = float(os.environ.get("RSI_CODEX_ROUND_RETRY_SLEEP_SEC", "60"))
        for transport_retry in range(transport_attempts + 1):
            proposal = controller.propose_change(
                round_id=round_id, baseline_summary=baseline,
                logs=[_operator_feedback(baseline)],
                examples=_probe_examples(root, active_version),
            )
            if proposal.get("returncode") == 0:
                break
            if not _transient_codex_transport_failure(proposal):
                break
            _write_state(
                root, status="running", phase="codex_transport_retry",
                round_id=round_id, version=active_version,
                transport_retry=transport_retry + 1,
                transport_retry_limit=transport_attempts,
            )
            time.sleep(transport_sleep)
        proposal["transport_retries_used"] = transport_retry
        (round_root / "codex_result.json").parent.mkdir(parents=True, exist_ok=True)
        (round_root / "codex_result.json").write_text(
            json.dumps({k: v for k, v in proposal.items() if k not in {"stdout", "stderr"}}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Codex may return exit code 0 while its internal tool calls fail
        # inside bwrap.  The controller retries with its fallback sandbox.
        # A historical first-attempt error is harmless when the final attempt
        # produced a complete proposal and changed the candidate workspace;
        # keep the warning in codex_result.json but continue validation.
        proposal_file = candidate_tree / "CHANGE_PROPOSAL.md"
        has_proposal_artifact = proposal_file.exists()
        if proposal.get("sandbox_failure") and not (
            proposal.get("returncode") == 0 and has_proposal_artifact
        ):
            decision = {"round_id": round_id, "decision": "reject",
                        "reason": "codex_environment_failure", "budget_used": index}
            (round_root / "decision.json").write_text(
                json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            _write_state(root, phase="codex_environment_failure",
                         round_id=round_id, version=f"v{index}")
            _write_state(root, status="failed", phase="codex_environment_failure",
                         round_id=round_id, version=f"v{index}")
            raise RuntimeError(f"{round_id} Codex environment failure after retries")
        if proposal.get("returncode") != 0:
            (round_root / "decision.json").write_text(json.dumps({"decision": "reject", "reason": "codex_failed"}, indent=2), encoding="utf-8")
            _write_state(root, status="failed", phase="codex_failed",
                         round_id=round_id, version=f"v{index}")
            raise RuntimeError(f"{round_id} Codex failed after retries")
        proposal_text = str(proposal.get("stdout", ""))
        if proposal_file.exists():
            proposal_text += "\n" + proposal_file.read_text(encoding="utf-8", errors="replace")
        marker = _last_decision_marker(proposal_text)
        if marker == "STOP" and _environment_failure(proposal_text):
            decision = {"round_id": round_id, "decision": "reject",
                        "reason": "codex_environment_failure", "budget_used": index}
            (round_root / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
            _write_state(root, phase="codex_environment_failure", round_id=round_id, version=f"v{index}")
            _write_state(root, status="failed", phase="codex_environment_failure",
                         round_id=round_id, version=f"v{index}")
            raise RuntimeError(f"{round_id} Codex environment failure")
        if marker == "STOP" and int(baseline.get("accepted_frontier_novel_count", 0)) <= 0:
            decision = {"round_id": round_id, "decision": "reject",
                        "reason": "stop_blocked_zero_frontier", "budget_used": index}
            (round_root / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
            _write_state(root, phase="stop_blocked_zero_frontier", round_id=round_id, version=f"v{index}")
            _write_state(root, status="failed", phase="stop_blocked_zero_frontier",
                         round_id=round_id, version=f"v{index}")
            raise RuntimeError(f"{round_id} stopped before producing a valid next version")
        if marker == "STOP":
            decision = {"round_id": round_id, "decision": "stop", "reason": "codex_stop", "budget_used": index}
            (round_root / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
            _write_state(root, status="stopped", phase="codex_stop", round_id=round_id, budget_used=index)
            break
        if _candidate_fingerprint(candidate_tree) == before_fingerprint:
            decision = {"round_id": round_id, "decision": "reject",
                        "reason": "codex_no_code_change", "budget_used": index}
            (round_root / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
            (root / "versions" / f"v{index}" / "REJECTED").parent.mkdir(parents=True, exist_ok=True)
            (root / "versions" / f"v{index}" / "REJECTED").write_text(
                json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            _write_state(root, phase="rejected_no_code_change", round_id=round_id, version=f"v{index}")
            _write_state(root, status="failed", phase="rejected_no_code_change",
                         round_id=round_id, version=f"v{index}")
            raise RuntimeError(f"{round_id} produced no code change")
        try:
            _write_state(root, phase="candidate_probe", round_id=round_id, version=f"v{index}")
            _run_tests(candidate_tree, candidate_tree / "src", round_root)
            candidate = _run_probe(candidate_tree / "src", args, f"v{index}", root)
            quality_assessment = controller.run_round(
                round_id=round_id,
                baseline_summary=baseline,
                candidate_summary=candidate,
            )
            changed_surfaces = _changed_surfaces(before_surfaces, candidate_tree)
            if not changed_surfaces:
                raise RuntimeError("candidate has no classified code surface change")
            if not candidate.get("eligible_for_promotion"):
                raise RuntimeError("candidate probe failed fixed hard gates")
            # RSI is a continuous lineage, not hill climbing on one scalar.
            # Once code, tests and the fixed probe are healthy, vN always
            # becomes the parent of vN+1.  Yield/difficulty/quality changes are
            # retained as feedback for Codex, never used to break the lineage.
            decision = {
                **quality_assessment,
                "decision": "inherit",
                "reason": "continuous_evolution_valid_candidate",
                "changed_surfaces": changed_surfaces,
                "quality_assessment": {
                    "decision": quality_assessment.get("decision"),
                    "reason": quality_assessment.get("reason"),
                    "baseline_proxy": quality_assessment.get("baseline_proxy"),
                    "candidate_proxy": quality_assessment.get("candidate_proxy"),
                    "comparison": quality_assessment.get("comparison", {}),
                },
            }
        except Exception as exc:
            decision = {"round_id": round_id, "decision": "reject", "reason": str(exc)}
            candidate = None
        (round_root / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
        if decision.get("decision") == "inherit" and candidate is not None:
            active_tree = candidate_tree
            baseline = candidate
            active_version = f"v{index}"
            _write_state(root, phase="inherited", round_id=round_id, version=f"v{index}", baseline_proxy=baseline.get("accepted_frontier_novel_count", 0))
            marker = root / "versions" / f"v{index}" / "CONTINUOUS_ACTIVE"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            # Keep the rejected candidate workspace but do not make it active.
            # Rejected versions are intentionally retained for audit and later
            # diagnosis; only the active pointer remains unchanged.
            (root / "versions" / f"v{index}" / "REJECTED").write_text(
                json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            _write_state(root, phase="rejected", round_id=round_id, version=f"v{index}")

    missing = [
        f"v{index}" for index in range(1, args.budget + 1)
        if not (root / "versions" / f"v{index}" / "CONTINUOUS_ACTIVE").exists()
    ]
    if missing:
        _write_state(root, status="failed", phase="incomplete_lineage", missing_versions=missing)
        raise RuntimeError(f"continuous lineage incomplete: {missing}")
    _write_state(root, status="complete", phase="finished", budget_used=args.budget)
    print(json.dumps(baseline, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
