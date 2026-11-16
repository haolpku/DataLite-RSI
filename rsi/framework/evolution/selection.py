"""User-configurable iteration selection, independent of criteria feedback."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .controller import Candidate
    from .evaluation.downstream_eval import DownstreamEvaluationResult
    from .models import TaskSpec


class SelectionPolicyError(ValueError):
    """A configured policy failed; selection must not silently fall back."""


@dataclass(frozen=True)
class SelectionPolicy:
    kind: str = "criteria"
    policy_file: str | None = None
    description: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    timeout_sec: float = 30.0
    sha256: str | None = None

    @classmethod
    def from_config(
        cls, raw: Mapping[str, Any] | None, *, base_dir: Path
    ) -> SelectionPolicy:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("selection_policy must be a mapping")
        unknown = set(raw) - {"kind", "policy_file", "description", "config", "timeout_sec"}
        if unknown:
            raise ValueError(f"unknown selection_policy keys: {sorted(unknown)}")
        kind = str(raw.get("kind", "criteria")).strip().lower()
        if kind not in {"criteria", "python"}:
            raise ValueError("selection_policy.kind must be criteria or python")
        config = raw.get("config", {})
        if not isinstance(config, Mapping):
            raise ValueError("selection_policy.config must be a JSON object")
        config = dict(config)
        try:
            json.dumps(config, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("selection_policy.config must be JSON serializable") from exc
        timeout = raw.get("timeout_sec", 30)
        if isinstance(timeout, bool):
            raise ValueError("selection_policy.timeout_sec must be positive")
        try:
            timeout = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("selection_policy.timeout_sec must be positive") from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("selection_policy.timeout_sec must be positive")
        description = str(raw.get("description", "") or "")
        name = raw.get("policy_file")
        if kind == "criteria":
            if name:
                raise ValueError("criteria selection does not take a policy_file")
            return cls(description=description, config=config, timeout_sec=timeout)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("python selection requires a policy_file")
        path = Path(name).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        path = path.resolve()
        try:
            source = path.read_bytes()
            compile(source, str(path), "exec")
        except (OSError, SyntaxError) as exc:
            raise ValueError(f"invalid selection policy file {path}: {exc}") from exc
        return cls(
            kind="python", policy_file=str(path), description=description,
            config=config, timeout_sec=timeout,
            sha256=hashlib.sha256(source).hexdigest(),
        )

    def identity(self) -> dict[str, str | None]:
        """Values used by the outer checkpoint without exposing policy parameters."""
        parameters = json.dumps(self.config, sort_keys=True, allow_nan=False)
        return {
            "kind": self.kind,
            "source_sha256": self.sha256,
            "config_sha256": hashlib.sha256(parameters.encode("utf-8")).hexdigest(),
        }


def _read_observation_rows(path: str | None) -> int | None:
    if not path:
        return None
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    final = value.get("final_output") if isinstance(value, dict) else None
    rows = final.get("rows") if isinstance(final, dict) else None
    return rows if type(rows) is int and rows >= 0 else None


class PolicySelector:
    """Run a trusted Python policy in a time-limited child process and audit it."""

    def __init__(
        self, policy: SelectionPolicy, *, run_dir: Path, workspace: Path, task: TaskSpec
    ) -> None:
        if policy.kind != "python" or not policy.policy_file or not policy.sha256:
            raise ValueError("PolicySelector requires a validated python policy")
        self.policy = policy
        self.run_dir = run_dir
        self.workspace = workspace
        self.task = task

    def _context(
        self,
        candidates: list[Candidate],
        incumbent: Candidate | None,
        feedback: list[DownstreamEvaluationResult],
        event: str,
    ) -> dict[str, Any]:
        eligible = []
        for candidate in candidates:
            review = candidate.review
            if (
                review is None or candidate.execution_error or not candidate.dataset_path
                or not Path(candidate.dataset_path).is_file()
            ):
                continue
            eligible.append({
                "iteration": candidate.iteration + 1,
                "directory": str(candidate.iteration_dir),
                "pipeline_path": str(candidate.iteration_dir / "pipeline.py"),
                "dataset_path": candidate.dataset_path,
                "review_score": candidate.score,
                "review_source": candidate.review_source,
                "passed": bool(review.passed),
                "issues": list(review.issues),
                "review_feedback": {
                    "score_components": review.score_components,
                    "domain_feedback": review.domain_feedback,
                },
                "output_rows": _read_observation_rows(
                    candidate.execution_observation_path
                ),
                "execution_observation_path": candidate.execution_observation_path,
                "wall_time_seconds": candidate.wall_time_seconds,
            })
        # Only summarized, validated benchmark scores enter the policy context.
        # Raw bad cases and attribution evidence remain outside this boundary.
        scores = [{
            "source": item.source,
            "stage": item.stage,
            "status": item.status,
            "checkpoint_iteration": item.checkpoint_iteration,
            "iteration": item.incumbent_iteration,
            "scores": {name: result.score for name, result in item.benchmarks.items()},
            "delta_previous": item.score_deltas,
            "delta_baseline": item.score_deltas_vs_baseline,
        } for item in feedback]
        return {
            "schema_version": 1,
            "event": event,
            "workspace_dir": str(self.workspace),
            "run_dir": str(self.run_dir),
            "task_description": self.task.task_description,
            "quality_criteria": list(self.task.quality_criteria),
            "incumbent_iteration": incumbent.iteration + 1 if incumbent else None,
            "policy": asdict(self.policy),
            "config": self.policy.config,
            "candidates": eligible,
            "feedback": scores,
        }

    def select(
        self,
        candidates: list[Candidate],
        incumbent: Candidate | None,
        feedback: list[DownstreamEvaluationResult],
        *,
        event: str,
    ) -> tuple[Candidate | None, dict[str, Any]]:
        audit = self.run_dir / "selection_history" / (
            f"{time.time_ns()}_{uuid.uuid4().hex[:8]}"
        )
        record: dict[str, Any] = {
            "status": "error", "event": event, "audit_dir": str(audit),
            "policy_kind": "python", "policy_sha256": self.policy.sha256,
        }
        try:
            audit.mkdir(parents=True)
            context = self._context(candidates, incumbent, feedback, event)
            request_path = audit / "request.json"
            result_path = audit / "result.json"
            request_path.write_text(
                json.dumps(context, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            source = Path(self.policy.policy_file).read_bytes()
            if hashlib.sha256(source).hexdigest() != self.policy.sha256:
                raise ValueError("selection policy changed; start a new run with the new file")
            (audit / "policy.py").write_bytes(source)
            with (audit / "policy.log").open("w", encoding="utf-8") as log:
                proc = subprocess.run(
                    [sys.executable, str(Path(__file__).with_name("selection_runner.py")),
                     str(request_path), str(result_path)],
                    cwd=self.workspace, stdout=log, stderr=subprocess.STDOUT,
                    timeout=self.policy.timeout_sec, check=False,
                )
            if proc.returncode:
                raise ValueError(
                    f"selection policy exited with code {proc.returncode}; see policy.log"
                )
            decision = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(decision, dict) or "iteration" not in decision:
                raise ValueError("select(context) must return iteration and reason")
            iteration = decision["iteration"]
            choices = {item["iteration"]: item for item in context["candidates"]}
            if iteration is not None and (type(iteration) is not int or iteration not in choices):
                raise ValueError("selected iteration must be an eligible integer or null")
            reason = decision.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("selection reason must be a non-empty string")
            selected = next(
                (candidate for candidate in candidates if candidate.iteration + 1 == iteration),
                None,
            )
            record.update(status="ok", iteration=iteration, reason=reason)
            return selected, record
        except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise SelectionPolicyError(
                f"selection policy failed at {event}: {record['error']}; audit: {audit}"
            ) from exc
        finally:
            if audit.is_dir():
                (audit / "selection.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
                )
