"""Key-checked execution shared by all RSI methods."""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import TaskEnvelope, fingerprint
from .operator import Operator, OperatorSpec
from ..io.storage import StorageBundle


MAX_TRACEBACK_CHARS = 32 * 1024
MAX_DIAGNOSTICS_CHARS = 64 * 1024
MAX_REPAIR_PROMPT_CHARS = 900_000


def bounded_text(value: Any, limit: int) -> str:
    text = str(value or "")
    if limit < 0:
        raise ValueError("text limit must be non-negative")
    if len(text) <= limit:
        return text
    marker = "\n...[truncated]...\n"
    if limit <= len(marker):
        return marker[:limit]
    budget = max(0, limit - len(marker))
    head = budget // 3
    tail = budget - head
    return text[:head] + marker + (text[-tail:] if tail else "")


def bounded_repair_context(
    failure_stage: str,
    traceback_text: str,
    diagnostics: str,
    *,
    max_chars: int = MAX_REPAIR_PROMPT_CHARS,
) -> str:
    text = (
        f"Failure stage: {failure_stage}\n"
        f"Traceback:\n{bounded_text(traceback_text, MAX_TRACEBACK_CHARS)}\n"
        f"Diagnostics:\n{bounded_text(diagnostics, MAX_DIAGNOSTICS_CHARS)}"
    )
    return bounded_text(text, max_chars)


@dataclass
class RepairBudget:
    limit: int
    used: int = 0

    def __post_init__(self) -> None:
        if self.limit < 0 or self.used < 0:
            raise ValueError("repair budget cannot be negative")

    def claim(self) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    version: str
    input_keys: tuple[str, ...]
    result_keys: tuple[str, ...]
    operators: tuple[OperatorSpec, ...]

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "input_keys": list(self.input_keys),
            "result_keys": list(self.result_keys),
            "operators": [operator.to_dict() for operator in self.operators],
        }


@dataclass
class RunContext:
    task: TaskEnvelope
    run_id: str
    storage: StorageBundle
    resume: bool = True
    resources: dict[str, Any] = field(default_factory=dict)

    @property
    def store(self) -> StorageBundle:
        return self.storage


@dataclass(frozen=True)
class PipelineResult:
    output: dict[str, Any]
    state: dict[str, Any]
    observation_path: Path
    spec: PipelineSpec


class PipelineExecutionError(RuntimeError):
    def __init__(self, stage: str, original: Exception, repair_context: str) -> None:
        super().__init__(f"pipeline failed at {stage}: {original}")
        self.stage = stage
        self.original = original
        self.repair_context = repair_context


def _output_summary(value: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": type(value).__name__}
    if isinstance(value, (dict, list, tuple)):
        summary["count"] = len(value)
    if isinstance(value, list) and value and isinstance(value[0], dict):
        summary["first_record_keys"] = sorted(str(key) for key in value[0])[:40]
        summary["bounded_sample"] = bounded_text(
            json.dumps(value[:2], ensure_ascii=False, default=str), 4000
        )
    return summary


class Pipeline:
    def __init__(
        self,
        name: str,
        operators: Iterable[Operator],
        *,
        result_keys: Iterable[str],
        version: str = "1",
    ) -> None:
        self.name = name
        self.version = version
        self.operators = tuple(operators)
        self.result_keys = tuple(result_keys)
        if not self.operators:
            raise ValueError("pipeline needs at least one operator")
        if not self.result_keys:
            raise ValueError("pipeline needs at least one result key")
        self._spec: PipelineSpec | None = None

    def compile(self, initial_keys: Iterable[str]) -> PipelineSpec:
        initial = tuple(sorted(set(initial_keys)))
        available = set(initial)
        specs: list[OperatorSpec] = []
        for operator in self.operators:
            missing = set(operator.input_keys) - available
            if missing:
                raise KeyError(f"{operator.name} requires missing keys {sorted(missing)}")
            collision = set(operator.output_keys) & available
            if collision:
                raise KeyError(f"{operator.name} would overwrite keys {sorted(collision)}")
            if len(set(operator.output_keys)) != len(operator.output_keys):
                raise ValueError(f"{operator.name} declares duplicate output keys")
            specs.append(
                OperatorSpec(
                    name=operator.name,
                    version=str(operator.version),
                    input_keys=tuple(operator.input_keys),
                    output_keys=tuple(operator.output_keys),
                    config=operator.config(),
                    fingerprint=operator.fingerprint,
                )
            )
            available.update(operator.output_keys)
        missing_results = set(self.result_keys) - available
        if missing_results:
            raise KeyError(f"pipeline result is missing keys {sorted(missing_results)}")
        self._spec = PipelineSpec(self.name, self.version, initial, self.result_keys, tuple(specs))
        return self._spec

    def spec(self, initial_keys: Iterable[str] | None = None) -> PipelineSpec:
        if initial_keys is not None:
            return self.compile(initial_keys)
        if self._spec is None:
            raise ValueError("pipeline must be compiled before reading its spec")
        return self._spec

    def run(self, state: Mapping[str, Any], context: RunContext) -> PipelineResult:
        spec = self.compile(state.keys())
        working = dict(state)
        input_fingerprint = fingerprint(working)
        observations: list[dict[str, Any]] = []
        stage_offset = int(context.resources.get("stage_offset", 0))
        if stage_offset < 0:
            raise ValueError("stage_offset must be non-negative")
        for index, operator in enumerate(self.operators, start=1):
            stage = f"{index:03d}-{operator.name}"
            context.resources["stage_index"] = stage_offset + index
            stage_identity = fingerprint(
                {
                    "task_id": context.task.task_id,
                    "method_id": context.task.method_id,
                    "run_id": context.run_id,
                    "pipeline": spec.fingerprint,
                    "input": input_fingerprint,
                    "index": index,
                    "operator": operator.fingerprint,
                    "skill_bundle": context.resources.get("skill_bundle_fingerprint"),
                    "provider": context.task.provider,
                    "role": context.task.role,
                }
            )
            resumed = False
            outputs = (
                context.storage.checkpoints.load(stage, stage_identity)
                if context.resume
                else None
            )
            try:
                if outputs is None:
                    missing = set(operator.input_keys) - set(working)
                    if missing:
                        raise KeyError(f"{operator.name} is missing runtime keys {sorted(missing)}")
                    raw = operator.run(working, context)
                    if not isinstance(raw, Mapping):
                        raise TypeError(f"{operator.name}.run must return a mapping")
                    outputs = dict(raw)
                    absent = set(operator.output_keys) - set(outputs)
                    extra = set(outputs) - set(operator.output_keys)
                    if absent or extra:
                        raise KeyError(
                            f"{operator.name} output mismatch: missing={sorted(absent)}, extra={sorted(extra)}"
                        )
                    json.dumps(outputs, ensure_ascii=False)
                    context.storage.checkpoints.save(stage, stage_identity, outputs)
                else:
                    resumed = True
                working.update(outputs)
                event = {
                    "stage": stage,
                    "operator": operator.name,
                    "operator_fingerprint": operator.fingerprint,
                    "resumed": resumed,
                    "output": {key: _output_summary(value) for key, value in outputs.items()},
                }
                observations.append(event)
                context.storage.provenance.append(
                    {
                        "task_id": context.task.task_id,
                        "method_id": context.task.method_id,
                        "run_id": context.run_id,
                        "pipeline_fingerprint": spec.fingerprint,
                        **event,
                    }
                )
            except Exception as exc:
                failure = {
                    "status": "failed",
                    "stage": stage,
                    "failure_type": type(exc).__name__,
                    "message": bounded_text(exc, 4000),
                    "traceback": bounded_text(traceback.format_exc(), MAX_TRACEBACK_CHARS),
                }
                context.storage.run.append_failure(failure)
                context.storage.run.write_observation(
                    {
                        "schema_version": 1,
                        "status": "failed",
                        "failure": failure,
                        "stages": observations,
                    }
                )
                raise PipelineExecutionError(
                    stage,
                    exc,
                    bounded_repair_context(stage, failure["traceback"], ""),
                ) from exc
        output = {key: working[key] for key in spec.result_keys}
        observation_path = context.storage.run.write_observation(
            {
                "schema_version": 1,
                "status": "completed",
                "task_id": context.task.task_id,
                "method_id": context.task.method_id,
                "run_id": context.run_id,
                "pipeline_fingerprint": spec.fingerprint,
                "stages": observations,
                "result_keys": list(output),
            }
        )
        return PipelineResult(output, working, observation_path, spec)
