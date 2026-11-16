"""Composable, key-checked, per-video streaming pipeline runtime."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .storage import RunStore


@dataclass
class RunContext:
    run_id: str
    store: RunStore
    resume: bool = True
    resources: dict[str, Any] = field(default_factory=dict)


class Operator(ABC):
    """One auditable transformation over a per-video state dictionary."""

    name = "operator"
    version = "1"
    input_keys: tuple[str, ...] = ()
    output_keys: tuple[str, ...] = ()

    def config(self) -> dict[str, Any]:
        return {}

    @property
    def fingerprint(self) -> str:
        payload = {
            "class": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "name": self.name,
            "version": self.version,
            "config": self.config(),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @abstractmethod
    def run(self, state: dict[str, Any], context: RunContext) -> dict[str, Any]:
        pass

    def stats(self, outputs: dict[str, Any]) -> dict[str, Any]:
        return {
            key: len(value) if isinstance(value, list) else 1
            for key, value in outputs.items()
        }


class Pipeline:
    def __init__(
        self,
        name: str,
        operators: Iterable[Operator],
        *,
        result_keys: Iterable[str] = ("semantic_events", "task_opportunities"),
        version: str = "0.1.0",
    ) -> None:
        self.name = name
        self.operators = list(operators)
        self.result_keys = tuple(result_keys)
        self.version = version
        if not self.operators:
            raise ValueError("pipeline needs at least one operator")
        self._compiled_keys: set[str] | None = None

    def compile(self, initial_keys: Iterable[str]) -> set[str]:
        keys = set(initial_keys)
        for operator in self.operators:
            missing = set(operator.input_keys) - keys
            if missing:
                raise KeyError(
                    f"{operator.name} requires missing keys {sorted(missing)}; "
                    f"available={sorted(keys)}"
                )
            duplicate = set(operator.output_keys) & keys
            if duplicate:
                raise KeyError(
                    f"{operator.name} would overwrite keys {sorted(duplicate)}"
                )
            keys.update(operator.output_keys)
        self._compiled_keys = keys
        return set(keys)

    def description(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "result_keys": list(self.result_keys),
            "operators": [
                {
                    "name": op.name,
                    "version": op.version,
                    "input_keys": list(op.input_keys),
                    "output_keys": list(op.output_keys),
                    "config": op.config(),
                    "fingerprint": op.fingerprint,
                }
                for op in self.operators
            ],
        }

    @property
    def fingerprint(self) -> str:
        payload = self.description()
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def run_record(
        self,
        state: dict[str, Any],
        context: RunContext,
    ) -> dict[str, Any]:
        if self._compiled_keys is None:
            self.compile(state.keys())
        video_key = str(state["video_key"])
        working = dict(state)
        stage_provenance = []
        for index, operator in enumerate(self.operators, start=1):
            outputs = None
            resumed = False
            if context.resume:
                outputs = context.store.load_stage(
                    video_key,
                    index,
                    operator.name,
                    operator.fingerprint,
                )
                resumed = outputs is not None
            if outputs is None:
                outputs = operator.run(working, context)
                if not isinstance(outputs, dict):
                    raise TypeError(f"{operator.name}.run() must return a dict")
                missing = set(operator.output_keys) - set(outputs)
                extra = set(outputs) - set(operator.output_keys)
                if missing or extra:
                    raise KeyError(
                        f"{operator.name} output mismatch: missing={sorted(missing)}, "
                        f"extra={sorted(extra)}"
                    )
                context.store.save_stage(
                    video_key=video_key,
                    stage_index=index,
                    stage_name=operator.name,
                    fingerprint=operator.fingerprint,
                    outputs=outputs,
                    stats=operator.stats(outputs),
                )
            working.update(outputs)
            stage_provenance.append(
                {
                    "stage_index": index,
                    "operator": operator.name,
                    "fingerprint": operator.fingerprint,
                    "resumed": resumed,
                }
            )
        result = {
            "schema_version": "video-rsi-result-v1",
            "status": "completed",
            "run_id": context.run_id,
            "pipeline": self.name,
            "pipeline_version": self.version,
            "pipeline_fingerprint": self.fingerprint,
            "video_key": video_key,
            "video": working["video"],
            "source_result": working["source_result"],
            "provenance": stage_provenance,
        }
        missing_results = set(self.result_keys) - set(working)
        if missing_results:
            raise KeyError(
                f"pipeline result is missing keys {sorted(missing_results)}"
            )
        result.update({key: working[key] for key in self.result_keys})
        if isinstance(result.get("pool_candidates"), list):
            producer = {
                "run_id": context.run_id,
                "pipeline": self.name,
                "pipeline_version": self.version,
                "pipeline_fingerprint": self.fingerprint,
                "operator_provenance": stage_provenance,
            }
            result["pool_candidates"] = [
                {
                    **sample,
                    "producer": producer,
                }
                if isinstance(sample, dict)
                else sample
                for sample in result["pool_candidates"]
            ]
        context.store.save_final(video_key, result)
        return result
