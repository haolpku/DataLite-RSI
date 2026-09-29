"""Versioned, serializable pipeline graphs assembled from the operator library."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .core import Operator, Pipeline
from .operator_library import OPERATOR_LIBRARY, OperatorLibrary


@dataclass(frozen=True)
class PipelineNode:
    operator: str
    config: dict[str, Any] = field(default_factory=dict)
    node_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id or self.operator,
            "operator": self.operator,
            "config": self.config,
        }


@dataclass(frozen=True)
class PipelineGraph:
    pipeline_id: str
    version: str
    task_type: str
    nodes: tuple[PipelineNode, ...]
    result_keys: tuple[str, ...] = ("pool_candidates", "feedback_signals")
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "version": self.version,
            "task_type": self.task_type,
            "nodes": [node.to_dict() for node in self.nodes],
            "result_keys": list(self.result_keys),
            "metadata": self.metadata,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PipelineGraph":
        return cls(
            pipeline_id=str(value["pipeline_id"]),
            version=str(value["version"]),
            task_type=str(value.get("task_type", "generic")),
            nodes=tuple(
                PipelineNode(
                    operator=str(node["operator"]),
                    config=dict(node.get("config", {})),
                    node_id=node.get("node_id"),
                )
                for node in value["nodes"]
            ),
            result_keys=tuple(value.get("result_keys", ("pool_candidates", "feedback_signals"))),
            metadata=dict(value.get("metadata", {})),
        )

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            {"pipeline_id": self.pipeline_id, "version": self.version,
             "task_type": self.task_type, "nodes": [n.to_dict() for n in self.nodes],
             "result_keys": self.result_keys, "metadata": self.metadata},
            sort_keys=True, ensure_ascii=False, default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def validate(self, initial_keys: set[str], library: OperatorLibrary = OPERATOR_LIBRARY) -> set[str]:
        keys = set(initial_keys)
        for node in self.nodes:
            spec = library.spec(node.operator)
            missing = set(spec.input_keys) - keys
            if missing:
                raise KeyError(f"{node.operator} requires missing keys {sorted(missing)}")
            duplicate = set(spec.output_keys) & keys
            if duplicate:
                raise KeyError(f"{node.operator} would overwrite keys {sorted(duplicate)}")
            keys.update(spec.output_keys)
        missing_results = set(self.result_keys) - keys
        if missing_results:
            raise KeyError(f"pipeline result is missing keys {sorted(missing_results)}")
        return keys

    def instantiate(
        self,
        *,
        operator_overrides: Mapping[str, Operator] | None = None,
        library: OperatorLibrary = OPERATOR_LIBRARY,
    ) -> Pipeline:
        """Build the existing linear runtime while keeping the graph auditable.

        Backend-bearing operators can be supplied through ``operator_overrides``;
        ordinary operators are constructed from their manifest config.
        """
        overrides = operator_overrides or {}
        operators: list[Operator] = []
        for node in self.nodes:
            op = overrides.get(node.node_id or node.operator)
            if op is None:
                op = overrides.get(node.operator)
            if op is None:
                op = library.resolve(node.operator, **node.config)
            operators.append(op)
        return Pipeline(
            self.pipeline_id,
            operators,
            result_keys=self.result_keys,
            version=self.version,
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target


class PipelineRegistry:
    """Registry and task router for immutable pipeline graph versions."""

    def __init__(self) -> None:
        self._graphs: dict[str, PipelineGraph] = {}

    def register(self, graph: PipelineGraph, *, replace: bool = False) -> PipelineGraph:
        key = f"{graph.pipeline_id}@{graph.version}"
        if key in self._graphs and not replace:
            raise KeyError(f"pipeline {key!r} is already registered")
        self._graphs[key] = graph
        return graph

    def get(self, ref: str) -> PipelineGraph:
        if ref in self._graphs:
            return self._graphs[ref]
        matches = [g for key, g in self._graphs.items() if key.split("@", 1)[0] == ref]
        if len(matches) == 1:
            return matches[0]
        raise KeyError(f"unknown or ambiguous pipeline {ref!r}; available={sorted(self._graphs)}")

    def route(self, task_type: str, *, version: str | None = None) -> PipelineGraph:
        matches = [g for g in self._graphs.values() if g.task_type == task_type]
        if version is not None:
            matches = [g for g in matches if g.version == version]
        if not matches:
            raise KeyError(f"no pipeline registered for task_type={task_type!r}")
        return sorted(matches, key=lambda g: g.version)[-1]

    def list(self, task_type: str | None = None) -> list[dict[str, Any]]:
        graphs = list(self._graphs.values())
        if task_type is not None:
            graphs = [g for g in graphs if g.task_type == task_type]
        return [g.to_dict() for g in sorted(graphs, key=lambda x: (x.task_type, x.pipeline_id, x.version))]

    def load(self, path: str | Path, *, replace: bool = False) -> PipelineGraph:
        graph = PipelineGraph.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        return self.register(graph, replace=replace)

