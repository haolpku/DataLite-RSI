"""Discoverable operator library used by task-specific pipeline builders.

The runtime registry remains the source of truth for backwards compatibility.
This facade adds cataloging, version references and a single place for loading
built-in operators before a graph is compiled.
"""

from __future__ import annotations

from typing import Any

from .registry import OPERATOR_REGISTRY, OperatorRegistry, OperatorSpec


class OperatorLibrary:
    def __init__(self, registry: OperatorRegistry = OPERATOR_REGISTRY) -> None:
        self.registry = registry

    @staticmethod
    def load_builtins() -> None:
        # Imports intentionally happen here so importing the framework does not
        # instantiate model backends or impose a pipeline choice.
        from .operators import events, evidence, production, semantic_refinement, task_candidates, task_mining  # noqa: F401

    def resolve(self, ref: str, **config: Any) -> Any:
        self.load_builtins()
        return self.registry.resolve(ref, **config)

    def spec(self, ref: str) -> OperatorSpec:
        self.load_builtins()
        name = ref.split("@", 1)[0]
        spec = self.registry.spec(name)
        requested = ref.split("@", 1)[1] if "@" in ref else None
        if requested and requested != spec.version:
            raise KeyError(f"operator {ref!r} is not installed (available={spec.ref!r})")
        return spec

    def catalog(self, task_type: str | None = None) -> list[dict[str, Any]]:
        self.load_builtins()
        return [spec.to_dict() for spec in self.registry.specs(task_type)]


OPERATOR_LIBRARY = OperatorLibrary()

