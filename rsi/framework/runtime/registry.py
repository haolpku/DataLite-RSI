"""Adapter contract for the framework-owned DataFlow evolution controller."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

from ..core.contracts import OutputContract, TaskEnvelope
from ..core.pipeline import Pipeline, RunContext

class MethodPlugin(ABC):
    method_id: str
    method_dir: Path
    optional_dependencies: tuple[str, ...] = ()
    artifact_types: tuple[str, ...] = ()

    @property
    def manifest(self) -> dict[str, Any]:
        value = json.loads((self.method_dir / "method.json").read_text(encoding="utf-8"))
        if value.get("id") != self.method_id:
            raise ValueError(f"manifest id does not match plugin {self.method_id}")
        return value

    @property
    def description(self) -> str:
        return str(self.manifest["description"])

    def method_skill_refs(self, task: TaskEnvelope) -> tuple[str, ...]:
        return ()

    def task_skill_refs(self, task: TaskEnvelope) -> tuple[str, ...]:
        return ()

    def load_private_config(self, task: TaskEnvelope) -> Mapping[str, Any]:
        return {}

    def output_contract(self, task: TaskEnvelope) -> OutputContract:
        return task.output_contract

    def role_prompts(self) -> Mapping[str, str]:
        return {}

    def diagnostic_isolation(
        self, task: TaskEnvelope, config: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return {
            "enabled": task.role == "pipeline_diagnostic",
            "project_skill_discovery": False if task.role == "pipeline_diagnostic" else None,
        }

    @abstractmethod
    def build_pipeline(
        self,
        task: TaskEnvelope,
        config: Mapping[str, Any],
        resources: Mapping[str, Any],
    ) -> Pipeline:
        raise NotImplementedError

    @abstractmethod
    def native_runner(self, task: TaskEnvelope, context: RunContext) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def evaluate_feedback(
        self, task: TaskEnvelope, output: Mapping[str, Any], context: RunContext
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def acceptance_logic(
        self, task: TaskEnvelope, feedback: Mapping[str, Any], context: RunContext
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    def failure_handling(self, error: Exception, context: RunContext) -> Mapping[str, Any]:
        return {"status": "failed", "error_type": type(error).__name__, "message": str(error)}
