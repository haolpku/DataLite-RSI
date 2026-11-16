"""DataFlow-centered entry point for text and multimodal evolution tasks."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Mapping

from .agents import ROLES
from ..core.contracts import TaskEnvelope, fingerprint
from ..core.pipeline import RunContext
from .routing import load_task_config
from .skills import SkillRegistry
from ..io.storage import StorageBundle


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
SKILL_REGISTRY = SkillRegistry(
    FRAMEWORK_ROOT / "skills", allowed_method_ids=frozenset({"dataflow-evolver"})
)


class FrameworkRegistry:
    """The active runtime has one evolution controller across modalities."""

    def ids(self) -> tuple[str, ...]:
        return ("dataflow-evolver",)

    def get(self, method_id: str):
        if method_id != "dataflow-evolver":
            raise ValueError(f"unknown method_id {method_id!r}; expected 'dataflow-evolver'")
        from ..evolution.runner import DataFlowPlugin

        return DataFlowPlugin()


METHOD_REGISTRY = FrameworkRegistry()


def run(
    task_config: TaskEnvelope | Mapping[str, Any] | str | Path,
    *,
    resources: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    task = load_task_config(task_config)
    plugin = METHOD_REGISTRY.get(task.method_id)
    return run_with_adapter(
        task, plugin, SKILL_REGISTRY,
        resources=resources, run_id=run_id, resume=resume,
    )


def run_with_adapter(
    task_config: TaskEnvelope | Mapping[str, Any] | str | Path,
    plugin: Any,
    skill_registry: SkillRegistry,
    *,
    resources: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Execute an explicitly supplied adapter without discovering source trees.

    This narrow seam also lets future image/video evaluators reuse the run
    lifecycle while keeping their domain logic outside the core framework.
    """
    task = load_task_config(task_config)
    if task.role not in ROLES:
        raise ValueError(f"unknown agent role {task.role!r}")
    if plugin.method_id != task.method_id:
        raise ValueError("adapter method_id differs from task method_id")
    private_config = plugin.load_private_config(task)
    pipeline = plugin.build_pipeline(task, private_config, resources or {})
    initial_state = dict(task.data)
    spec = pipeline.compile(initial_state.keys())
    contract = plugin.output_contract(task)
    missing_contract_keys = set(contract.required_keys) - set(spec.result_keys)
    if missing_contract_keys:
        raise ValueError(
            f"method output contract is not produced by pipeline: {sorted(missing_contract_keys)}"
        )
    method_refs = plugin.method_skill_refs(task)
    task_refs = plugin.task_skill_refs(task)
    bundle = skill_registry.bundle(task.method_id, method_refs, task_refs)
    actual_run_id = run_id or f"{task.task_id}-{uuid.uuid4().hex[:12]}"
    storage = StorageBundle(task.resolved_workspace(REPOSITORY_ROOT), actual_run_id)
    context = RunContext(task, actual_run_id, storage, resume, dict(resources or {}))
    context.resources["pipeline_spec"] = spec
    context.resources["skill_bundle_fingerprint"] = fingerprint(
        {skill.ref: skill.fingerprint for skill in bundle}
    )
    context.resources["method_skill_contexts"] = {
        skill.ref: skill.path.read_text(encoding="utf-8")
        for skill in bundle if skill.kind == "method"
    }
    context.resources["task_skill_contexts"] = {
        skill.path.stem: skill.path.read_text(encoding="utf-8")
        for skill in bundle if skill.kind == "task"
    }
    raw_provider_config = task.metadata.get("provider_config", {})
    if not isinstance(raw_provider_config, Mapping):
        raise ValueError("metadata.provider_config must be an object")
    provider_config = {
        key: raw_provider_config[key]
        for key in ("model", "backend", "binary", "read_only", "timeout_sec", "max_turns")
        if key in raw_provider_config
    }
    diagnostic_isolation = dict(plugin.diagnostic_isolation(task, private_config))
    manifest: dict[str, Any] = {
        "schema_version": "0.1",
        "status": "running",
        "run_id": actual_run_id,
        "task_id": task.task_id,
        "method_id": task.method_id,
        "method_manifest": plugin.manifest,
        "pipeline": spec.to_dict(),
        "pipeline_fingerprint": spec.fingerprint,
        "input_fingerprint": fingerprint(initial_state),
        "input_contract": task.input_contract.to_dict() if task.input_contract else None,
        "method_skill_refs": list(method_refs),
        "task_skill_refs": list(task_refs),
        "skill_fingerprints": {skill.ref: skill.fingerprint for skill in bundle},
        "provider": task.provider,
        "provider_configuration": provider_config,
        "role": task.role,
        "diagnostic_isolation": diagnostic_isolation,
        "artifact_types": sorted(set(plugin.artifact_types) | set(contract.artifact_types)),
    }
    storage.run.write_manifest(manifest)
    try:
        result = pipeline.run(initial_state, context)
        contract.validate(result.output)
        context.resources["pipeline_output"] = result.output
        context.resources["pipeline_state"] = result.state
        method_artifacts = dict(plugin.native_runner(task, context))
        feedback = dict(plugin.evaluate_feedback(task, result.output, context))
        acceptance = dict(plugin.acceptance_logic(task, feedback, context))
        storage.records.write_json("output", result.output)
        storage.records.write_json("feedback", feedback)
        storage.records.write_json("acceptance", acceptance)
        manifest.update(
            {
                "status": "completed",
                "output_keys": sorted(result.output),
                "feedback": feedback,
                "acceptance": acceptance,
                "method_artifacts": method_artifacts,
                "execution_observation": str(result.observation_path.relative_to(storage.root)),
            }
        )
    except Exception as exc:
        failure = dict(plugin.failure_handling(exc, context))
        manifest.update({"status": "failed", "failure": failure})
        storage.run.append_failure(failure)
    storage.run.write_manifest(manifest)
    return manifest
