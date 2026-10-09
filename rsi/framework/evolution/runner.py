"""Evolution-method adapter to the shared task and run lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from rsi.framework.core.contracts import ArtifactRef, TaskEnvelope, fingerprint
from rsi.framework.core.operator import Operator
from rsi.framework.core.pipeline import Pipeline, RunContext
from rsi.framework.runtime.registry import MethodPlugin
from rsi.framework.evolution.providers.agent_sdk import default_diagnostic_system_prompt_append
from .media import resolve_local_input_artifact


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOTS = {
    "codex": ".agents/skills/dataflow-evolver-pipeline",
    "claude": ".claude/skills/dataflow-evolver-pipeline",
    "opencode": ".opencode/skills/dataflow-evolver-pipeline",
}
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ENV_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::[^}]*)?\}")
_CONFIG_OVERRIDES = {
    ("project", "workspace_dir"),
    ("project", "input_path"),
    ("project", "run_name"),
    ("agent", "backend"),
}


def _missing_required_env(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    """Find unset, no-default placeholders that survive task-owned overrides."""
    if path in _CONFIG_OVERRIDES:
        return []
    if isinstance(value, Mapping):
        return [
            name
            for key, item in value.items()
            for name in _missing_required_env(item, (*path, str(key)))
        ]
    if isinstance(value, list):
        return [name for item in value for name in _missing_required_env(item, path)]
    if isinstance(value, str):
        return [name for name in _ENV_REF.findall(value) if name not in os.environ]
    return []


def _load_native():
    from .controller import build_loop
    from .utils.config import Config, load_config

    return build_loop, Config, load_config


def _portable_path(path: str | Path | None, root: Path) -> str | None:
    if not path:
        return None
    resolved = Path(path).resolve()
    return resolved.relative_to(root.resolve()).as_posix() if resolved.is_relative_to(root.resolve()) else None


def _input_file_identity(path: str | Path) -> dict[str, Any]:
    """Bind the outer checkpoint to the fixed corpus contents, not only its path."""
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": source.stat().st_size, "sha256": digest.hexdigest()}


def _config_environment_identity(config_text: str) -> str:
    """Hash resolved settings without exposing credentials in a manifest."""
    names = sorted(set(_ENV_PLACEHOLDER.findall(config_text)))
    return fingerprint({name: os.environ.get(name) for name in names})


def _authoring_task_description(task: TaskEnvelope) -> str:
    """Expose non-secret task routing metadata to the generated-pipeline agent."""
    profile = task.metadata.get("pipeline_skill_profile")
    if not isinstance(profile, str) or not profile.strip():
        return task.objective
    contract = task.input_contract.to_dict() if task.input_contract else None
    return (
        f"{task.objective}\n\n"
        f"Pipeline skill profile: {profile.strip()}\n"
        f"Input contract: {json.dumps(contract, ensure_ascii=False, sort_keys=True)}\n"
        "Apply the matching provider-native skill reference before authoring the pipeline."
    )


def _skill_tree_identity(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _mirror_artifact_refs(
    value: Any, source_run: Path, input_entry: Path, context: RunContext, location: str
) -> Any:
    """Rehome generated media so final records resolve in the outer run store."""
    if isinstance(value, Mapping) and {"kind", "uri", "media_type"} <= set(value):
        ref = ArtifactRef.from_mapping(value)
        name = f"dataflow/{fingerprint(location)[:20]}"
        if ref.kind == "video_reference":
            uri = ref.uri
            if "://" not in uri:
                path = Path(uri).expanduser()
                uri = str((path if path.is_absolute() else input_entry.parent / path).resolve())
            return context.storage.artifacts.put_video_reference(
                name, uri, metadata=ref.metadata, media_type=ref.media_type,
                sha256=ref.sha256,
            ).to_dict()
        uri = Path(ref.uri)
        if uri.parts and uri.parts[0] == "blobs":
            source = (source_run / uri).resolve()
            if not source.is_relative_to((source_run / "blobs").resolve()):
                raise ValueError(f"generated artifact reference is unsafe: {ref.uri}")
        elif "://" not in ref.uri:
            source = resolve_local_input_artifact(input_entry, ref)
        else:
            return ref.to_dict()
        imported = context.storage.artifacts.put_file(
            name, source, kind=ref.kind, media_type=ref.media_type,
            metadata=ref.metadata,
        )
        if ref.sha256 and imported.sha256 != ref.sha256:
            raise ValueError(f"artifact checksum mismatch at {location}")
        return imported.to_dict()
    if isinstance(value, Mapping):
        return {
            key: _mirror_artifact_refs(item, source_run, input_entry, context, f"{location}/{key}")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _mirror_artifact_refs(item, source_run, input_entry, context, f"{location}/{index}")
            for index, item in enumerate(value)
        ]
    return value


class _EvolutionOperator(Operator):
    name = "dataflow_evolution"
    version = "datalite-multimodal-1"
    input_keys = ("input_path",)
    output_keys = ("evolution_result",)

    def __init__(self, task: TaskEnvelope, config_path: Path) -> None:
        self.task = task
        self.config_path = config_path

    def config(self) -> dict[str, Any]:
        config_text = self.config_path.read_text(encoding="utf-8")
        return {
            "baseline_commit": "ba6e489524bbf5835a06cccad555c5182e4ce4f4",
            "config_ref": self.config_path.relative_to(FRAMEWORK_ROOT / "configs").as_posix(),
            "config_fingerprint": fingerprint(config_text),
            "environment_fingerprint": _config_environment_identity(config_text),
            "fixed_input": _input_file_identity(self.task.data["input_path"]),
            "input_contract": self.task.input_contract.to_dict() if self.task.input_contract else None,
        }

    def run(self, state: dict[str, Any], context: RunContext) -> Mapping[str, Any]:
        build_loop, config_cls, load_config = _load_native()
        import yaml

        raw_config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        missing_env = _missing_required_env(raw_config)
        if missing_env:
            raise ValueError(f"DataFlow private config is missing environment variables: {sorted(set(missing_env))}")
        config = load_config(self.config_path).to_dict()
        project = dict(config.get("project") or {})
        native_workspace = context.storage.root / "native"
        project.update({
            "workspace_dir": str(native_workspace),
            "input_path": str(Path(state["input_path"]).expanduser().resolve()),
            "run_name": "evolution",
        })
        config["project"] = project
        config.setdefault("agent", {})["backend"] = self.task.provider
        skill_ref = SKILL_ROOTS[self.task.provider]
        source = FRAMEWORK_ROOT / "skills" / "providers" / self.task.provider / "dataflow-evolver-pipeline"
        destination = native_workspace / skill_ref
        if not source.is_dir():
            raise FileNotFoundError(f"DataFlow provider skill is missing: {skill_ref}")
        if destination.exists():
            if (
                destination.is_symlink()
                or not destination.is_dir()
                or _skill_tree_identity(destination) != _skill_tree_identity(source)
            ):
                raise RuntimeError(
                    "run-local authoring skill differs from the framework source; "
                    "choose a fresh run_id"
                )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination)
        loop_options: dict[str, Any] = {}
        if self.task.input_contract is not None:
            loop_options["input_contract"] = self.task.input_contract
        if "candidate_evaluator" in context.resources:
            loop_options["candidate_evaluator"] = context.resources["candidate_evaluator"]
        if "pipeline_agent" in context.resources:
            loop_options["pipeline_agent_override"] = context.resources["pipeline_agent"]
        try:
            loop = build_loop(config_cls(config), **loop_options)
        except ImportError as exc:
            raise RuntimeError(f"the evolution loop is missing an optional runtime dependency: {exc}") from exc
        loop.task.task_description = _authoring_task_description(self.task)
        best = loop.run()
        root = context.storage.root
        candidates = []
        for candidate in loop.candidates:
            record = {
                "iteration": candidate.iteration,
                "accepted_as_incumbent": candidate.accepted_as_incumbent,
                "score": candidate.score,
                "passed": bool(candidate.review and candidate.review.passed),
                "review": asdict(candidate.review) if candidate.review else None,
                "execution_repairs": candidate.execution_repairs,
                "failure_stage": candidate.failure_stage,
                "execution_error": candidate.execution_error,
                "execution_observation": _portable_path(candidate.execution_observation_path, root),
                "pipeline_diagnostic": candidate.pipeline_diagnostic,
            }
            candidates.append(record)
            if candidate.pipeline_diagnostic is not None:
                context.storage.run.write_diagnostic(
                    f"dataflow-iteration-{candidate.iteration + 1:03d}",
                    candidate.pipeline_diagnostic,
                )
        context.storage.records.write_jsonl("dataflow/candidates", candidates)
        if best and best.dataset_path:
            dataset_path = Path(best.dataset_path)
            generated_run = best.iteration_dir / "runs" / "generated"
            input_entry = Path(state["input_path"]).expanduser().resolve()
            def rows():
                with dataset_path.open(encoding="utf-8") as handle:
                    for number, line in enumerate(handle, start=1):
                        if line.strip():
                            value = json.loads(line)
                            if not isinstance(value, dict):
                                raise ValueError("DataFlow final dataset must contain JSON objects")
                            yield _mirror_artifact_refs(
                                value, generated_run, input_entry, context, f"row/{number}"
                            )
            context.storage.records.write_jsonl("dataflow/final_dataset", rows())
        return {
            "evolution_result": {
                "best_iteration": best.iteration if best else None,
                "best_score": best.score if best else None,
                "best_review": asdict(best.review) if best and best.review else None,
                "dataset_ref": "records/dataflow/final_dataset.jsonl" if best and best.dataset_path else None,
                "candidate_history_ref": "records/dataflow/candidates.jsonl",
                "candidates": candidates,
                "native_run_ref": "native/runs/evolution/",
            }
        }


class DataFlowPlugin(MethodPlugin):
    method_id = "dataflow-evolver"
    method_dir = FRAMEWORK_ROOT
    artifact_types = ("structured_records", "execution_observation", "pipeline_diagnostic")
    optional_dependencies = ("openai", "requests", "PyYAML")

    def diagnostic_isolation(
        self, task: TaskEnvelope, config: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        try:
            from .utils.config import load_config
        except ImportError:
            return {"enabled": None, "project_skill_discovery": False, "read_only": True}
        diagnostic = load_config(config["config_path"]).get("pipeline_diagnostic")
        raw_enabled = diagnostic.get("enabled", False) if diagnostic is not None else False
        enabled = (
            raw_enabled.strip().lower() in {"1", "true", "yes", "on"}
            if isinstance(raw_enabled, str) else bool(raw_enabled)
        )
        isolation = {
            "enabled": enabled,
            "read_only": enabled,
            "isolated_cwd": enabled,
            "project_skill_discovery": False if enabled else None,
            "separate_role_prompt": enabled,
            "score_input": False,
        }
        if enabled:
            backend = str(diagnostic.get("backend", "codex") or "codex").strip().lower()
            isolation.update({
                "role": "pipeline_diagnostic",
                "prompt_source": "builtin",
                "prompt_fingerprint": fingerprint(default_diagnostic_system_prompt_append(backend)),
            })
        return isolation

    def method_skill_refs(self, task: TaskEnvelope) -> tuple[str, ...]:
        if task.role != "pipeline_builder":
            raise ValueError("DataFlow evolution task must use pipeline_builder role")
        if task.provider not in SKILL_ROOTS:
            raise ValueError("the evolution loop requires codex, claude, or opencode provider")
        return (f"providers/{task.provider}/dataflow-evolver-pipeline/SKILL.md",)

    def load_private_config(self, task: TaskEnvelope) -> Mapping[str, Any]:
        ref = task.metadata.get("method_config_ref")
        if not isinstance(ref, str) or not ref:
            raise ValueError("DataFlow task metadata requires method_config_ref")
        root = (FRAMEWORK_ROOT / "configs").resolve()
        path = (root / ref).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"DataFlow private config must be in {root.name}/: {ref!r}")
        return {"config_path": path}

    def build_pipeline(
        self, task: TaskEnvelope, config: Mapping[str, Any], resources: Mapping[str, Any]
    ) -> Pipeline:
        if task.provider not in SKILL_ROOTS:
            raise ValueError("the evolution loop requires codex, claude, or opencode provider")
        if task.role != "pipeline_builder":
            raise ValueError("DataFlow evolution task must use pipeline_builder role")
        if not task.data.get("input_path"):
            raise ValueError("DataFlow task data requires input_path to a fixed raw corpus")
        if (
            task.input_contract is not None
            and set(task.input_contract.modalities) & {"image", "video"}
            and "candidate_evaluator" not in resources
        ):
            raise ValueError(
                "image/video DataFlow tasks require a candidate_evaluator; "
                "text ReviewAgent cannot score visual artifacts"
            )
        return Pipeline("dataflow_evolution", (_EvolutionOperator(task, config["config_path"]),),
                        result_keys=("evolution_result",))

    def native_runner(self, task: TaskEnvelope, context: RunContext) -> Mapping[str, Any]:
        result = context.resources["pipeline_output"]["evolution_result"]
        return {
            "dataset": result["dataset_ref"],
            "candidate_history": result["candidate_history_ref"],
            "native_run": result["native_run_ref"],
        }

    def evaluate_feedback(
        self, task: TaskEnvelope, output: Mapping[str, Any], context: RunContext
    ) -> Mapping[str, Any]:
        result = output["evolution_result"]
        return {
            "review": result["best_review"],
            "review_score": result["best_score"],
            "candidates": result["candidates"],
        }

    def acceptance_logic(
        self, task: TaskEnvelope, feedback: Mapping[str, Any], context: RunContext
    ) -> Mapping[str, Any]:
        # The native loop already selected by (ReviewAgent passed, review_score).
        # Diagnostics stay outside that comparison.
        candidates = feedback["candidates"]
        accepted = [item for item in candidates if item["accepted_as_incumbent"]]
        return {
            "decision": "native_best_so_far",
            "accepted_iterations": [item["iteration"] for item in accepted],
            "best_score": feedback["review_score"],
            "diagnostic_used_for_score": False,
        }


def create_plugin() -> DataFlowPlugin:
    return DataFlowPlugin()
