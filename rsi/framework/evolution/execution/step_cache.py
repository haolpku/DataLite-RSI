"""Framework-owned logical step cache manifests and reuse plans."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rsi.framework.evolution.models import PipelineConfig

CACHE_PLAN_ENV = "DF_FRAMEWORK_STEP_CACHE_PLAN"
MANIFEST_FILENAME = "step_manifest.json"
REUSE_PLAN_FILENAME = "step_reuse_plan.json"
SCHEMA_VERSION = 3


@dataclass(frozen=True)
class FrameworkStepCache:
    """Mechanical pipeline adapter for a reuse decision already made by the framework."""

    entry_path: str
    prefix_count: int
    operator_names: tuple[str, ...]

    @classmethod
    def from_env(
        cls,
        *,
        raw_entry_path: str,
        operator_names: list[str] | tuple[str, ...],
    ) -> "FrameworkStepCache":
        names = tuple(str(name) for name in operator_names)
        raw_plan = os.environ.get(CACHE_PLAN_ENV, "").strip()
        if not raw_plan:
            return cls(str(raw_entry_path), 0, names)
        try:
            plan = json.loads(raw_plan)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid {CACHE_PLAN_ENV}: {exc}") from exc
        if int(plan.get("schema_version", 0)) != SCHEMA_VERSION:
            raise RuntimeError(f"unsupported framework step-cache schema: {plan.get('schema_version')}")
        planned_names = tuple(str(name) for name in plan.get("operator_names", []))
        if planned_names != names:
            raise RuntimeError(
                "framework step-cache operator order mismatch: "
                f"plan={planned_names!r} pipeline={names!r}"
            )
        prefix_count = int(plan.get("prefix_count", 0))
        if prefix_count < 0 or prefix_count > len(names):
            raise RuntimeError(f"invalid framework prefix_count={prefix_count}")
        entry_path = str(plan.get("entry_path") or raw_entry_path)
        if prefix_count and not Path(entry_path).is_file():
            raise RuntimeError(f"framework cache input does not exist: {entry_path}")
        return cls(entry_path, prefix_count, names)

    def should_run(self, logical_index: int) -> bool:
        return int(logical_index) >= self.prefix_count


@dataclass
class StepReusePlan:
    operator_names: list[str]
    prefix_count: int
    entry_path: str
    raw_entry_path: str
    parent_manifest_path: str | None = None
    reason: str = "raw input"
    reused_steps: list[dict[str, Any]] | None = None

    @property
    def fully_reused(self) -> bool:
        return bool(self.operator_names) and self.prefix_count == len(self.operator_names)

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "operator_names": self.operator_names,
            "prefix_count": self.prefix_count,
            "entry_path": self.entry_path,
            "raw_entry_path": self.raw_entry_path,
            "parent_manifest_path": self.parent_manifest_path,
            "reason": self.reason,
        }

    def env(self) -> dict[str, str]:
        return {CACHE_PLAN_ENV: json.dumps(self.payload(), ensure_ascii=False, sort_keys=True)}


def prepare_step_reuse(
    config: PipelineConfig,
    iteration_dir: Path,
    *,
    parent_iteration_dir: Path | None,
    raw_entry_path: str,
    pipeline_filename: str,
) -> StepReusePlan:
    """Resolve the reusable logical prefix; generated code never chooses an artifact."""
    iteration_dir = iteration_dir.resolve()
    operator_names = [_operator_name(item) for item in config.operators]
    raw_path = str(Path(raw_entry_path).resolve())
    plan = StepReusePlan(operator_names, 0, raw_path, raw_path)
    pipeline_path = iteration_dir / pipeline_filename
    if not operator_names:
        plan.reason = "decision.json has no ordered operators"
        return _persist_plan(iteration_dir, plan)
    if not _supports_framework_cache(pipeline_path):
        plan.reason = "pipeline does not use FrameworkStepCache; safe raw-input fallback"
        return _persist_plan(iteration_dir, plan)
    if parent_iteration_dir is None:
        plan.reason = "no parent incumbent"
        return _persist_plan(iteration_dir, plan)
    parent_manifest_path = parent_iteration_dir.resolve() / MANIFEST_FILENAME
    parent = _read_json(parent_manifest_path)
    if not parent or int(parent.get("schema_version", 0)) != SCHEMA_VERSION:
        plan.reason = "parent step manifest unavailable"
        return _persist_plan(iteration_dir, plan)
    current_raw = _file_identity(Path(raw_path))
    if current_raw != parent.get("raw_input"):
        plan.reason = "raw input identity changed"
        return _persist_plan(iteration_dir, plan)

    current_steps = operator_descriptors(config, iteration_dir, pipeline_filename)
    parent_steps = list(parent.get("steps", []))
    prefix = 0
    reused: list[dict[str, Any]] = []
    for current, previous in zip(current_steps, parent_steps):
        artifact = Path(str(previous.get("output_artifact", "")))
        if (
            not current.get("cacheable")
            or not previous.get("cacheable")
            or current.get("logical_index") != previous.get("logical_index")
            or current.get("operator_name") != previous.get("operator_name")
            or current.get("fingerprint") != previous.get("fingerprint")
            # A blobs/... URI is scoped to the run that wrote it. Reusing that
            # JSONL in a new run would leave later operators with a dangling ref.
            # Older manifests lack this field and safely fall back to recompute.
            or previous.get("contains_local_blob_refs") is not False
            or not artifact.is_file()
            or artifact.stat().st_size <= 0
            or _sha256_file(artifact) != previous.get("output_sha256")
        ):
            break
        prefix += 1
        reused.append(dict(previous))

    if prefix:
        plan.prefix_count = prefix
        plan.entry_path = str(reused[-1]["output_artifact"])
        plan.parent_manifest_path = str(parent_manifest_path)
        plan.reason = f"verified identical logical prefix of {prefix} operator(s)"
        plan.reused_steps = reused
    else:
        plan.reason = "no verified identical logical prefix"
    return _persist_plan(iteration_dir, plan)


def operator_descriptors(
    config: PipelineConfig,
    iteration_dir: Path,
    pipeline_filename: str,
) -> list[dict[str, Any]]:
    pipeline_path = iteration_dir / pipeline_filename
    init_calls = _operator_init_calls(pipeline_path)
    local_sources = _local_operator_sources(iteration_dir, pipeline_path, {_operator_name(item) for item in config.operators})
    descriptors: list[dict[str, Any]] = []
    for index, item in enumerate(config.operators):
        name = _operator_name(item)
        payload = {
            "operator_name": name,
            "params": item.get("params", {}) if isinstance(item, dict) else {},
            "init_call_ast": init_calls.get(name, []),
            "source": local_sources.get(name),
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        descriptors.append({
            "logical_index": index,
            "operator_name": name,
            "fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "fingerprint_components": payload,
            "cacheable": payload["source"] is not None,
        })
    return descriptors


def write_step_manifest(
    config: PipelineConfig,
    iteration_dir: Path,
    *,
    raw_entry_path: str,
    pipeline_filename: str,
    plan: StepReusePlan,
    executed_artifacts: list[Path],
    final_dataset_path: str,
) -> Path:
    descriptors = operator_descriptors(config, iteration_dir, pipeline_filename)
    expected_suffix = max(0, len(descriptors) - plan.prefix_count)
    if len(executed_artifacts) != expected_suffix:
        raise ValueError(
            "framework step artifact count mismatch: "
            f"expected {expected_suffix}, found {len(executed_artifacts)}"
        )
    steps: list[dict[str, Any]] = []
    reused = plan.reused_steps or []
    for index in range(plan.prefix_count):
        item = dict(reused[index])
        item["reused"] = True
        item["reused_from_manifest"] = plan.parent_manifest_path
        steps.append(item)
    for offset, artifact in enumerate(executed_artifacts):
        logical_index = plan.prefix_count + offset
        item = dict(descriptors[logical_index])
        item.update(_artifact_summary(artifact))
        item["reused"] = False
        steps.append(item)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "raw_input": _file_identity(Path(raw_entry_path).resolve()),
        "reuse": plan.payload(),
        "steps": steps,
        "final_dataset_path": str(Path(final_dataset_path).resolve()),
    }
    path = iteration_dir / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _persist_plan(iteration_dir: Path, plan: StepReusePlan) -> StepReusePlan:
    path = iteration_dir / REUSE_PLAN_FILENAME
    path.write_text(json.dumps(plan.payload(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return plan


def _supports_framework_cache(path: Path) -> bool:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return "run_generated_pipeline" in source or (
        "FrameworkStepCache" in source and ".should_run(" in source
    )


def _operator_name(item: dict[str, Any] | Any) -> str:
    if isinstance(item, dict):
        return str(item.get("name", "?") or "?")
    return str(item)


def _operator_init_calls(path: Path) -> dict[str, list[str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return {}
    result: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name:
            result.setdefault(name, []).append(ast.dump(node, include_attributes=False))
    return result


#: Pipeline-level constructions whose settings change what every operator reads
#: or writes, so they belong in each operator's identity even though they live
#: in the pipeline class.
_RUNTIME_CONSTRUCTION_NAMES = (
    "FileStorage",
    "BatchedFileStorage",
    "StreamBatchedFileStorage",
    "PipelineLLMServing",
)


def _local_operator_sources(
    iteration_dir: Path, pipeline_path: Path, stage_names: set[str]
) -> dict[str, Any]:
    """Fingerprint local class bodies and shared helpers without coupling sibling stages.

    An unresolved or ambiguous class is deliberately non-cacheable. External
    dependency behavior remains the author's responsibility to version.

    The pipeline class and the operator-name manifest are excluded from the
    shared body: they are wiring, and they necessarily change whenever any
    operator is added or renamed, which would make every prefix operator
    non-reusable. What they actually contribute to behavior is captured
    separately -- operator constructor arguments through ``init_call_ast``, and
    storage/serving settings through ``runtime_calls``. The ordered operator
    identity itself is validated by the framework, not by this fingerprint.
    """
    paths = [pipeline_path, *sorted((iteration_dir / "operators").rglob("*.py"))]
    classes: dict[str, list[tuple[str, str]]] = {}
    support: list[tuple[str, str]] = []
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            return {}
        relative = path.relative_to(iteration_dir).as_posix()
        shared_nodes = [
            node for node in tree.body
            if not (isinstance(node, ast.ClassDef) and node.name in stage_names)
            and not _is_main_guard(node)
            and not _is_pipeline_class(node)
            and not _is_operator_name_manifest(node, stage_names)
        ]
        support.append((relative, ast.dump(ast.Module(body=shared_nodes, type_ignores=[]), include_attributes=False)))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.setdefault(node.name, []).append(
                    (relative, ast.dump(node, include_attributes=False))
                )
    shared = sorted(support)
    runtime = _runtime_construction_calls(pipeline_path)
    wiring = _per_operator_wiring(pipeline_path)
    return {
        name: {
            "class": definitions[0],
            "shared": shared,
            "runtime": runtime,
            "wiring": wiring.get(name, []),
        }
        for name, definitions in classes.items() if len(definitions) == 1
    }


def _is_pipeline_class(node: ast.AST) -> bool:
    """Is this the generated pipeline class rather than an operator or helper?"""
    if not isinstance(node, ast.ClassDef):
        return False
    return any(
        _call_name(base) in {"PipelineABC", "BatchedPipelineABC", "StreamBatchedPipelineABC"}
        for base in node.bases
    )


def _is_operator_name_manifest(node: ast.AST, stage_names: set[str]) -> bool:
    """Is this a module-level restatement of the ordered operator names?"""
    if not isinstance(node, ast.Assign) or not isinstance(node.value, (ast.List, ast.Tuple)):
        return False
    values = [
        element.value for element in node.value.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    ]
    return bool(values) and len(values) == len(node.value.elts) and set(values) <= stage_names


def _runtime_construction_calls(pipeline_path: Path) -> list[tuple[str, list[str]]]:
    """Dump the storage and serving constructions the pipeline hands operators."""
    calls = _operator_init_calls(pipeline_path)
    return [
        (name, calls[name])
        for name in _RUNTIME_CONSTRUCTION_NAMES
        if name in calls
    ]


def _per_operator_wiring(pipeline_path: Path) -> dict[str, list[str]]:
    """Attribute each operator's own ``run(...)`` call to that operator.

    The pipeline class is excluded from the shared body so that renaming one
    operator does not invalidate its siblings, but a changed ``run(...)``
    argument is still a behavior change. Resolve ``self.attr = ClassName(...)``
    assignments, then collect the ``self.attr.run(...)`` calls for each class.
    """
    try:
        tree = ast.parse(pipeline_path.read_text(encoding="utf-8"), filename=str(pipeline_path))
    except (OSError, SyntaxError):
        return {}
    attribute_to_class: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        class_name = _call_name(node.value.func)
        if not class_name:
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                attribute_to_class[target.attr] = class_name
    wiring: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if not (
            isinstance(owner, ast.Attribute)
            and isinstance(owner.value, ast.Name)
            and owner.value.id == "self"
        ):
            continue
        class_name = attribute_to_class.get(owner.attr)
        if class_name:
            wiring.setdefault(class_name, []).append(
                ast.dump(node, include_attributes=False)
            )
    return {name: sorted(calls) for name, calls in wiring.items()}


def _is_main_guard(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "__main__"
    )


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "sha256": _sha256_file(path)}


def _artifact_summary(path: Path) -> dict[str, Any]:
    rows = 0
    fields: list[str] = []
    contains_local_blob_refs = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            if not fields or (not contains_local_blob_refs and "blobs/" in line):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    value = None
                if isinstance(value, dict):
                    if not fields:
                        fields = sorted(str(key) for key in value)
                    if not contains_local_blob_refs and _contains_local_blob_ref(value):
                        contains_local_blob_refs = True
    return {
        "output_artifact": str(path.resolve()),
        "output_rows": rows,
        "output_fields": fields,
        "output_sha256": _sha256_file(path),
        "contains_local_blob_refs": contains_local_blob_refs,
    }


def _contains_local_blob_ref(value: Any) -> bool:
    if isinstance(value, dict):
        if (
            value.get("kind") in {"image", "binary", "blob"}
            and isinstance(value.get("uri"), str)
            and value["uri"].startswith("blobs/")
        ):
            return True
        return any(_contains_local_blob_ref(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_local_blob_ref(item) for item in value)
    return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None
