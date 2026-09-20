#!/usr/bin/env python3
"""Validate DataLite-RSI contribution manifests without third-party packages."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "0.1"
TRACKS = {"llm", "multimodal", "generative"}
KEBAB_CASE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


@dataclass(frozen=True)
class ManifestSpec:
    kind: str
    pattern: str
    id_field: str
    required: tuple[str, ...]


SPECS = (
    ManifestSpec(
        "benchmark",
        "benchmarks/*/benchmark.json",
        "id",
        (
            "schema_version",
            "id",
            "name",
            "version",
            "track",
            "description",
            "task_types",
            "dataset",
            "evaluator",
            "metrics",
            "license",
        ),
    ),
    ManifestSpec(
        "dataset",
        "datasets/*/dataset.json",
        "id",
        (
            "schema_version",
            "id",
            "name",
            "description",
            "huggingface",
            "modalities",
            "formats",
            "splits",
            "license",
            "access",
            "contains_pii",
            "provenance_url",
        ),
    ),
    ManifestSpec(
        "result",
        "results/submissions/*/result.json",
        "submission_id",
        (
            "schema_version",
            "submission_id",
            "status",
            "track",
            "benchmark",
            "model",
            "method_id",
            "metrics",
            "settings",
            "reproducibility",
        ),
    ),
    ManifestSpec(
        "method",
        "rsi/methods/*/method.json",
        "id",
        (
            "schema_version",
            "id",
            "name",
            "version",
            "description",
            "tracks",
            "entrypoint",
            "modifies",
            "feedback_signal",
            "stopping_rule",
            "human_intervention",
            "license",
        ),
    ),
)


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_nonempty_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value)


def _require_mapping(
    data: dict[str, Any], field: str, required: Iterable[str], errors: list[str]
) -> dict[str, Any]:
    value = data.get(field)
    if not isinstance(value, dict):
        errors.append(f"{field}: expected an object")
        return {}
    for key in required:
        if key not in value:
            errors.append(f"{field}.{key}: required field is missing")
    return value


def _check_common(data: dict[str, Any], spec: ManifestSpec, errors: list[str]) -> None:
    for field in spec.required:
        if field not in data:
            errors.append(f"{field}: required field is missing")

    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"schema_version: expected {SCHEMA_VERSION!r}, got "
            f"{data.get('schema_version')!r}"
        )

    manifest_id = data.get(spec.id_field)
    if not _is_nonempty_string(manifest_id) or not KEBAB_CASE.fullmatch(manifest_id):
        errors.append(f"{spec.id_field}: expected lowercase kebab-case")

    if "version" in data and (
        not _is_nonempty_string(data["version"])
        or not SEMVER.fullmatch(data["version"])
    ):
        errors.append("version: expected semantic version such as 0.1.0")


def _check_track(track: Any, field: str, errors: list[str]) -> None:
    if not isinstance(track, str) or track not in TRACKS:
        errors.append(f"{field}: expected one of {sorted(TRACKS)}, got {track!r}")


def _check_benchmark(data: dict[str, Any], errors: list[str]) -> None:
    _check_track(data.get("track"), "track", errors)
    if not _is_nonempty_list(data.get("task_types")):
        errors.append("task_types: expected a non-empty array")

    dataset = _require_mapping(data, "dataset", ("registry_id", "revision"), errors)
    if dataset and not _is_nonempty_string(dataset.get("registry_id")):
        errors.append("dataset.registry_id: expected a non-empty string")
    if dataset and not _is_nonempty_string(dataset.get("revision")):
        errors.append("dataset.revision: expected an immutable revision")

    evaluator = _require_mapping(data, "evaluator", ("entrypoint",), errors)
    if evaluator and not _is_nonempty_string(evaluator.get("entrypoint")):
        errors.append("evaluator.entrypoint: expected a non-empty string")

    metrics = data.get("metrics")
    if not _is_nonempty_list(metrics):
        errors.append("metrics: expected a non-empty array")
    else:
        for index, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                errors.append(f"metrics[{index}]: expected an object")
                continue
            if not _is_nonempty_string(metric.get("name")):
                errors.append(f"metrics[{index}].name: expected a non-empty string")
            if not isinstance(metric.get("higher_is_better"), bool):
                errors.append(f"metrics[{index}].higher_is_better: expected a boolean")


def _check_dataset(data: dict[str, Any], errors: list[str]) -> None:
    huggingface = _require_mapping(
        data, "huggingface", ("repo_id", "revision"), errors
    )
    if huggingface:
        repo_id = huggingface.get("repo_id")
        if not _is_nonempty_string(repo_id) or repo_id.count("/") != 1:
            errors.append("huggingface.repo_id: expected namespace/repository")
        revision = huggingface.get("revision")
        if not _is_nonempty_string(revision) or revision in {"main", "latest"}:
            errors.append("huggingface.revision: use an immutable commit revision")

    for field in ("modalities", "formats"):
        if not _is_nonempty_list(data.get(field)):
            errors.append(f"{field}: expected a non-empty array")
    if not isinstance(data.get("splits"), dict) or not data.get("splits"):
        errors.append("splits: expected a non-empty object")
    if data.get("access") not in ("public", "gated", "private"):
        errors.append("access: expected public, gated, or private")
    if not isinstance(data.get("contains_pii"), bool):
        errors.append("contains_pii: expected a boolean")


def _check_result(data: dict[str, Any], errors: list[str]) -> None:
    _check_track(data.get("track"), "track", errors)
    if data.get("status") not in ("unverified", "verified", "rejected"):
        errors.append("status: expected unverified, verified, or rejected")

    benchmark = _require_mapping(data, "benchmark", ("id", "version"), errors)
    if benchmark and not SEMVER.fullmatch(str(benchmark.get("version", ""))):
        errors.append("benchmark.version: expected semantic version")
    if benchmark and not _is_nonempty_string(benchmark.get("id")):
        errors.append("benchmark.id: expected a non-empty string")
    model = _require_mapping(data, "model", ("id", "revision"), errors)
    for field in ("id", "revision"):
        if model and not _is_nonempty_string(model.get(field)):
            errors.append(f"model.{field}: expected a non-empty string")
    if not _is_nonempty_string(data.get("method_id")):
        errors.append("method_id: expected a non-empty string")
    metrics = _require_mapping(data, "metrics", ("baseline", "final"), errors)
    if metrics:
        for phase in ("baseline", "final"):
            if not isinstance(metrics.get(phase), dict) or not metrics.get(phase):
                errors.append(f"metrics.{phase}: expected a non-empty object")
            else:
                score = metrics[phase].get("primary_score")
                if type(score) not in (int, float) or not math.isfinite(score):
                    errors.append(f"metrics.{phase}.primary_score: expected a finite number")
    settings = _require_mapping(
        data,
        "settings",
        ("seeds", "iterations", "compute_budget", "human_intervention"),
        errors,
    )
    if settings:
        seeds = settings.get("seeds")
        if not isinstance(seeds, list) or any(type(seed) is not int for seed in seeds):
            errors.append("settings.seeds: expected an array of integers (empty allowed)")
        iterations = settings.get("iterations")
        if type(iterations) is not int or iterations < 1:
            errors.append("settings.iterations: expected a positive integer")
    reproducibility = _require_mapping(
        data,
        "reproducibility",
        ("code_revision", "dataset_revision", "container_image", "command"),
        errors,
    )
    if reproducibility:
        image = reproducibility.get("container_image")
        if not _is_nonempty_string(image) or image.endswith(":latest"):
            errors.append("reproducibility.container_image: use a version or digest")
        for field in ("code_revision", "dataset_revision", "command"):
            if not _is_nonempty_string(reproducibility.get(field)):
                errors.append(f"reproducibility.{field}: expected a non-empty string")


def _check_method(data: dict[str, Any], errors: list[str]) -> None:
    tracks = data.get("tracks")
    if not _is_nonempty_list(tracks):
        errors.append("tracks: expected a non-empty array")
    else:
        for index, track in enumerate(tracks):
            _check_track(track, f"tracks[{index}]", errors)
    if not _is_nonempty_list(data.get("modifies")):
        errors.append("modifies: expected a non-empty array")
    for field in (
        "entrypoint",
        "feedback_signal",
        "stopping_rule",
        "human_intervention",
    ):
        if not _is_nonempty_string(data.get(field)):
            errors.append(f"{field}: expected a non-empty string")


KIND_CHECKS = {
    "benchmark": _check_benchmark,
    "dataset": _check_dataset,
    "result": _check_result,
    "method": _check_method,
}


def validate_manifest(
    data: Any, spec: ManifestSpec, expected_id: str
) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["document root: expected an object"]

    _check_common(data, spec, errors)
    manifest_id = data.get(spec.id_field)
    if _is_nonempty_string(manifest_id) and manifest_id != expected_id:
        errors.append(
            f"{spec.id_field}: {manifest_id!r} must match directory {expected_id!r}"
        )
    KIND_CHECKS[spec.kind](data, errors)
    return errors


def validate_repository(root: Path) -> list[str]:
    errors: list[str] = []
    manifests: dict[str, dict[str, dict[str, Any]]] = {
        spec.kind: {} for spec in SPECS
    }

    for spec in SPECS:
        for path in sorted(root.glob(spec.pattern)):
            relative = path.relative_to(root)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                errors.append(f"{relative}: cannot read valid UTF-8 JSON: {exc}")
                continue

            file_errors = validate_manifest(data, spec, path.parent.name)
            errors.extend(f"{relative}: {error}" for error in file_errors)
            if isinstance(data, dict) and _is_nonempty_string(data.get(spec.id_field)):
                manifest_id = data[spec.id_field]
                if manifest_id in manifests[spec.kind]:
                    errors.append(
                        f"{relative}: duplicate {spec.kind} ID {manifest_id!r}"
                    )
                manifests[spec.kind][manifest_id] = data

    dataset_ids = set(manifests["dataset"])
    for benchmark_id, benchmark in manifests["benchmark"].items():
        dataset = benchmark.get("dataset")
        registry_id = dataset.get("registry_id") if isinstance(dataset, dict) else None
        if not isinstance(registry_id, str) or registry_id not in dataset_ids:
            errors.append(
                f"benchmark {benchmark_id!r}: dataset registry ID {registry_id!r} "
                "is not registered under datasets/"
            )

    benchmark_ids = set(manifests["benchmark"])
    method_ids = set(manifests["method"])
    for submission_id, result in manifests["result"].items():
        benchmark = result.get("benchmark")
        benchmark_id = benchmark.get("id") if isinstance(benchmark, dict) else None
        if not isinstance(benchmark_id, str) or benchmark_id not in benchmark_ids:
            errors.append(
                f"result {submission_id!r}: benchmark ID {benchmark_id!r} is not registered"
            )
        else:
            registered = manifests["benchmark"][benchmark_id]
            if benchmark.get("version") != registered.get("version"):
                errors.append(f"result {submission_id!r}: benchmark version does not match registry")
            if result.get("track") != registered.get("track"):
                errors.append(f"result {submission_id!r}: track does not match benchmark")
        method_id = result.get("method_id")
        if not isinstance(method_id, str) or method_id not in method_ids:
            errors.append(
                f"result {submission_id!r}: method ID {method_id!r} is not registered"
            )
        else:
            tracks = manifests["method"][method_id].get("tracks")
            if isinstance(tracks, list) and result.get("track") not in tracks:
                errors.append(f"result {submission_id!r}: track is not supported by method")

    return errors


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        type=Path,
        help="DataLite-RSI repository root (default: current directory)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = args.root.resolve()
    errors = validate_repository(root)
    if errors:
        print(f"DataLite-RSI validation failed with {len(errors)} error(s):")
        for error in errors:
            print(f"- {error}")
        return 1
    print("DataLite-RSI contribution manifests are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
