from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from rsi.framework.evolution.models import PipelineConfig

OBSERVATION_FILENAME = "execution_observation.json"
DEFAULT_SAMPLE_ROWS = 2
DEFAULT_MAX_SAMPLE_CHARS = 4000
MAX_DISTRIBUTION_FIELDS = 40
MAX_DISTRIBUTION_VALUES = 20
MAX_DISTRIBUTION_CANDIDATES = 256
MAX_DISTRIBUTION_VALUE_CHARS = 160
MAX_EMBEDDED_JSON_CHARS = 64 * 1024
MAX_TOP_LEVEL_FIELDS = 256


def write_execution_observation(
    iteration_dir: Path,
    *,
    config: PipelineConfig,
    raw_entry_path: str,
    final_dataset_path: str | None,
    status: str,
    failure_stage: str | None = None,
    error: str | None = None,
    sample_rows_per_step: int = DEFAULT_SAMPLE_ROWS,
    max_sample_chars: int = DEFAULT_MAX_SAMPLE_CHARS,
) -> Path:
    """Write a task-agnostic, framework-owned execution receipt."""
    iteration_dir = Path(iteration_dir).resolve()
    manifest_path = iteration_dir / "step_manifest.json"
    manifest = _read_object(manifest_path)
    raw_path = Path(raw_entry_path).resolve()
    raw_rows = _count_jsonl(raw_path)
    declared = [_operator_name(item) for item in config.operators]
    steps = []
    observed_names = []
    previous_rows = raw_rows
    last_artifact: Path | None = None
    last_inspection: dict[str, Any] | None = None
    for index, item in enumerate(manifest.get("steps", [])):
        artifact = Path(str(item.get("output_artifact", "")))
        if not artifact.is_absolute():
            artifact = (iteration_dir / artifact).resolve()
        manifest_rows = _int_or_none(item.get("output_rows"))
        manifest_fields = [
            str(value) for value in (item.get("output_fields") or [])
        ]
        artifact_exists = artifact.is_file()
        inspection = (
            _inspect_jsonl(
                artifact,
                sample_rows_per_step,
                max_sample_chars,
            )
            if artifact_exists else None
        )
        last_artifact = artifact
        last_inspection = inspection
        output_rows = inspection["rows"] if inspection is not None else manifest_rows
        output_rows = int(output_rows or 0)
        output_fields = (
            inspection["fields"] if inspection is not None else manifest_fields
        )
        name = str(item.get("operator_name", "?"))
        observed_names.append(name)
        retention = _ratio(output_rows, previous_rows)
        steps.append(
            {
                "index": int(item.get("logical_index", index)),
                "operator_name": name,
                "reused": bool(item.get("reused", False)),
                "input_rows": previous_rows,
                "output_rows": output_rows,
                "retention_rate": retention,
                "output_fields": output_fields,
                "manifest_output_rows": manifest_rows,
                "manifest_output_fields": manifest_fields,
                "artifact": str(artifact),
                "artifact_exists": artifact_exists,
                "field_distributions": (
                    inspection["field_distributions"] if inspection is not None else {}
                ),
                "sample_rows": inspection["sample_rows"] if inspection is not None else [],
            }
        )
        previous_rows = output_rows
    final_path = Path(final_dataset_path).resolve() if final_dataset_path else None
    if final_path and final_path.is_file():
        final_inspection = (
            last_inspection
            if final_path == last_artifact and last_inspection is not None
            else _inspect_jsonl(final_path, sample_rows_per_step, max_sample_chars)
        )
    else:
        final_inspection = None
    final_rows = final_inspection["rows"] if final_inspection is not None else None
    final_fields = final_inspection["fields"] if final_inspection is not None else []
    mismatches = []
    if declared and observed_names and declared != observed_names:
        mismatches.append(
            {
                "kind": "operator_order_mismatch",
                "declared": declared,
                "observed": observed_names,
            }
        )
    if manifest_path.exists() and len(declared) != len(observed_names):
        mismatches.append(
            {
                "kind": "operator_count_mismatch",
                "declared": len(declared),
                "observed": len(observed_names),
            }
        )
    warnings = []
    for step in steps:
        if step["input_rows"] and step["retention_rate"] is not None:
            if step["retention_rate"] < 0.5:
                warnings.append(
                    {
                        "kind": "large_row_drop",
                        "step": step["index"],
                        "operator_name": step["operator_name"],
                        "input_rows": step["input_rows"],
                        "output_rows": step["output_rows"],
                        "retention_rate": step["retention_rate"],
                    }
                )
        if (
            step["manifest_output_rows"] is not None
            and step["manifest_output_rows"] != step["output_rows"]
        ):
            warnings.append(
                {
                    "kind": "manifest_actual_row_mismatch",
                    "step": step["index"],
                    "operator_name": step["operator_name"],
                    "manifest_rows": step["manifest_output_rows"],
                    "actual_rows": step["output_rows"],
                }
            )
        if (
            step["manifest_output_fields"]
            and sorted(step["manifest_output_fields"]) != sorted(step["output_fields"])
        ):
            warnings.append(
                {
                    "kind": "manifest_actual_field_mismatch",
                    "step": step["index"],
                    "operator_name": step["operator_name"],
                    "manifest_fields": step["manifest_output_fields"],
                    "actual_fields": step["output_fields"],
                }
            )
        if not step["artifact_exists"]:
            warnings.append(
                {
                    "kind": "missing_step_artifact",
                    "step": step["index"],
                    "operator_name": step["operator_name"],
                    "artifact": step["artifact"],
                }
            )
        if step["output_rows"] == 0:
            warnings.append(
                {
                    "kind": "empty_step_output",
                    "step": step["index"],
                    "operator_name": step["operator_name"],
                }
            )
    payload: dict[str, Any] = {
        "schema_version": 2,
        "status": status,
        "failure_stage": failure_stage,
        "error": error,
        "raw_input": {"path": str(raw_path), "rows": raw_rows},
        "declared_operators": [
            {
                "index": index,
                "name": name,
                "declaration": _jsonable(config.operators[index]),
            }
            for index, name in enumerate(declared)
        ],
        "observed_steps": steps,
        "final_output": {
            "path": str(final_path) if final_path else None,
            "rows": final_rows,
            "fields": final_fields,
            "exists": bool(final_path and final_path.is_file()),
            "nonempty": bool(final_path and final_rows),
            "field_distributions": (
                final_inspection["field_distributions"]
                if final_inspection is not None else {}
            ),
            "sample_rows": (
                final_inspection["sample_rows"] if final_inspection is not None else []
            ),
        },
        "contract_checks": {
            "step_manifest_present": manifest_path.is_file(),
            "declared_observed_operator_order_match": not any(
                item["kind"] == "operator_order_mismatch" for item in mismatches
            ),
            "declared_observed_operator_count_match": not any(
                item["kind"] == "operator_count_mismatch" for item in mismatches
            ),
            "final_output_found": bool(final_path and final_path.is_file()),
        },
        "mismatches": mismatches,
        "warnings": warnings,
        "artifacts": {
            "step_manifest": str(manifest_path),
            "final_dataset": str(final_path) if final_path else None,
        },
    }
    path = iteration_dir / OBSERVATION_FILENAME
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _jsonable(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return str(value)


def _operator_name(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("name", "?") or "?")
    return str(item)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _count_jsonl(path: Path | None) -> int:
    if path is None or not path.is_file():
        return 0
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            return sum(bool(line.strip()) for line in handle)
    except OSError:
        return 0


def _inspect_jsonl(path: Path, sample_limit: int, max_sample_chars: int) -> dict[str, Any]:
    """Inspect one artifact in a single streaming pass with bounded summaries."""
    rows = 0
    field_names: set[str] = set()
    samples: list[dict[str, Any]] = []
    counters: dict[str, Counter[str]] = {}
    present: Counter[str] = Counter()
    high_cardinality: set[str] = set()
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows += 1
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    for key in value:
                        if len(field_names) >= MAX_TOP_LEVEL_FIELDS:
                            break
                        field_names.add(str(key))
                    if len(samples) < sample_limit:
                        samples.append(_bounded_sample(value, max_sample_chars))
                    for field_path, scalar in _iter_scalar_fields(value):
                        if field_path in high_cardinality:
                            continue
                        if (
                            field_path not in counters
                            and len(counters) + len(high_cardinality)
                            >= MAX_DISTRIBUTION_CANDIDATES
                        ):
                            continue
                        counter = counters.setdefault(field_path, Counter())
                        counter[_distribution_key(scalar)] += 1
                        present[field_path] += 1
                        if len(counter) > MAX_DISTRIBUTION_VALUES:
                            counters.pop(field_path, None)
                            present.pop(field_path, None)
                            high_cardinality.add(field_path)
    except OSError:
        pass
    ranked_paths = sorted(
        counters,
        key=lambda field_path: (
            -present[field_path],
            -len(counters[field_path]),
            field_path,
        ),
    )[:MAX_DISTRIBUTION_FIELDS]
    distributions = {}
    for field_path in ranked_paths:
        counts = counters[field_path]
        present_rows = present[field_path]
        distributions[field_path] = {
            "present_rows": present_rows,
            "missing_rows": max(0, rows - present_rows),
            "coverage_rate": _ratio(present_rows, rows),
            "distinct_values": len(counts),
            "value_counts": dict(
                sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            ),
        }
    return {
        "rows": rows,
        "fields": sorted(field_names),
        "field_distributions": distributions,
        "sample_rows": samples,
    }


def _iter_scalar_fields(
    value: dict[str, Any],
    prefix: str = "",
    depth: int = 0,
) -> list[tuple[str, Any]]:
    output: list[tuple[str, Any]] = []
    if depth >= 3:
        return output
    for raw_key, item in value.items():
        key = str(raw_key)
        field_path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            output.extend(_iter_scalar_fields(item, field_path, depth + 1))
            continue
        if isinstance(item, list):
            continue
        if isinstance(item, str):
            parsed = _embedded_json_object(item)
            if parsed is not None:
                output.extend(_iter_scalar_fields(parsed, field_path, depth + 1))
                continue
            if len(item) > MAX_DISTRIBUTION_VALUE_CHARS:
                continue
        output.append((field_path, item))
    return output


def _embedded_json_object(value: str) -> dict[str, Any] | None:
    stripped = value.strip()
    if (
        not stripped.startswith("{")
        or not stripped.endswith("}")
        or len(stripped) > MAX_EMBEDDED_JSON_CHARS
    ):
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _distribution_key(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _bounded_sample(value: dict[str, Any], max_chars: int) -> dict[str, Any]:
    encoded = json.dumps(value, ensure_ascii=False)
    if len(encoded) <= max_chars:
        return value
    return {"_truncated_record": encoded[: max_chars - 1] + "…"}


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    value = numerator / denominator
    return round(value, 6) if math.isfinite(value) else None
