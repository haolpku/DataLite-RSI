from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .schema import (
    CompatibilityDefinition,
    EditDefinition,
    EnvironmentSnapshot,
    SceneDefinition,
)


T = TypeVar("T", bound=BaseModel)


def _load_jsonl(path: Path, model: type[T]) -> list[T]:
    records: list[T] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            records.append(model.model_validate_json(line))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return records


def load_environment_snapshot(
    snapshot_path: Path,
) -> tuple[
    EnvironmentSnapshot,
    list[SceneDefinition],
    list[EditDefinition],
    list[CompatibilityDefinition],
]:
    snapshot_path = snapshot_path.resolve()
    snapshot = EnvironmentSnapshot.model_validate_json(
        snapshot_path.read_text(encoding="utf-8")
    )
    base = snapshot_path.parent
    scenes = _load_jsonl(base / snapshot.scene_registry, SceneDefinition)
    edits = _load_jsonl(base / snapshot.edit_registry, EditDefinition)
    compatibility = _load_jsonl(
        base / snapshot.compatibility_registry, CompatibilityDefinition
    )

    scene_by_ref = {record.ref: record for record in scenes}
    edit_by_ref = {record.ref: record for record in edits}
    compatibility_by_ref = {record.ref: record for record in compatibility}

    missing_scenes = set(snapshot.scene_refs) - set(scene_by_ref)
    missing_edits = set(snapshot.edit_refs) - set(edit_by_ref)
    missing_compatibility = set(snapshot.compatibility_refs) - set(
        compatibility_by_ref
    )
    if missing_scenes or missing_edits or missing_compatibility:
        raise ValueError(
            "snapshot references missing records: "
            f"scenes={sorted(missing_scenes)}, edits={sorted(missing_edits)}, "
            f"compatibility={sorted(missing_compatibility)}"
        )

    return (
        snapshot,
        [scene_by_ref[ref] for ref in snapshot.scene_refs],
        [edit_by_ref[ref] for ref in snapshot.edit_refs],
        [compatibility_by_ref[ref] for ref in snapshot.compatibility_refs],
    )


def write_jsonl(path: Path, records: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json(exclude_none=True) + "\n")


def write_snapshot(path: Path, snapshot: EnvironmentSnapshot) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
