"""Task, structured-record and artifact contracts for the DataFlow runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


METHOD_IDS = frozenset({"dataflow-evolver"})
MODALITIES = frozenset({"text", "image", "video"})
ARTIFACT_KINDS = frozenset({"image", "video_reference", "blob", "binary"})


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OutputContract:
    required_keys: tuple[str, ...]
    artifact_types: tuple[str, ...] = ()
    format: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OutputContract":
        keys = value.get("required_keys", ())
        types = value.get("artifact_types", ())
        if not isinstance(keys, (list, tuple)) or not keys:
            raise ValueError("output.required_keys must be a nonempty list")
        if not isinstance(types, (list, tuple)):
            raise ValueError("output.artifact_types must be a list")
        normalized = tuple(str(key) for key in keys)
        if any(not key for key in normalized) or len(set(normalized)) != len(normalized):
            raise ValueError("output.required_keys contains an empty or duplicate key")
        return cls(normalized, tuple(str(item) for item in types), value.get("format"))

    def validate(self, output: Mapping[str, Any]) -> None:
        missing = set(self.required_keys) - set(output)
        if missing:
            raise ValueError(f"output contract is missing keys {sorted(missing)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_keys": list(self.required_keys),
            "artifact_types": list(self.artifact_types),
            "format": self.format,
        }


@dataclass(frozen=True)
class ArtifactRef:
    kind: str
    uri: str
    media_type: str
    sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArtifactRef":
        if not isinstance(value, Mapping):
            raise ValueError("artifact reference must be an object")
        kind = str(value.get("kind") or "")
        uri = str(value.get("uri") or "")
        media_type = str(value.get("media_type") or "")
        metadata = value.get("metadata", {})
        if kind not in ARTIFACT_KINDS or not uri or not media_type:
            raise ValueError("artifact reference needs a valid kind, uri and media_type")
        if not isinstance(metadata, Mapping):
            raise ValueError("artifact reference metadata must be an object")
        if kind == "image" and not media_type.startswith("image/"):
            raise ValueError("image reference needs an image/* media_type")
        if kind == "video_reference" and not media_type.startswith("video/"):
            raise ValueError("video reference needs a video/* media_type")
        digest = value.get("sha256")
        return cls(kind, uri, media_type, str(digest) if digest else None, dict(metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "uri": self.uri,
            "media_type": self.media_type,
            "sha256": self.sha256,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class InputContract:
    """Declare modalities and reference fields in a fixed structured corpus.

    A row can contain several modalities. The media stay outside JSONL; rows
    carry references that operators resolve through their own backends.
    """

    modalities: tuple[str, ...]
    required_fields: tuple[str, ...] = ()
    artifact_fields: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "InputContract":
        if not isinstance(value, Mapping):
            raise ValueError("input_contract must be an object")
        raw_modalities = value.get("modalities")
        raw_required = value.get("required_fields", ())
        raw_artifacts = value.get("artifact_fields", {})
        if not isinstance(raw_modalities, (list, tuple)) or not raw_modalities:
            raise ValueError("input_contract.modalities must be a nonempty list")
        if not isinstance(raw_required, (list, tuple)):
            raise ValueError("input_contract.required_fields must be a list")
        if not isinstance(raw_artifacts, Mapping):
            raise ValueError("input_contract.artifact_fields must be an object")
        modalities = tuple(str(item) for item in raw_modalities)
        required = tuple(str(item) for item in raw_required)
        artifacts = {str(key): str(kind) for key, kind in raw_artifacts.items()}
        if len(set(modalities)) != len(modalities) or set(modalities) - MODALITIES:
            raise ValueError(f"unsupported or duplicate modalities: {modalities}")
        if any(not item for item in required) or len(set(required)) != len(required):
            raise ValueError("input_contract.required_fields contains an empty or duplicate key")
        if any(not key or kind not in ARTIFACT_KINDS for key, kind in artifacts.items()):
            raise ValueError("input_contract.artifact_fields has an invalid field or kind")
        if any(kind == "image" for kind in artifacts.values()) and "image" not in modalities:
            raise ValueError("image artifact fields require the image modality")
        if any(kind == "video_reference" for kind in artifacts.values()) and "video" not in modalities:
            raise ValueError("video artifact fields require the video modality")
        return cls(modalities, required, artifacts)

    def validate_record(self, record: Mapping[str, Any], *, row_number: int = 1) -> None:
        if not isinstance(record, Mapping):
            raise ValueError(f"input row {row_number} must be an object")
        missing = [field for field in self.required_fields if record.get(field) in (None, "")]
        if missing:
            raise ValueError(f"input row {row_number} is missing required fields {missing}")
        for field, expected_kind in self.artifact_fields.items():
            if field not in record or record[field] is None:
                continue
            try:
                actual = ArtifactRef.from_mapping(record[field])
            except ValueError as exc:
                raise ValueError(f"input row {row_number} field {field!r}: {exc}") from exc
            if actual.kind != expected_kind:
                raise ValueError(
                    f"input row {row_number} field {field!r} must be {expected_kind!r}"
                )

    def validate_entry(self, path: str | Path) -> int:
        """Stream an input manifest without loading binary media into memory."""
        entry = Path(path)
        if entry.suffix.lower() != ".jsonl":
            raise ValueError("an explicit input_contract requires a JSONL entry manifest")
        rows = 0
        with entry.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"input row {number} is invalid JSON") from exc
                self.validate_record(record, row_number=number)
                rows += 1
        if not rows:
            raise ValueError("input manifest contains no records")
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "modalities": list(self.modalities),
            "required_fields": list(self.required_fields),
            "artifact_fields": dict(self.artifact_fields),
        }


@dataclass(frozen=True)
class TaskEnvelope:
    task_id: str
    method_id: str
    objective: str
    data: Mapping[str, Any]
    output_contract: OutputContract
    input_contract: InputContract | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    workspace: str = "outputs/rsi"
    provider: str = "offline"
    role: str = "pipeline_builder"
    schema_version: str = "0.1"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TaskEnvelope":
        if not isinstance(value, Mapping):
            raise TypeError("task config must be a mapping")
        if "method_id" not in value or not str(value["method_id"]).strip():
            raise ValueError("task config requires an explicit method_id")
        method_id = str(value["method_id"]).strip()
        if method_id not in METHOD_IDS:
            raise ValueError(f"unknown method_id {method_id!r}; expected {sorted(METHOD_IDS)}")
        for key in ("task_id", "objective"):
            if not str(value.get(key, "")).strip():
                raise ValueError(f"task config requires {key}")
        data = value.get("data")
        output = value.get("output", value.get("output_contract"))
        metadata = value.get("metadata", {})
        input_contract = value.get("input_contract")
        if not isinstance(data, Mapping):
            raise ValueError("task config data must be an object")
        if not isinstance(output, Mapping):
            raise ValueError("task config output must be an object")
        if not isinstance(metadata, Mapping):
            raise ValueError("task config metadata must be an object")
        if value.get("schema_version", "0.1") != "0.1":
            raise ValueError("unsupported task schema_version")
        return cls(
            task_id=str(value["task_id"]),
            method_id=method_id,
            objective=str(value["objective"]),
            data=dict(data),
            output_contract=OutputContract.from_mapping(output),
            input_contract=(
                InputContract.from_mapping(input_contract)
                if input_contract is not None else None
            ),
            metadata=dict(metadata),
            workspace=str(value.get("workspace") or "outputs/rsi"),
            provider=str(value.get("provider") or "offline").lower(),
            role=str(value.get("role") or "pipeline_builder"),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "method_id": self.method_id,
            "objective": self.objective,
            "data": dict(self.data),
            "output": self.output_contract.to_dict(),
            "metadata": dict(self.metadata),
            "workspace": self.workspace,
            "provider": self.provider,
            "role": self.role,
        }
        if self.input_contract is not None:
            result["input_contract"] = self.input_contract.to_dict()
        return result

    def resolved_workspace(self, repository_root: Path) -> Path:
        path = Path(self.workspace).expanduser()
        return (path if path.is_absolute() else repository_root / path).resolve()
